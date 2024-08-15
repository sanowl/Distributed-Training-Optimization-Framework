import tensorflow as tf
import torch
import horovod.tensorflow as hvd_tf
import horovod.torch as hvd_torch
from typing import Union, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import os

class DistributedTrainerBase(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def prepare_model(self, model):
        pass

    @abstractmethod
    def prepare_optimizer(self, optimizer):
        pass

    @abstractmethod
    def prepare_data(self, dataset):
        pass

    @abstractmethod
    def train_step(self, model, inputs, labels, optimizer, loss_fn):
        pass

    @abstractmethod
    def distributed_train(self, model, dataset, optimizer, loss_fn, epochs, callbacks=None):
        pass

class TensorFlowTrainer(DistributedTrainerBase):
    def setup(self):
        hvd_tf.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd_tf.local_rank()], 'GPU')
        self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    def prepare_model(self, model):
        with self.strategy.scope():
            return model

    def prepare_optimizer(self, optimizer):
        optimizer = hvd_tf.DistributedOptimizer(optimizer)
        return optimizer

    def prepare_data(self, dataset):
        dataset = dataset.shard(num_shards=hvd_tf.size(), index=hvd_tf.rank())
        return dataset

    @tf.function
    def train_step(self, model, inputs, labels, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)
        tape = hvd_tf.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def distributed_train(self, model, dataset, optimizer, loss_fn, epochs, callbacks=None):
        model = self.prepare_model(model)
        optimizer = self.prepare_optimizer(optimizer)
        dataset = self.prepare_data(dataset)

        @tf.function
        def distributed_train_step(inputs, labels):
            per_replica_losses = self.strategy.run(self.train_step, args=(model, inputs, labels, optimizer, loss_fn))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in dataset:
                total_loss += distributed_train_step(batch['inputs'], batch['labels'])
                num_batches += 1
            
            average_loss = total_loss / num_batches
            if hvd_tf.rank() == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
            
            if callbacks:
                for callback in callbacks:
                    callback(epoch, {'loss': average_loss})

class PyTorchTrainer(DistributedTrainerBase):
    def setup(self):
        hvd_torch.init()
        torch.cuda.set_device(hvd_torch.local_rank())

    def prepare_model(self, model):
        model.cuda()
        return hvd_torch.DistributedDataParallel(model)

    def prepare_optimizer(self, optimizer):
        optimizer = hvd_torch.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd_torch.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd_torch.broadcast_optimizer_state(optimizer, root_rank=0)
        return optimizer

    def prepare_data(self, dataset):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=hvd_torch.size(), rank=hvd_torch.rank()
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=32, pin_memory=True
        )
        return dataloader

    def train_step(self, model, inputs, labels, optimizer, loss_fn):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def distributed_train(self, model, dataset, optimizer, loss_fn, epochs, callbacks=None):
        model = self.prepare_model(model)
        optimizer = self.prepare_optimizer(optimizer)
        dataloader = self.prepare_data(dataset)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()
                loss = self.train_step(model, inputs, labels, optimizer, loss_fn)
                total_loss += loss
                num_batches += 1
            
            average_loss = total_loss / num_batches
            if hvd_torch.rank() == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
            
            if callbacks:
                for callback in callbacks:
                    callback(epoch, {'loss': average_loss})

class UnifiedDistributedTrainer:
    def __init__(self, framework: str = "tensorflow"):
        self.framework = framework
        if framework == "tensorflow":
            self.trainer = TensorFlowTrainer()
        elif framework == "pytorch":
            self.trainer = PyTorchTrainer()
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        self.trainer.setup()

    def train(self, model: Union[tf.keras.Model, torch.nn.Module],
              dataset: Union[tf.data.Dataset, torch.utils.data.Dataset],
              optimizer: Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
              loss_fn: Union[tf.keras.losses.Loss, torch.nn.Module],
              epochs: int,
              callbacks: Optional[List[Callable[[int, Dict[str, float]], None]]] = None) -> None:
        self.trainer.distributed_train(model, dataset, optimizer, loss_fn, epochs, callbacks)

# Example usage
if __name__ == "__main__":
    # TensorFlow example
    tf_model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        ({"inputs": tf.random.normal((100, 5))}, {"labels": tf.random.uniform((100, 10))})
    ).batch(32)
    tf_optimizer = tf.keras.optimizers.Adam(0.01)
    tf_loss = tf.keras.losses.MeanSquaredError()

    tf_trainer = UnifiedDistributedTrainer("tensorflow")
    tf_trainer.train(tf_model, tf_dataset, tf_optimizer, tf_loss, epochs=5)

    # PyTorch example
    torch_model = torch.nn.Linear(5, 10)
    torch_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 5), torch.randint(0, 10, (100,))
    )
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    torch_loss = torch.nn.CrossEntropyLoss()

    torch_trainer = UnifiedDistributedTrainer("pytorch")
    torch_trainer.train(torch_model, torch_dataset, torch_optimizer, torch_loss, epochs=5)