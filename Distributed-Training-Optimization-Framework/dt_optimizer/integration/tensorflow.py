import tensorflow as tf
import horovod.tensorflow as hvd
from typing import Optional, List, Any, Union, Callable
import os

class TensorFlowIntegration:
    def __init__(self, seed: int = 42) -> None:
        self.strategy: tf.distribute.Strategy = self.setup_distributed_training()
        self.set_random_seed(seed)

    def setup_distributed_training(self) -> tf.distribute.Strategy:
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()

    @staticmethod
    def set_random_seed(seed: int) -> None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def compile_model(
        self, 
        model: tf.keras.Model, 
        optimizer: Union[str, tf.keras.optimizers.Optimizer],
        loss_fn: Union[str, tf.keras.losses.Loss], 
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None
    ) -> tf.keras.Model:
        with self.strategy.scope():
            if isinstance(optimizer, str):
                optimizer = tf.keras.optimizers.get(optimizer)
            optimizer = hvd.DistributedOptimizer(optimizer)

            if isinstance(loss_fn, str):
                loss_fn = tf.keras.losses.get(loss_fn)

            if metrics:
                metrics = [tf.keras.metrics.get(m) if isinstance(m, str) else m for m in metrics]

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        return model

    def prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = True
    ) -> tf.data.Dataset:
        if shuffle:
            dataset = dataset.shuffle(10000)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def train_model(
        self, 
        model: tf.keras.Model, 
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10, 
        steps_per_epoch: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        callbacks = callbacks or []
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1))
        
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

        return model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=1 if hvd.rank() == 0 else 0
        )

    def evaluate_model(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        steps: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> Any:
        return model.evaluate(dataset, steps=steps, callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)

    @staticmethod
    def create_lr_schedule(
        initial_lr: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False
    ) -> Callable[[int], float]:
        return lambda epoch: initial_lr * decay_rate ** (epoch // decay_steps if staircase else epoch / decay_steps)

    def save_model(self, model: tf.keras.Model, filepath: str) -> None:
        if hvd.rank() == 0:
            model.save(filepath)

    def load_model(self, filepath: str) -> tf.keras.Model:
        with self.strategy.scope():
            return tf.keras.models.load_model(filepath)

# Example usage
if __name__ == "__main__":
    tf_integration = TensorFlowIntegration()

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compiled_model = tf_integration.compile_model(
        model,
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Create dummy data
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal((1000, 10)), tf.random.normal((1000, 1)))
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal((200, 10)), tf.random.normal((200, 1)))
    )

    # Prepare datasets for distributed training
    train_dataset = tf_integration.prepare_dataset(train_dataset, batch_size=32)
    val_dataset = tf_integration.prepare_dataset(val_dataset, batch_size=32, shuffle=False, repeat=False)

    # Create learning rate schedule
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        tf_integration.create_lr_schedule(initial_lr=0.1, decay_steps=100, decay_rate=0.9)
    )

    # Train the model
    history = tf_integration.train_model(
        compiled_model,
        train_dataset,
        validation_dataset=val_dataset,
        epochs=5,
        steps_per_epoch=100,
        callbacks=[lr_schedule]
    )

    # Evaluate the model
    evaluation = tf_integration.evaluate_model(compiled_model, val_dataset)

    if hvd.rank() == 0:
        print(f"Evaluation results: {evaluation}")

    # Save the model
    tf_integration.save_model(compiled_model, './my_model.h5')

    # Load the model
    loaded_model = tf_integration.load_model('./my_model.h5')