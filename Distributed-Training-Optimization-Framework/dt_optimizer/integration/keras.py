from __future__ import annotations

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from typing import Optional, List, Callable, Dict, Any, Protocol, Union
from dataclasses import dataclass, field
import logging
from functools import partial
from contextlib import contextmanager
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizerProtocol(Protocol):
    def apply_gradients(self, grads_and_vars: List[tuple[Any, Any]], name: Optional[str] = None) -> None: ...

class LossProtocol(Protocol):
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: ...

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    use_mixed_precision: bool = False
    learning_rate: float = 0.001
    callbacks: List[tf.keras.callbacks.Callback] = field(default_factory=list)
    custom_callbacks: List[Callable[[int, Dict[str, float]], None]] = field(default_factory=list)

class KerasIntegration:
    def __init__(self) -> None:
        self.strategy: tf.distribute.Strategy = self._setup_distributed_strategy()
        self.setup_environment()

    @staticmethod
    def _setup_distributed_strategy() -> tf.distribute.Strategy:
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        return tf.distribute.experimental.MultiWorkerMirroredStrategy()

    @staticmethod
    def setup_environment() -> None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        tf.keras.backend.set_floatx('float32')

    @contextmanager
    def strategy_scope(self):
        with self.strategy.scope():
            yield

    def prepare_model(self, model: tf.keras.Model) -> tf.keras.Model:
        with self.strategy_scope():
            return model

    def prepare_optimizer(self, optimizer: Union[OptimizerProtocol, str], learning_rate: float) -> OptimizerProtocol:
        with self.strategy_scope():
            if isinstance(optimizer, str):
                optimizer = tf.keras.optimizers.get(optimizer)
            optimizer = optimizer(learning_rate=learning_rate * hvd.size())
            return hvd.DistributedOptimizer(optimizer)

    def prepare_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
        return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    @tf.function
    def train_step(
        self, 
        model: tf.keras.Model, 
        inputs: tf.Tensor, 
        labels: tf.Tensor, 
        optimizer: OptimizerProtocol, 
        loss_fn: LossProtocol,
        train_loss: tf.keras.metrics.Mean,
        train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy
    ) -> None:
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    def distributed_train(
        self, 
        model: tf.keras.Model, 
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset],
        optimizer: Union[OptimizerProtocol, str],
        loss_fn: Union[LossProtocol, str],
        config: TrainingConfig
    ) -> Dict[str, List[float]]:
        model = self.prepare_model(model)
        optimizer = self.prepare_optimizer(optimizer, config.learning_rate)
        train_dataset = self.prepare_dataset(train_dataset, config.batch_size)
        val_dataset = self.prepare_dataset(val_dataset, config.batch_size) if val_dataset else None

        if config.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        if isinstance(loss_fn, str):
            loss_fn = tf.keras.losses.get(loss_fn)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        @tf.function
        def distributed_train_step(inputs, labels):
            self.strategy.run(
                partial(self.train_step, model, inputs, labels, optimizer, loss_fn, train_loss, train_accuracy)
            )

        @tf.function
        def distributed_val_step(inputs, labels):
            predictions = model(inputs, training=False)
            v_loss = loss_fn(labels, predictions)
            val_loss(v_loss)
            val_accuracy(labels, predictions)

        history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(config.epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

            for batch in train_dataset:
                distributed_train_step(batch['inputs'], batch['labels'])

            if val_dataset:
                for batch in val_dataset:
                    distributed_val_step(batch['inputs'], batch['labels'])

            if hvd.rank() == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs}, "
                    f"Loss: {train_loss.result():.4f}, "
                    f"Accuracy: {train_accuracy.result():.4f}, "
                    f"Val Loss: {val_loss.result():.4f}, "
                    f"Val Accuracy: {val_accuracy.result():.4f}"
                )

            history['train_loss'].append(train_loss.result().numpy())
            history['train_accuracy'].append(train_accuracy.result().numpy())
            history['val_loss'].append(val_loss.result().numpy())
            history['val_accuracy'].append(val_accuracy.result().numpy())

            for callback in config.callbacks:
                callback.on_epoch_end(epoch, logs=history)

            for custom_callback in config.custom_callbacks:
                custom_callback(epoch, {
                    'loss': train_loss.result().numpy(),
                    'accuracy': train_accuracy.result().numpy(),
                    'val_loss': val_loss.result().numpy(),
                    'val_accuracy': val_accuracy.result().numpy()
                })

        return history

    @staticmethod
    def save_model(model: tf.keras.Model, filepath: str) -> None:
        if hvd.rank() == 0:
            model.save(filepath)

    @staticmethod
    def load_model(filepath: str) -> tf.keras.Model:
        return tf.keras.models.load_model(filepath)

# Example usage
if __name__ == "__main__":
    integration = KerasIntegration()

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Create dummy datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': tf.random.normal((1000, 10)),
        'labels': tf.random.uniform((1000, 1))
    })
    val_dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': tf.random.normal((200, 10)),
        'labels': tf.random.uniform((200, 1))
    })

    # Configure training
    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        use_mixed_precision=True,
        learning_rate=0.001,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ],
        custom_callbacks=[lambda epoch, logs: print(f"Custom log - Epoch {epoch}: {logs}")]
    )

    # Run distributed training
    history = integration.distributed_train(
        model,
        train_dataset,
        val_dataset,
        optimizer='adam',
        loss_fn='mse',
        config=config
    )

    print("Training completed. Final metrics:", history)

    # Save the model
    integration.save_model(model, './my_keras_model')

    # Load the model
    loaded_model = integration.load_model('./my_keras_model')