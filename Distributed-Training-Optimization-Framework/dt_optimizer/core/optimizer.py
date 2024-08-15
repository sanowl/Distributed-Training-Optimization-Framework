import torch
import horovod.torch as hvd
import tensorflow as tf
import horovod.tensorflow as hvd_tf
from typing import Union, Optional, Any, Dict, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float = 0.0
    momentum: float = 0.0
    use_mixed_precision: bool = False
    gradient_clip_value: Optional[float] = None
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Optional[Dict[str, Any]] = None

class BaseDistributedOptimizer(ABC):
    @abstractmethod
    def prepare_optimizer(self, model: Any, optimizer: Any, config: OptimizerConfig) -> Any:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        pass

    @abstractmethod
    def log_optimizer_state(self) -> None:
        pass

    @abstractmethod
    def update_lr(self, new_lr: float) -> None:
        pass

class TensorFlowDistributedOptimizer(BaseDistributedOptimizer):
    def __init__(self):
        self.optimizer: Optional[tf.keras.optimizers.Optimizer] = None
        self.lr_scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None
        self.scaler: Optional[tf.mixed_precision.LossScaler] = None

    def prepare_optimizer(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, config: OptimizerConfig) -> tf.keras.optimizers.Optimizer:
        if config.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            self.scaler = tf.mixed_precision.LossScaler('dynamic')

        if config.lr_scheduler:
            self.lr_scheduler = self._get_lr_scheduler(config)
            optimizer.learning_rate = self.lr_scheduler

        self.optimizer = hvd_tf.DistributedOptimizer(
            optimizer,
            backward_passes_per_step=1,
            average_aggregated_gradients=True
        )

        return self.optimizer

    def step(self) -> None:
        if self.optimizer:
            self.optimizer.apply_gradients(self.gradients)

    def zero_grad(self) -> None:
        self.gradients = []

    @tf.function
    def compute_gradients(self, model: tf.keras.Model, loss: Callable, inputs: tf.Tensor, targets: tf.Tensor) -> None:
        with tf.GradientTape() as tape:
            if self.scaler:
                with self.scaler.scale_loss():
                    predictions = model(inputs, training=True)
                    loss_value = loss(targets, predictions)
            else:
                predictions = model(inputs, training=True)
                loss_value = loss(targets, predictions)

        if self.scaler:
            scaled_gradients = tape.gradient(self.scaler.scale(loss_value), model.trainable_variables)
            self.gradients = self.scaler.unscale(scaled_gradients)
        else:
            self.gradients = tape.gradient(loss_value, model.trainable_variables)

        if self.optimizer.gradient_clip_value:
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, self.optimizer.gradient_clip_value)

    def log_optimizer_state(self) -> None:
        logger.info(f"TensorFlow Optimizer: {self.optimizer}")
        if self.lr_scheduler:
            logger.info(f"Current learning rate: {self.lr_scheduler(self.optimizer.iterations)}")

    def update_lr(self, new_lr: float) -> None:
        self.optimizer.learning_rate.assign(new_lr)

    def _get_lr_scheduler(self, config: OptimizerConfig) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        if config.lr_scheduler == "exponential_decay":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=config.lr_scheduler_params['decay_steps'],
                decay_rate=config.lr_scheduler_params['decay_rate']
            )
        elif config.lr_scheduler == "cosine_decay":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=config.lr_scheduler_params['decay_steps']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {config.lr_scheduler}")

class PyTorchDistributedOptimizer(BaseDistributedOptimizer):
    def __init__(self):
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

    def prepare_optimizer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> torch.optim.Optimizer:
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            backward_passes_per_step=1,
            op=hvd.Average
        )

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        if config.lr_scheduler:
            self.lr_scheduler = self._get_lr_scheduler(config)

        return self.optimizer

    def step(self) -> None:
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def compute_gradients(self, model: torch.nn.Module, loss_fn: Callable, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            self.scaler.scale(loss).backward()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

        if self.optimizer.gradient_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.optimizer.gradient_clip_value)

    def log_optimizer_state(self) -> None:
        logger.info(f"PyTorch Optimizer: {self.optimizer}")
        logger.info(f"Optimizer state: {self.optimizer.state_dict()}")
        if self.lr_scheduler:
            logger.info(f"Current learning rate: {self.lr_scheduler.get_last_lr()}")

    def update_lr(self, new_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _get_lr_scheduler(self, config: OptimizerConfig) -> torch.optim.lr_scheduler._LRScheduler:
        if config.lr_scheduler == "step_lr":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_scheduler_params['step_size'],
                gamma=config.lr_scheduler_params['gamma']
            )
        elif config.lr_scheduler == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.lr_scheduler_params['T_max']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {config.lr_scheduler}")

class DistributedOptimizerManager:
    def __init__(self, framework: str = "tensorflow"):
        self.framework = framework.lower()
        if self.framework == "tensorflow":
            self.optimizer_handler = TensorFlowDistributedOptimizer()
        elif self.framework == "pytorch":
            self.optimizer_handler = PyTorchDistributedOptimizer()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def prepare_optimizer(self, model: Any, optimizer: Any, config: OptimizerConfig) -> Any:
        return self.optimizer_handler.prepare_optimizer(model, optimizer, config)

    def step(self) -> None:
        self.optimizer_handler.step()

    def zero_grad(self) -> None:
        self.optimizer_handler.zero_grad()

    def compute_gradients(self, model: Any, loss_fn: Callable, inputs: Any, targets: Any) -> None:
        self.optimizer_handler.compute_gradients(model, loss_fn, inputs, targets)

    def log_optimizer_state(self) -> None:
        self.optimizer_handler.log_optimizer_state()

    def update_lr(self, new_lr: float) -> None:
        self.optimizer_handler.update_lr(new_lr)

# Example usage
if __name__ == "__main__":
    # Initialize Horovod
    hvd.init()

    # TensorFlow example
    tf_manager = DistributedOptimizerManager("tensorflow")
    tf_model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    tf_optimizer = tf.keras.optimizers.Adam(0.01)
    tf_config = OptimizerConfig(
        learning_rate=0.01,
        use_mixed_precision=True,
        gradient_clip_value=1.0,
        lr_scheduler="exponential_decay",
        lr_scheduler_params={"decay_steps": 1000, "decay_rate": 0.9}
    )
    tf_distributed_optimizer = tf_manager.prepare_optimizer(tf_model, tf_optimizer, tf_config)

    # PyTorch example
    torch_manager = DistributedOptimizerManager("pytorch")
    torch_model = torch.nn.Linear(10, 10)
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    torch_config = OptimizerConfig(
        learning_rate=0.01,
        use_mixed_precision=True,
        gradient_clip_value=1.0,
        lr_scheduler="cosine_annealing",
        lr_scheduler_params={"T_max": 1000}
    )
    torch_distributed_optimizer = torch_manager.prepare_optimizer(torch_model, torch_optimizer, torch_config)

    # Log optimizer states
    tf_manager.log_optimizer_state()
    torch_manager.log_optimizer_state()

    logger.info("Distributed optimizers prepared successfully!")