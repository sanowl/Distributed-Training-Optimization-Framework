import os
import json
import torch
import tensorflow as tf
import horovod.tensorflow as hvd_tf
import horovod.torch as hvd_torch
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    epoch: int
    step: int
    framework: str
    additional_info: Dict[str, Any] = None

class BaseFaultToleranceManager(ABC):
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        self.metadata_path = os.path.join(checkpoint_dir, "metadata.json")

    @abstractmethod
    def initialize_checkpoint(self, model: Any, optimizer: Any) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, step: int, additional_info: Dict[str, Any] = None) -> None:
        pass

    @abstractmethod
    def restore_checkpoint(self, model: Any, optimizer: Any) -> CheckpointMetadata:
        pass

    def _save_metadata(self, metadata: CheckpointMetadata) -> None:
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f)

    def _load_metadata(self) -> Optional[CheckpointMetadata]:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            return CheckpointMetadata(**data)
        return None

    @contextmanager
    def _error_handling(self, operation: str):
        try:
            yield
        except Exception as e:
            logger.error(f"Error during {operation}: {str(e)}")
            raise

class TensorFlowFaultToleranceManager(BaseFaultToleranceManager):
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        super().__init__(checkpoint_dir)
        self.checkpoint: Optional[tf.train.Checkpoint] = None

    def initialize_checkpoint(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> None:
        with self._error_handling("checkpoint initialization"):
            self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def save_checkpoint(self, epoch: int, step: int, additional_info: Dict[str, Any] = None) -> None:
        with self._error_handling("checkpoint saving"):
            if self.checkpoint:
                save_path = self.checkpoint.save(file_prefix=self.checkpoint_path)
                metadata = CheckpointMetadata(epoch=epoch, step=step, framework="tensorflow", additional_info=additional_info)
                self._save_metadata(metadata)
                logger.info(f"TensorFlow: Checkpoint saved at {save_path} for epoch {epoch}, step {step}")

    def restore_checkpoint(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> CheckpointMetadata:
        with self._error_handling("checkpoint restoration"):
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                self.checkpoint.restore(latest_checkpoint)
                metadata = self._load_metadata()
                logger.info(f"TensorFlow: Checkpoint restored from {latest_checkpoint}")
                return metadata
            logger.warning("TensorFlow: No checkpoint found, starting from scratch.")
            return CheckpointMetadata(epoch=0, step=0, framework="tensorflow")

class PyTorchFaultToleranceManager(BaseFaultToleranceManager):
    def __init__(self, checkpoint_dir: str = "/tmp/checkpoints"):
        super().__init__(checkpoint_dir)
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def initialize_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        with self._error_handling("checkpoint initialization"):
            self.model = model
            self.optimizer = optimizer

    def save_checkpoint(self, epoch: int, step: int, additional_info: Dict[str, Any] = None) -> None:
        with self._error_handling("checkpoint saving"):
            checkpoint_file = f"{self.checkpoint_path}_epoch_{epoch}_step_{step}.pth"
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'additional_info': additional_info
            }, checkpoint_file)
            metadata = CheckpointMetadata(epoch=epoch, step=step, framework="pytorch", additional_info=additional_info)
            self._save_metadata(metadata)
            logger.info(f"PyTorch: Checkpoint saved at {checkpoint_file}")

    def restore_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> CheckpointMetadata:
        with self._error_handling("checkpoint restoration"):
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                latest_checkpoint = sorted(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)))[-1]
                checkpoint = torch.load(os.path.join(self.checkpoint_dir, latest_checkpoint))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                metadata = CheckpointMetadata(
                    epoch=checkpoint['epoch'],
                    step=checkpoint['step'],
                    framework="pytorch",
                    additional_info=checkpoint.get('additional_info')
                )
                logger.info(f"PyTorch: Checkpoint restored from {latest_checkpoint}")
                return metadata
            logger.warning("PyTorch: No checkpoint found, starting from scratch.")
            return CheckpointMetadata(epoch=0, step=0, framework="pytorch")

class FaultToleranceFactory:
    @staticmethod
    def create_manager(framework: str, checkpoint_dir: str = "/tmp/checkpoints") -> BaseFaultToleranceManager:
        managers = {
            "tensorflow": TensorFlowFaultToleranceManager,
            "pytorch": PyTorchFaultToleranceManager
        }
        manager_class = managers.get(framework.lower())
        if manager_class is None:
            raise ValueError(f"Unsupported framework: {framework}")
        return manager_class(checkpoint_dir)

# Example usage
if __name__ == "__main__":
    # TensorFlow example
    tf_manager = FaultToleranceFactory.create_manager("tensorflow")
    tf_model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    tf_optimizer = tf.keras.optimizers.Adam(0.01)
    tf_manager.initialize_checkpoint(tf_model, tf_optimizer)
    tf_manager.save_checkpoint(epoch=1, step=100, additional_info={"loss": 0.5})
    metadata = tf_manager.restore_checkpoint(tf_model, tf_optimizer)
    logger.info(f"TensorFlow restored metadata: {metadata}")

    # PyTorch example
    torch_manager = FaultToleranceFactory.create_manager("pytorch")
    torch_model = torch.nn.Linear(5, 10)
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    torch_manager.initialize_checkpoint(torch_model, torch_optimizer)
    torch_manager.save_checkpoint(epoch=1, step=100, additional_info={"loss": 0.3})
    metadata = torch_manager.restore_checkpoint(torch_model, torch_optimizer)
    logger.info(f"PyTorch restored metadata: {metadata}")