import os
import json
import torch
import tensorflow as tf
from typing import Any, Optional, Dict, Union
from pathlib import Path
import shutil
import time

class CheckpointManager:
    def __init__(self, framework: str = "tensorflow", checkpoint_dir: str = "/tmp/checkpoints", max_to_keep: int = 5):
        self.framework = framework.lower()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint: Optional[Any] = None
        self.max_to_keep = max_to_keep
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata: Dict[str, Any] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": [], "latest": None}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def initialize_checkpoint(self, model: Any, optimizer: Any) -> None:
        if self.framework == "tensorflow":
            self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        elif self.framework == "pytorch":
            self.checkpoint = {'model': model, 'optimizer': optimizer}
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def save_checkpoint(self, epoch: int, step: int, metrics: Optional[Dict[str, float]] = None) -> None:
        timestamp = int(time.time())
        checkpoint_name = f"ckpt-{epoch}-{step}-{timestamp}"
        
        if self.framework == "tensorflow" and self.checkpoint:
            save_path = self.checkpoint.save(self.checkpoint_dir / checkpoint_name)
        elif self.framework == "pytorch" and self.checkpoint:
            save_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.checkpoint['model'].state_dict(),
                'optimizer_state_dict': self.checkpoint['optimizer'].state_dict(),
                'metrics': metrics
            }, save_path)
        else:
            raise ValueError("Checkpoint not initialized or unsupported framework")

        self.metadata["checkpoints"].append({
            "name": checkpoint_name,
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics
        })
        self.metadata["latest"] = checkpoint_name
        self._save_metadata()
        self._manage_checkpoints()

    def _manage_checkpoints(self):
        if len(self.metadata["checkpoints"]) > self.max_to_keep:
            to_remove = self.metadata["checkpoints"][:-self.max_to_keep]
            for checkpoint in to_remove:
                self._remove_checkpoint(checkpoint["name"])
            self.metadata["checkpoints"] = self.metadata["checkpoints"][-self.max_to_keep:]
            self._save_metadata()

    def _remove_checkpoint(self, checkpoint_name: str):
        if self.framework == "tensorflow":
            for file in self.checkpoint_dir.glob(f"{checkpoint_name}*"):
                file.unlink()
        elif self.framework == "pytorch":
            (self.checkpoint_dir / f"{checkpoint_name}.pth").unlink(missing_ok=True)

    def load_checkpoint(self, model: Any, optimizer: Any, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        if checkpoint_name is None:
            checkpoint_name = self.metadata.get("latest")
        
        if not checkpoint_name:
            print("No checkpoint found, starting from scratch.")
            return {"epoch": 0, "step": 0}

        if self.framework == "tensorflow":
            self.checkpoint.restore(self.checkpoint_dir / checkpoint_name)
            checkpoint_info = next((c for c in self.metadata["checkpoints"] if c["name"] == checkpoint_name), None)
        elif self.framework == "pytorch":
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_info = checkpoint
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        if checkpoint_info:
            print(f"Restored from checkpoint: {checkpoint_name}")
            return {
                "epoch": checkpoint_info["epoch"],
                "step": checkpoint_info["step"],
                "metrics": checkpoint_info.get("metrics", {})
            }
        else:
            raise ValueError(f"Checkpoint information not found for {checkpoint_name}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return self.metadata["checkpoints"]

    def get_best_checkpoint(self, metric: str, mode: str = 'max') -> Optional[str]:
        if not self.metadata["checkpoints"]:
            return None
        
        sorted_checkpoints = sorted(
            self.metadata["checkpoints"],
            key=lambda x: x.get("metrics", {}).get(metric, float('-inf') if mode == 'max' else float('inf')),
            reverse=(mode == 'max')
        )
        return sorted_checkpoints[0]["name"] if sorted_checkpoints else None

    def delete_checkpoint(self, checkpoint_name: str) -> None:
        self._remove_checkpoint(checkpoint_name)
        self.metadata["checkpoints"] = [c for c in self.metadata["checkpoints"] if c["name"] != checkpoint_name]
        if self.metadata["latest"] == checkpoint_name:
            self.metadata["latest"] = self.metadata["checkpoints"][-1]["name"] if self.metadata["checkpoints"] else None
        self._save_metadata()

    def export_checkpoint(self, checkpoint_name: str, export_dir: Union[str, Path]) -> None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if self.framework == "tensorflow":
            for file in self.checkpoint_dir.glob(f"{checkpoint_name}*"):
                shutil.copy(file, export_dir)
        elif self.framework == "pytorch":
            shutil.copy(self.checkpoint_dir / f"{checkpoint_name}.pth", export_dir)
        
        checkpoint_info = next((c for c in self.metadata["checkpoints"] if c["name"] == checkpoint_name), None)
        if checkpoint_info:
            with open(export_dir / f"{checkpoint_name}_metadata.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)

    def import_checkpoint(self, import_dir: Union[str, Path], checkpoint_name: str) -> None:
        import_dir = Path(import_dir)
        
        if self.framework == "tensorflow":
            for file in import_dir.glob(f"{checkpoint_name}*"):
                shutil.copy(file, self.checkpoint_dir)
        elif self.framework == "pytorch":
            shutil.copy(import_dir / f"{checkpoint_name}.pth", self.checkpoint_dir)
        
        metadata_file = import_dir / f"{checkpoint_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                checkpoint_info = json.load(f)
                self.metadata["checkpoints"].append(checkpoint_info)
                self.metadata["latest"] = checkpoint_name
                self._save_metadata()

# Example usage
if __name__ == "__main__":
    # For TensorFlow
    tf_manager = CheckpointManager(framework="tensorflow", checkpoint_dir="./tf_checkpoints")
    tf_model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
    tf_optimizer = tf.keras.optimizers.Adam()
    tf_manager.initialize_checkpoint(tf_model, tf_optimizer)
    tf_manager.save_checkpoint(epoch=1, step=100, metrics={"loss": 0.5, "accuracy": 0.8})
    tf_info = tf_manager.load_checkpoint(tf_model, tf_optimizer)
    print(f"TensorFlow checkpoint info: {tf_info}")

    # For PyTorch
    torch_manager = CheckpointManager(framework="pytorch", checkpoint_dir="./pytorch_checkpoints")
    torch_model = torch.nn.Linear(10, 10)
    torch_optimizer = torch.optim.Adam(torch_model.parameters())
    torch_manager.initialize_checkpoint(torch_model, torch_optimizer)
    torch_manager.save_checkpoint(epoch=1, step=100, metrics={"loss": 0.5, "accuracy": 0.8})
    torch_info = torch_manager.load_checkpoint(torch_model, torch_optimizer)
    print(f"PyTorch checkpoint info: {torch_info}")

    # List checkpoints
    print("TensorFlow checkpoints:", tf_manager.list_checkpoints())
    print("PyTorch checkpoints:", torch_manager.list_checkpoints())

    # Get best checkpoint
    best_tf_ckpt = tf_manager.get_best_checkpoint("accuracy", mode="max")
    print(f"Best TensorFlow checkpoint: {best_tf_ckpt}")

    # Export and import checkpoint
    tf_manager.export_checkpoint(best_tf_ckpt, "./exported_checkpoints")
    torch_manager.import_checkpoint("./exported_checkpoints", best_tf_ckpt)

    print("Checkpoint Manager demonstration completed!")