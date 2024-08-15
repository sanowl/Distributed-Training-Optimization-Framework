from __future__ import annotations

import torch
import torch.nn as nn
import horovod.torch as hvd
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Optional, Callable, Dict, List, Union, Any, Protocol
from functools import partial, reduce
import operator
from dataclasses import dataclass, field
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizerProtocol(Protocol):
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]: ...
    def zero_grad(self, set_to_none: bool = ...) -> None: ...

class SchedulerProtocol(Protocol):
    def step(self, metrics: Optional[float] = None) -> None: ...

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    use_amp: bool = False
    lr_scheduler: Optional[SchedulerProtocol] = None
    callbacks: List[Callable[[int, Dict[str, float]], None]] = field(default_factory=list)

class PyTorchIntegration:
    def __init__(self, seed: int = 42):
        self.init_distributed()
        self.set_random_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_distributed(self) -> None:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def set_random_seed(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        return hvd.DistributedDataParallel(model.to(self.device))

    def prepare_optimizer(self, optimizer: OptimizerProtocol, model: nn.Module) -> OptimizerProtocol:
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            backward_passes_per_step=1,
            op=hvd.Adasum if hvd.nccl_built() else hvd.Average
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        return optimizer

    def prepare_dataset(
        self, 
        dataset: Dataset, 
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True
        )

    @staticmethod
    def train_step(
        model: nn.Module, 
        batch: Any, 
        optimizer: OptimizerProtocol, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> float:
        model.train()
        optimizer.zero_grad()
        
        inputs, labels = batch
        
        loss_computation = lambda: loss_fn(model(inputs), labels)
        
        if scaler:
            with torch.cuda.amp.autocast():
                loss = loss_computation()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_computation()
            loss.backward()
            optimizer.step()
        
        return loss.item()

    def distributed_train(
        self, 
        model: nn.Module, 
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        optimizer: OptimizerProtocol, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        config: TrainingConfig
    ) -> Dict[str, List[float]]:
        model = self.prepare_model(model)
        optimizer = self.prepare_optimizer(optimizer, model)
        train_loader = self.prepare_dataset(train_dataset, config.batch_size)
        val_loader = self.prepare_dataset(val_dataset, config.batch_size, shuffle=False) if val_dataset else None

        scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(config.epochs):
            model.train()
            train_loss = self.run_epoch(model, train_loader, optimizer, loss_fn, scaler, is_training=True)
            history['train_loss'].append(train_loss)

            val_loss = self.run_epoch(model, val_loader, optimizer, loss_fn, scaler, is_training=False) if val_loader else None
            if val_loss:
                history['val_loss'].append(val_loss)

            self.log_epoch_results(epoch, config.epochs, train_loss, val_loss)
            self.update_lr_scheduler(config.lr_scheduler, val_loss or train_loss)
            self.run_callbacks(config.callbacks, epoch, {'loss': train_loss, 'val_loss': val_loss})

        return history

    def run_epoch(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        optimizer: OptimizerProtocol, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: Optional[torch.cuda.amp.GradScaler],
        is_training: bool
    ) -> float:
        total_loss = 0.0
        
        with torch.set_grad_enabled(is_training):
            for batch in tqdm(dataloader, disable=not hvd.rank() == 0):
                batch = self.to_device(batch)
                step_fn = partial(self.train_step, model, batch, optimizer, loss_fn) if is_training else partial(self.eval_step, model, batch, loss_fn)
                loss = step_fn(scaler) if is_training else step_fn()
                total_loss += loss

        return total_loss / len(dataloader)

    @staticmethod
    def eval_step(
        model: nn.Module, 
        batch: Any, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        inputs, labels = batch
        outputs = model(inputs)
        return loss_fn(outputs, labels).item()

    def to_device(self, batch: Union[torch.Tensor, List, Dict[str, torch.Tensor]]) -> Any:
        to_device_func = lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
        
        match batch:
            case torch.Tensor():
                return to_device_func(batch)
            case list():
                return [self.to_device(item) for item in batch]
            case dict():
                return {k: self.to_device(v) for k, v in batch.items()}
            case _:
                return batch

    @staticmethod
    def log_epoch_results(epoch: int, total_epochs: int, train_loss: float, val_loss: Optional[float]) -> None:
        if hvd.rank() == 0:
            log_message = f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {train_loss:.4f}"
            log_message += f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
            logger.info(log_message)

    @staticmethod
    def update_lr_scheduler(scheduler: Optional[SchedulerProtocol], loss: float) -> None:
        if scheduler:
            scheduler.step(loss)

    @staticmethod
    def run_callbacks(callbacks: List[Callable[[int, Dict[str, float]], None]], epoch: int, metrics: Dict[str, float]) -> None:
        for callback in callbacks:
            callback(epoch, metrics)

    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer: OptimizerProtocol, epoch: int, filepath: str) -> None:
        if hvd.rank() == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str, model: nn.Module, optimizer: Optional[OptimizerProtocol] = None) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

# Example usage
if __name__ == "__main__":
    integration = PyTorchIntegration()

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10),
        torch.randn(1000, 1)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.randn(200, 10),
        torch.randn(200, 1)
    )

    # Define loss function
    loss_fn = nn.MSELoss()

    # Configure training
    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        use_amp=True,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
        callbacks=[lambda epoch, metrics: print(f"Epoch {epoch}: {metrics}")]
    )

    # Run distributed training
    history = integration.distributed_train(model, train_dataset, val_dataset, optimizer, loss_fn, config)

    print("Training completed. Final losses:", history['train_loss'][-1], history['val_loss'][-1])