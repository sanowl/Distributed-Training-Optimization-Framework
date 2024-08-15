import torch
import torch.optim as optim
from typing import Optional, Union, Dict, Any, Callable, List
from enum import Enum, auto

class SchedulerStrategy(Enum):
    EXPONENTIAL_DECAY = auto()
    REDUCE_ON_PLATEAU = auto()
    STEP_LR = auto()
    COSINE_ANNEALING = auto()
    CYCLIC_LR = auto()

class AdaptiveLRScheduler:
    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        strategy: Union[str, SchedulerStrategy] = SchedulerStrategy.EXPONENTIAL_DECAY, 
        **kwargs: Any
    ) -> None:
        self.optimizer: optim.Optimizer = optimizer
        self.strategy: SchedulerStrategy = self._parse_strategy(strategy)
        self.scheduler: Union[
            optim.lr_scheduler.ExponentialLR,
            optim.lr_scheduler.ReduceLROnPlateau,
            optim.lr_scheduler.StepLR,
            optim.lr_scheduler.CosineAnnealingLR,
            optim.lr_scheduler.CyclicLR
        ] = self._create_scheduler(**kwargs)

    def _parse_strategy(self, strategy: Union[str, SchedulerStrategy]) -> SchedulerStrategy:
        match strategy:
            case SchedulerStrategy() as s:
                return s
            case str() as s:
                try:
                    return SchedulerStrategy[s.upper()]
                except KeyError:
                    raise ValueError(f"Unsupported strategy: {s}")
            case _:
                raise TypeError(f"Strategy must be string or SchedulerStrategy, not {type(strategy)}")

    def _create_scheduler(self, **kwargs: Any) -> Union[
        optim.lr_scheduler.ExponentialLR,
        optim.lr_scheduler.ReduceLROnPlateau,
        optim.lr_scheduler.StepLR,
        optim.lr_scheduler.CosineAnnealingLR,
        optim.lr_scheduler.CyclicLR
    ]:
        scheduler_map: Dict[SchedulerStrategy, Callable] = {
            SchedulerStrategy.EXPONENTIAL_DECAY: lambda: optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs),
            SchedulerStrategy.REDUCE_ON_PLATEAU: lambda: optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs),
            SchedulerStrategy.STEP_LR: lambda: optim.lr_scheduler.StepLR(self.optimizer, **kwargs),
            SchedulerStrategy.COSINE_ANNEALING: lambda: optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs),
            SchedulerStrategy.CYCLIC_LR: lambda: optim.lr_scheduler.CyclicLR(self.optimizer, **kwargs)
        }
        
        return scheduler_map.get(self.strategy, lambda: ValueError(f"Scheduler not implemented for strategy: {self.strategy}"))()

    def step(self, metric: Optional[float] = None) -> None:
        match self.scheduler:
            case optim.lr_scheduler.ReduceLROnPlateau():
                if metric is None:
                    raise ValueError("Metric value required for ReduceLROnPlateau")
                self.scheduler.step(metric)
            case _:
                self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        return (
            [group['lr'] for group in self.optimizer.param_groups]
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau)
            else self.scheduler.get_last_lr()
        )

    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.name,
            'scheduler_state_dict': self.scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.strategy = SchedulerStrategy[state_dict['strategy']]
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Example with ExponentialLR
    scheduler1 = AdaptiveLRScheduler(optimizer, strategy="exponential_decay", gamma=0.9)
    print(f"Initial LR: {scheduler1.get_lr()}")
    for _ in range(5):
        scheduler1.step()
        print(f"Current LR: {scheduler1.get_lr()}")
    
    # Example with ReduceLROnPlateau
    optimizer.param_groups[0]['lr'] = 0.1  # Reset LR
    scheduler2 = AdaptiveLRScheduler(optimizer, strategy="reduce_on_plateau", mode='min', factor=0.1, patience=2)
    print("\nReduceLROnPlateau:")
    for i in range(10):
        loss = 1 / (i + 1)  # Simulated loss
        scheduler2.step(loss)
        print(f"Epoch {i+1}, Loss: {loss:.4f}, LR: {scheduler2.get_lr()}")
    
    # Saving and loading state
    state = scheduler2.state_dict()
    new_scheduler = AdaptiveLRScheduler(optimizer, strategy="reduce_on_plateau")
    new_scheduler.load_state_dict(state)
    print(f"\nLoaded scheduler strategy: {new_scheduler.strategy}")