import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import pipeline_parallel_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelismStrategy:
    def __init__(self, model: nn.Module, pipeline_chunks: int = 1, balance: List[float] = None) -> None:
        self.model = model
        self.pipeline_chunks = pipeline_chunks
        self.balance = balance

    def apply(self, device_ids: Optional[List[int]] = None) -> nn.Module:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        return self._apply_parallelism(device_ids)

    def _apply_parallelism(self, device_ids: List[int]) -> nn.Module:
        raise NotImplementedError("This method should be implemented by subclasses")

class DataParallelism(ParallelismStrategy):
    def _apply_parallelism(self, device_ids: List[int]) -> nn.Module:
        logger.info(f"Applying Data Parallelism on devices: {device_ids}")
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        return DistributedDataParallel(self.model.to(device_ids[0]), device_ids=device_ids)

class ModelParallelism(ParallelismStrategy):
    def _apply_parallelism(self, device_ids: List[int]) -> nn.Module:
        logger.info(f"Applying Model Parallelism on devices: {device_ids}")
        layers = list(self.model.children())
        if self.balance:
            assert len(self.balance) == len(device_ids), "Balance must match number of devices"
            cumsum = torch.cumsum(torch.tensor(self.balance), 0)
            device_assignment = torch.bucketize(torch.linspace(0, 1, len(layers)), cumsum)
        else:
            device_assignment = torch.tensor([i % len(device_ids) for i in range(len(layers))])

        for i, layer in enumerate(layers):
            layer.to(f'cuda:{device_ids[device_assignment[i]]}')
        
        class ModelParallelSequential(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x.to(next(layer.parameters()).device))
                return x
        
        return ModelParallelSequential(layers)

class PipelineParallelism(ParallelismStrategy):
    def _apply_parallelism(self, device_ids: List[int]) -> nn.Module:
        logger.info(f"Applying Pipeline Parallelism on devices: {device_ids}")
        splits = pipeline_parallel_split(self.model, device_ids, self.pipeline_chunks, self.balance)
        return PipelineParallelModule(splits, device_ids)

class HybridParallelism(ParallelismStrategy):
    def _apply_parallelism(self, device_ids: List[int]) -> nn.Module:
        logger.info(f"Applying Hybrid Parallelism on devices: {device_ids}")
        model_parallel = ModelParallelism(self.model)._apply_parallelism(device_ids[:len(device_ids)//2])
        return DataParallelism(model_parallel)._apply_parallelism([device_ids[len(device_ids)//2]])

class PipelineParallelModule(nn.Module):
    def __init__(self, splits: List[nn.Module], device_ids: List[int]):
        super().__init__()
        self.splits = nn.ModuleList([split.to(f'cuda:{device_ids[i]}') for i, split in enumerate(splits)])
        self.device_ids = device_ids

    def forward(self, x):
        for i, split in enumerate(self.splits):
            x = split(x.to(f'cuda:{self.device_ids[i]}'))
        return x

class ParallelismOrchestrator:
    def __init__(self, model: nn.Module):
        self.model = model
        self.parallel_model = None

    def parallelize(self, config: Dict[str, Union[str, List[int], int, List[float]]]) -> nn.Module:
        parallel_type = config.get('type', 'data')
        device_ids = config.get('device_ids', None)
        pipeline_chunks = config.get('pipeline_chunks', 1)
        balance = config.get('balance', None)

        strategy_map = {
            'data': DataParallelism,
            'model': ModelParallelism,
            'pipeline': PipelineParallelism,
            'hybrid': HybridParallelism
        }

        if parallel_type not in strategy_map:
            raise ValueError(f"Unsupported parallelism type: {parallel_type}")

        strategy = strategy_map[parallel_type](self.model, pipeline_chunks, balance)
        self.parallel_model = strategy.apply(device_ids)
        return self.parallel_model

    def get_parallel_model(self) -> Optional[nn.Module]:
        return self.parallel_model

# Example usage
if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    orchestrator = ParallelismOrchestrator(model)

    # Data Parallelism
    data_parallel_model = orchestrator.parallelize({
        'type': 'data',
        'device_ids': [0, 1, 2, 3]
    })

    # Model Parallelism
    model_parallel_model = orchestrator.parallelize({
        'type': 'model',
        'device_ids': [0, 1, 2, 3],
        'balance': [0.3, 0.3, 0.2, 0.2]  # Distribute model across GPUs with this balance
    })

    # Pipeline Parallelism
    pipeline_parallel_model = orchestrator.parallelize({
        'type': 'pipeline',
        'device_ids': [0, 1, 2, 3],
        'pipeline_chunks': 4
    })

    # Hybrid Parallelism
    hybrid_parallel_model = orchestrator.parallelize({
        'type': 'hybrid',
        'device_ids': [0, 1, 2, 3]
    })

    print("Parallelism applied successfully!")