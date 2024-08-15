import os
from typing import Tuple, Dict, List, Union, Any, Optional
import threading
import time
import logging
from collections import deque
import psutil
import GPUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceStatus:
    def __init__(self, usage: float, allocation: Union[int, List[int]]):
        self.usage: float = usage
        self.allocation: Union[int, List[int]] = allocation

class CPUResource:
    def __init__(self):
        self.count: int = psutil.cpu_count()

    def get_usage(self) -> float:
        return psutil.cpu_percent()

    def get_available(self) -> int:
        return psutil.cpu_count(logical=False)

class MemoryResource:
    def __init__(self):
        self.total: int = psutil.virtual_memory().total

    def get_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_available(self) -> int:
        return psutil.virtual_memory().available

class GPUResource:
    def __init__(self):
        self.gpus: List[GPUtil.GPU] = GPUtil.getGPUs()

    def get_usage(self) -> float:
        return max([gpu.memoryUtil for gpu in self.gpus]) if self.gpus else 0.0

    def get_available(self) -> List[int]:
        return [gpu.id for gpu in self.gpus if gpu.memoryUtil < 0.5]

class ResourceHistory:
    def __init__(self, history_size: int):
        self.history: Dict[str, deque[float]] = {
            'cpu': deque(maxlen=history_size),
            'memory': deque(maxlen=history_size),
            'gpu': deque(maxlen=history_size)
        }

    def add(self, resource_type: str, usage: float) -> None:
        self.history[resource_type].append(usage)

    def get_trend(self, resource_type: str) -> float:
        history = self.history[resource_type]
        if len(history) > 1:
            return (history[-1] - history[0]) / len(history)
        return 0.0

class EnhancedResourceManager:
    def __init__(self, monitoring_interval: int = 5, history_size: int = 100) -> None:
        self.cpu: CPUResource = CPUResource()
        self.memory: MemoryResource = MemoryResource()
        self.gpu: GPUResource = GPUResource()
        self.monitoring_interval: int = monitoring_interval
        self.stop_monitoring_flag: threading.Event = threading.Event()
        self.resource_history: ResourceHistory = ResourceHistory(history_size)
        self.monitoring_thread: Optional[threading.Thread] = None

    def allocate_resources(self) -> Dict[str, Union[int, List[int]]]:
        allocation = {
            'cpu': self.cpu.get_available(),
            'memory': self.memory.get_available(),
            'gpu': self.gpu.get_available()
        }
        logger.info(f"Allocating {allocation['cpu']} CPUs, {allocation['memory'] // (1024 ** 3)} GB RAM, "
                    f"and GPUs: {allocation['gpu']}")
        return allocation

    def monitor_resources(self) -> Dict[str, float]:
        usage = {
            'cpu': self.cpu.get_usage(),
            'memory': self.memory.get_usage(),
            'gpu': self.gpu.get_usage()
        }
        logger.info(f"CPU Usage: {usage['cpu']:.2f}% | Memory Usage: {usage['memory']:.2f}% | "
                    f"GPU Usage: {usage['gpu']:.2f}%")
        for resource_type, resource_usage in usage.items():
            self.resource_history.add(resource_type, resource_usage)
        return usage

    def optimize_allocation(self, load_factors: Dict[str, float]) -> Dict[str, Any]:
        optimizations: Dict[str, Any] = {}
        for resource, load in load_factors.items():
            if load > 0.8:
                logger.warning(f"High {resource} load detected ({load:.2f}), increasing allocation")
                optimizations[resource] = self._increase_allocation(resource)
            elif load < 0.2:
                logger.info(f"Low {resource} load detected ({load:.2f}), decreasing allocation")
                optimizations[resource] = self._decrease_allocation(resource)
            else:
                logger.info(f"{resource.capitalize()} load within acceptable range ({load:.2f})")
        return optimizations

    def _increase_allocation(self, resource: str) -> Union[int, List[int]]:
        if resource == 'cpu':
            return min(self.cpu.count, self.cpu.get_available() + 2)
        elif resource == 'memory':
            return min(self.memory.total, int(self.memory.get_available() * 1.2))
        elif resource == 'gpu':
            return [gpu.id for gpu in self.gpu.gpus if gpu.memoryUtil < 0.8]
        else:
            raise ValueError(f"Unknown resource type: {resource}")

    def _decrease_allocation(self, resource: str) -> Union[int, List[int]]:
        if resource == 'cpu':
            return max(1, self.cpu.get_available() - 1)
        elif resource == 'memory':
            return max(1024**3, int(self.memory.get_available() * 0.8))  # Ensure at least 1GB
        elif resource == 'gpu':
            return [gpu.id for gpu in self.gpu.gpus if gpu.memoryUtil < 0.2]
        else:
            raise ValueError(f"Unknown resource type: {resource}")

    def start_monitoring(self) -> None:
        def monitoring_task() -> None:
            while not self.stop_monitoring_flag.is_set():
                self.monitor_resources()
                time.sleep(self.monitoring_interval)

        self.monitoring_thread = threading.Thread(target=monitoring_task)
        self.monitoring_thread.start()

    def stop_monitoring(self) -> None:
        if self.monitoring_thread is not None:
            self.stop_monitoring_flag.set()
            self.monitoring_thread.join()
            self.monitoring_thread = None
        else:
            logger.warning("Monitoring thread is not running.")

    def get_resource_trends(self) -> Dict[str, float]:
        return {
            'cpu': self.resource_history.get_trend('cpu'),
            'memory': self.resource_history.get_trend('memory'),
            'gpu': self.resource_history.get_trend('gpu')
        }

    def suggest_scaling_strategy(self) -> str:
        trends = self.get_resource_trends()
        cpu_trend, memory_trend, gpu_trend = trends['cpu'], trends['memory'], trends['gpu']

        if cpu_trend > 0.5 and memory_trend > 0.5:
            return "Consider scaling out (adding more nodes)"
        elif gpu_trend > 0.5:
            return "Consider adding more GPUs or using more powerful GPUs"
        elif cpu_trend > 0.5:
            return "Consider using CPUs with more cores"
        elif memory_trend > 0.5:
            return "Consider adding more RAM"
        else:
            return "Current resource allocation seems sufficient"

# Example usage
if __name__ == "__main__":
    resource_manager = EnhancedResourceManager()

    try:
        initial_allocation = resource_manager.allocate_resources()
        print(f"Initial allocation: {initial_allocation}")

        resource_manager.start_monitoring()
        time.sleep(30)  # Simulate some work

        current_usage = resource_manager.monitor_resources()
        optimizations = resource_manager.optimize_allocation(current_usage)
        print(f"Suggested optimizations: {optimizations}")

        trends = resource_manager.get_resource_trends()
        print(f"Resource trends: {trends}")

        scaling_suggestion = resource_manager.suggest_scaling_strategy()
        print(f"Scaling suggestion: {scaling_suggestion}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

    finally:
        resource_manager.stop_monitoring()