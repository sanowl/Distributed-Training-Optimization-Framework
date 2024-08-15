import os
import psutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import threading
import time
import numpy as np
from abc import ABC, abstractmethod

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    network_usage: float = 0.0
    disk_usage: float = 0.0

@dataclass
class ResourceAllocation:
    cpus: int
    memory: int
    gpus: int = 0

class CloudProvider(ABC):
    @abstractmethod
    def scale_up(self, resource_type: str, amount: int) -> bool:
        pass

    @abstractmethod
    def scale_down(self, resource_type: str, amount: int) -> bool:
        pass

class AWSProvider(CloudProvider):
    def __init__(self):
        self.ec2 = boto3.client('ec2')

    def scale_up(self, resource_type: str, amount: int) -> bool:
        # Implement AWS-specific scaling logic
        logger.info(f"Scaling up {amount} {resource_type} on AWS")
        return True

    def scale_down(self, resource_type: str, amount: int) -> bool:
        # Implement AWS-specific scaling logic
        logger.info(f"Scaling down {amount} {resource_type} on AWS")
        return True

class ResourceManager:
    def __init__(self, cloud_provider: Optional[CloudProvider] = None) -> None:
        self.cpu_count: int = psutil.cpu_count()
        self.total_memory: int = psutil.virtual_memory().total
        self.cloud_provider = cloud_provider
        self.resource_history: List[ResourceMetrics] = []
        self.allocation_history: List[ResourceAllocation] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

    def start_monitoring(self, interval: int = 5) -> None:
        def monitor_task():
            while not self.stop_monitoring.is_set():
                self.monitor_resources()
                time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor_task)
        self.monitoring_thread.start()

    def stop_monitoring(self) -> None:
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def allocate_resources(self) -> ResourceAllocation:
        available_cpus = psutil.cpu_count(logical=False)
        available_memory = psutil.virtual_memory().available
        available_gpus = len(GPUtil.getGPUs()) if GPU_AVAILABLE else 0

        allocation = ResourceAllocation(cpus=available_cpus, memory=available_memory, gpus=available_gpus)
        self.allocation_history.append(allocation)

        logger.info(f"Allocating {allocation.cpus} CPUs, {allocation.memory // (1024 ** 3)} GB RAM, and {allocation.gpus} GPUs")
        return allocation

    def monitor_resources(self) -> ResourceMetrics:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = self._get_gpu_usage()
        network_usage = self._get_network_usage()
        disk_usage = psutil.disk_usage('/').percent

        metrics = ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            network_usage=network_usage,
            disk_usage=disk_usage
        )
        self.resource_history.append(metrics)

        logger.info(f"CPU: {cpu_usage:.2f}% | Memory: {memory_usage:.2f}% | GPU: {gpu_usage:.2f}% | "
                    f"Network: {network_usage:.2f}Mbps | Disk: {disk_usage:.2f}%")
        return metrics

    def optimize_allocation(self, prediction_window: int = 5) -> None:
        if len(self.resource_history) < prediction_window:
            logger.warning("Not enough history for prediction. Skipping optimization.")
            return

        predicted_usage = self._predict_resource_usage(prediction_window)
        current_allocation = self.allocation_history[-1]

        if predicted_usage.cpu_usage > 80 or predicted_usage.memory_usage > 80:
            self._scale_up(current_allocation)
        elif predicted_usage.cpu_usage < 20 and predicted_usage.memory_usage < 20:
            self._scale_down(current_allocation)
        else:
            logger.info("Resource usage within acceptable range. No changes needed.")

    def _predict_resource_usage(self, window: int) -> ResourceMetrics:
        recent_history = self.resource_history[-window:]
        cpu_trend = np.polyfit(range(window), [m.cpu_usage for m in recent_history], 1)[0]
        memory_trend = np.polyfit(range(window), [m.memory_usage for m in recent_history], 1)[0]

        last_metrics = recent_history[-1]
        predicted_cpu = min(100, max(0, last_metrics.cpu_usage + cpu_trend * window))
        predicted_memory = min(100, max(0, last_metrics.memory_usage + memory_trend * window))

        return ResourceMetrics(cpu_usage=predicted_cpu, memory_usage=predicted_memory)

    def _scale_up(self, current_allocation: ResourceAllocation) -> None:
        if self.cloud_provider:
            if self.cloud_provider.scale_up('cpu', 1):
                current_allocation.cpus += 1
            if self.cloud_provider.scale_up('memory', 1024 ** 3):  # 1 GB
                current_allocation.memory += 1024 ** 3
        else:
            logger.warning("No cloud provider available for scaling up.")

    def _scale_down(self, current_allocation: ResourceAllocation) -> None:
        if self.cloud_provider:
            if current_allocation.cpus > 1 and self.cloud_provider.scale_down('cpu', 1):
                current_allocation.cpus -= 1
            if current_allocation.memory > 2 * (1024 ** 3) and self.cloud_provider.scale_down('memory', 1024 ** 3):
                current_allocation.memory -= 1024 ** 3
        else:
            logger.warning("No cloud provider available for scaling down.")

    def _get_gpu_usage(self) -> float:
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            return sum(gpu.load for gpu in gpus) / len(gpus) * 100 if gpus else 0.0
        return 0.0

    def _get_network_usage(self) -> float:
        net_io = psutil.net_io_counters()
        return (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # Convert to Mbps

    def get_resource_trends(self) -> Dict[str, float]:
        if len(self.resource_history) < 2:
            return {}

        last_metrics = self.resource_history[-1]
        first_metrics = self.resource_history[0]
        time_diff = len(self.resource_history) - 1

        return {
            "cpu_trend": (last_metrics.cpu_usage - first_metrics.cpu_usage) / time_diff,
            "memory_trend": (last_metrics.memory_usage - first_metrics.memory_usage) / time_diff,
            "gpu_trend": (last_metrics.gpu_usage - first_metrics.gpu_usage) / time_diff,
            "network_trend": (last_metrics.network_usage - first_metrics.network_usage) / time_diff,
            "disk_trend": (last_metrics.disk_usage - first_metrics.disk_usage) / time_diff
        }

# Example usage
if __name__ == "__main__":
    cloud_provider = AWSProvider() if AWS_AVAILABLE else None
    resource_manager = ResourceManager(cloud_provider)

    try:
        resource_manager.start_monitoring(interval=2)

        for _ in range(10):
            allocation = resource_manager.allocate_resources()
            resource_manager.optimize_allocation()
            time.sleep(5)

        trends = resource_manager.get_resource_trends()
        logger.info(f"Resource trends: {trends}")

    finally:
        resource_manager.stop_monitoring()