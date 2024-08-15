import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Any, Optional, Union
import time
import json
import os
from dataclasses import dataclass, field
import threading
from collections import deque

@dataclass
class MetricHistory:
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float) -> None:
        self.values.append(value)
    
    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0

class PerformanceTracker:
    def __init__(self):
        self.metrics: Dict[str, MetricHistory] = {}
        self._lock = threading.Lock()

    def track(self, metric_name: str, value: float) -> None:
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = MetricHistory()
            self.metrics[metric_name].add(value)

    def get_average(self, metric_name: str) -> float:
        with self._lock:
            return self.metrics[metric_name].average() if metric_name in self.metrics else 0

    def get_all_averages(self) -> Dict[str, float]:
        with self._lock:
            return {name: metric.average() for name, metric in self.metrics.items()}

class CustomJsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'metrics'):
            log_record['metrics'] = record.metrics
        return json.dumps(log_record)

class AdvancedLogger:
    def __init__(self, 
                 log_file: str = "training.log", 
                 level: int = logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5,
                 rotation: str = 'size',
                 console_output: bool = True,
                 json_output: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.log_file = log_file
        self.performance_tracker = PerformanceTracker()

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler with rotation
        if rotation == 'size':
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        elif rotation == 'time':
            file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=backup_count)
        else:
            raise ValueError("Invalid rotation type. Use 'size' or 'time'.")

        # Formatter
        formatter = CustomJsonFormatter() if json_output else logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        log_method = getattr(self.logger, level.lower(), None)
        if log_method is None:
            raise ValueError(f"Invalid log level: {level}")
        log_method(message, extra=extra)

    def log_info(self, message: str) -> None:
        self.log('info', message)

    def log_error(self, message: str) -> None:
        self.log('error', message)

    def log_warning(self, message: str) -> None:
        self.log('warning', message)

    def log_debug(self, message: str) -> None:
        self.log('debug', message)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.log_info(f"Epoch {epoch}: {metric_str}", extra={'metrics': metrics})
        
        for metric_name, value in metrics.items():
            self.performance_tracker.track(metric_name, value)

    def log_performance(self) -> None:
        averages = self.performance_tracker.get_all_averages()
        self.log_info("Performance Metrics (Averages):", extra={'metrics': averages})

    @staticmethod
    def time_function(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger = AdvancedLogger()
            logger.log_info(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
            return result
        return wrapper

    def set_log_level(self, level: Union[int, str]) -> None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)

    def add_file_handler(self, log_file: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
        handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def remove_file_handler(self, log_file: str) -> None:
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler) and handler.baseFilename == os.path.abspath(log_file):
                self.logger.removeHandler(handler)
                handler.close()
                break

# Example usage
if __name__ == "__main__":
    logger = AdvancedLogger(log_file="advanced_training.log", rotation='time', json_output=True)

    # Logging different levels
    logger.log_info("Starting training process")
    logger.log_warning("GPU memory is running low")
    logger.log_error("Failed to load checkpoint")

    # Logging metrics
    for epoch in range(5):
        metrics = {"loss": 0.5 - epoch * 0.1, "accuracy": 0.8 + epoch * 0.04}
        logger.log_metrics(epoch, metrics)

    # Log performance averages
    logger.log_performance()

    # Using the timer decorator
    @AdvancedLogger.time_function
    def some_long_running_function():
        time.sleep(2)
        return "Function completed"

    result = some_long_running_function()
    logger.log_info(result)

    # Changing log level
    logger.set_log_level("DEBUG")
    logger.log_debug("This is a debug message")

    # Adding and removing file handlers
    logger.add_file_handler("secondary_log.log")
    logger.log_info("This message goes to both log files")
    logger.remove_file_handler("secondary_log.log")
    logger.log_info("This message only goes to the main log file")