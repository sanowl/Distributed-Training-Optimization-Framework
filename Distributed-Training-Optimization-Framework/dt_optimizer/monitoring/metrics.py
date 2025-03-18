from typing import Callable, Dict, List, Optional, Union
from enum import Enum
import time
import threading
import logging
from dataclasses import dataclass
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class Alert:
    name: str
    message: str
    level: AlertLevel
    timestamp: float

class AlertChannel:
    def send_alert(self, alert: Alert) -> None:
        raise NotImplementedError("Subclasses must implement send_alert method")

class ConsoleAlertChannel(AlertChannel):
    def send_alert(self, alert: Alert) -> None:
        logger.log(
            logging.INFO if alert.level == AlertLevel.INFO else 
            logging.WARNING if alert.level == AlertLevel.WARNING else
            logging.ERROR if alert.level == AlertLevel.ERROR else
            logging.CRITICAL,
            f"{alert.name} - {alert.message}"
        )

class EmailAlertChannel(AlertChannel):
    def __init__(self, smtp_server: str, smtp_port: int, sender: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients

    def send_alert(self, alert: Alert) -> None:
        # Implement email sending logic here
        logger.info(f"Sending email alert: {alert.name} - {alert.message}")

class SlackAlertChannel(AlertChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, alert: Alert) -> None:
        payload = {
            "text": f"*{alert.level.name}*: {alert.name}\n{alert.message}"
        }
        response = requests.post(self.webhook_url, json=payload, timeout=60)
        if response.status_code != 200:
            logger.error(f"Failed to send Slack alert: {response.text}")

class AdvancedAlertManager:
    def __init__(self):
        self.alert_callbacks: Dict[str, List[Callable[[Alert], None]]] = {}
        self.alert_channels: Dict[str, AlertChannel] = {}
        self.alert_history: List[Alert] = []
        self.alert_throttle: Dict[str, float] = {}
        self.default_throttle_period: float = 300  # 5 minutes
        self.lock = threading.Lock()

    def register_alert(self, alert_name: str, callback: Callable[[Alert], None], level: AlertLevel = AlertLevel.INFO) -> None:
        with self.lock:
            if alert_name not in self.alert_callbacks:
                self.alert_callbacks[alert_name] = []
            self.alert_callbacks[alert_name].append(callback)

    def register_channel(self, channel_name: str, channel: AlertChannel) -> None:
        with self.lock:
            self.alert_channels[channel_name] = channel

    def trigger_alert(self, alert_name: str, message: str, level: AlertLevel = AlertLevel.INFO, channels: Optional[List[str]] = None) -> None:
        with self.lock:
            current_time = time.time()
            if alert_name in self.alert_throttle:
                if current_time - self.alert_throttle[alert_name] < self.default_throttle_period:
                    logger.info(f"Alert {alert_name} throttled")
                    return
            
            self.alert_throttle[alert_name] = current_time
            alert = Alert(alert_name, message, level, current_time)
            self.alert_history.append(alert)

            if alert_name in self.alert_callbacks:
                for callback in self.alert_callbacks[alert_name]:
                    callback(alert)

            if channels:
                for channel in channels:
                    if channel in self.alert_channels:
                        self.alert_channels[channel].send_alert(alert)
            else:
                for channel in self.alert_channels.values():
                    channel.send_alert(alert)

    def check_conditions(self, condition_fn: Callable[[], bool], alert_name: str, message: str, 
                         level: AlertLevel = AlertLevel.INFO, channels: Optional[List[str]] = None) -> None:
        if condition_fn():
            self.trigger_alert(alert_name, message, level, channels)

    def set_throttle_period(self, alert_name: str, period: float) -> None:
        with self.lock:
            self.alert_throttle[alert_name] = period

    def get_alert_history(self, alert_name: Optional[str] = None, 
                          level: Optional[AlertLevel] = None, 
                          start_time: Optional[float] = None, 
                          end_time: Optional[float] = None) -> List[Alert]:
        filtered_history = self.alert_history
        if alert_name:
            filtered_history = [alert for alert in filtered_history if alert.name == alert_name]
        if level:
            filtered_history = [alert for alert in filtered_history if alert.level == level]
        if start_time:
            filtered_history = [alert for alert in filtered_history if alert.timestamp >= start_time]
        if end_time:
            filtered_history = [alert for alert in filtered_history if alert.timestamp <= end_time]
        return filtered_history

    def export_alert_history(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump([alert.__dict__ for alert in self.alert_history], f, default=str)

    def import_alert_history(self, filename: str) -> None:
        with open(filename, 'r') as f:
            data = json.load(f)
            self.alert_history = [Alert(**alert) for alert in data]

# Example usage
if __name__ == "__main__":
    alert_manager = AdvancedAlertManager()

    # Register channels
    alert_manager.register_channel("console", ConsoleAlertChannel())
    alert_manager.register_channel("slack", SlackAlertChannel("https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"))

    # Register alerts
    alert_manager.register_alert("high_cpu_usage", lambda alert: print(f"High CPU alert: {alert.message}"), AlertLevel.WARNING)
    alert_manager.register_alert("low_disk_space", lambda alert: print(f"Low disk space alert: {alert.message}"), AlertLevel.ERROR)

    # Set custom throttle period for an alert
    alert_manager.set_throttle_period("high_cpu_usage", 600)  # 10 minutes

    # Trigger alerts
    alert_manager.trigger_alert("high_cpu_usage", "CPU usage is at 90%", AlertLevel.WARNING, ["console", "slack"])
    alert_manager.trigger_alert("low_disk_space", "Disk space is below 10%", AlertLevel.ERROR)

    # Check condition and trigger alert if true
    def check_memory_usage():
        return True  # Replace with actual memory check logic

    alert_manager.check_conditions(check_memory_usage, "high_memory_usage", "Memory usage is at 95%", AlertLevel.CRITICAL)

    # Get alert history
    recent_alerts = alert_manager.get_alert_history(level=AlertLevel.WARNING, start_time=time.time() - 3600)
    print("Recent warning alerts:", recent_alerts)

    # Export alert history
    alert_manager.export_alert_history("alert_history.json")

    print("Advanced Alert Manager demonstration completed!")
