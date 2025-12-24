import time
import json
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import threading
import psutil
import GPUtil
import socket

class ProductionMonitor:
    """
    Production monitoring system
    """
    
    def __init__(self,
                 monitor_interval: int = 5,  # seconds
                 history_size: int = 1000):
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        self.metrics_history = []
        self.alerts = []
        self.running = False
        self.monitor_thread = None
        
        # Metric thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "inference_latency_ms": 100.0,
            "error_rate": 0.05,  # 5%
            "throughput_drop": 0.3  # 30% drop
        }
    
    def start_monitoring(self):
        """Start monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history size limited
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history = self.metrics_history[-self.history_size:]
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                # Log metrics
                if len(self.metrics_history) % 10 == 0:  # Log every 10 samples
                    self._log_metrics(metrics)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> Dict:
        """Collect system and application metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "application": {},
            "network": {}
        }
        
        # System metrics
        metrics["system"]["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        metrics["system"]["memory_percent"] = psutil.virtual_memory().percent
        metrics["system"]["disk_percent"] = psutil.disk_usage("/").percent
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics["system"]["gpu_load"] = gpu.load * 100
                metrics["system"]["gpu_memory_percent"] = gpu.memoryUtil * 100
                metrics["system"]["gpu_temperature"] = gpu.temperature
        except:
            pass
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics["network"]["bytes_sent"] = net_io.bytes_sent
        metrics["network"]["bytes_recv"] = net_io.bytes_recv
        metrics["network"]["packets_sent"] = net_io.packets_sent
        metrics["network"]["packets_recv"] = net_io.packets_recv
        
        # Application-specific metrics (to be populated by application)
        metrics["application"]["inference_count"] = 0
        metrics["application"]["error_count"] = 0
        metrics["application"]["avg_latency_ms"] = 0.0
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict):
        """Check metrics against thresholds"""
        alerts = []
        
        # CPU check
        if metrics["system"].get("cpu_percent", 0) > self.thresholds["cpu_percent"]:
            alerts.append({
                "type": "warning",
                "metric": "cpu_percent",
                "value": metrics["system"]["cpu_percent"],
                "threshold": self.thresholds["cpu_percent"],
                "message": f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%"
            })
        
        # Memory check
        if metrics["system"].get("memory_percent", 0) > self.thresholds["memory_percent"]:
            alerts.append({
                "type": "warning",
                "metric": "memory_percent",
                "value": metrics["system"]["memory_percent"],
                "threshold": self.thresholds["memory_percent"],
                "message": f"High memory usage: {metrics['system']['memory_percent']:.1f}%"
            })
        
        # GPU memory check
        if "gpu_memory_percent" in metrics["system"]:
            if metrics["system"]["gpu_memory_percent"] > self.thresholds["gpu_memory_percent"]:
                alerts.append({
                    "type": "warning",
                    "metric": "gpu_memory_percent",
                    "value": metrics["system"]["gpu_memory_percent"],
                    "threshold": self.thresholds["gpu_memory_percent"],
                    "message": f"High GPU memory usage: {metrics['system']['gpu_memory_percent']:.1f}%"
                })
        
        # Add alerts to history
        for alert in alerts:
            alert["timestamp"] = metrics["timestamp"]
            self.alerts.append(alert)
            
            # Print alert
            print(f"ALERT: {alert['message']}")
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to file"""
        log_entry = {
            "timestamp": metrics["timestamp"],
            "metrics": {
                "cpu": metrics["system"].get("cpu_percent", 0),
                "memory": metrics["system"].get("memory_percent", 0),
                "inference_latency": metrics["application"].get("avg_latency_ms", 0)
            }
        }
        
        # Write to log file
        with open("production_metrics.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_metrics_summary(self, 
                           time_window_minutes: int = 5) -> Dict:
        """Get metrics summary for time window"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=time_window_minutes)
        
        # Filter metrics by time window
        window_metrics = []
        for metric in self.metrics_history:
            try:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff:
                    window_metrics.append(metric)
            except:
                pass
        
        if not window_metrics:
            return {"error": "No metrics in time window"}
        
        # Calculate statistics
        summary = {
            "time_window_minutes": time_window_minutes,
            "sample_count": len(window_metrics),
            "averages": {},
            "maximums": {},
            "minimums": {},
            "alerts_count": len([a for a in self.alerts 
                               if datetime.fromisoformat(a["timestamp"]) >= cutoff])
        }
        
        # Calculate for each metric
        metrics_to_analyze = [
            ("cpu_percent", "system"),
            ("memory_percent", "system"),
            ("avg_latency_ms", "application")
        ]
        
        for metric_name, category in metrics_to_analyze:
            values = []
            for metric in window_metrics:
                value = metric.get(category, {}).get(metric_name)
                if value is not None:
                    values.append(float(value))
            
            if values:
                summary["averages"][metric_name] = np.mean(values)
                summary["maximums"][metric_name] = np.max(values)
                summary["minimums"][metric_name] = np.min(values)
        
        return summary
    
    def add_custom_metric(self,
                         metric_name: str,
                         value: float,
                         category: str = "application"):
        """Add custom application metric"""
        if self.metrics_history:
            self.metrics_history[-1][category][metric_name] = value

class AlertSystem:
    """
    Alert system for production monitoring
    """
    
    def __init__(self):
        self.alert_handlers = []
        self.alert_history = []
        
        # Alert configurations
        self.alert_configs = {
            "critical": {
                "cooldown_seconds": 60,
                "max_alerts_per_hour": 10,
                "notify_channels": ["log", "email"]
            },
            "warning": {
                "cooldown_seconds": 300,
                "max_alerts_per_hour": 30,
                "notify_channels": ["log"]
            },
            "info": {
                "cooldown_seconds": 60,
                "max_alerts_per_hour": 100,
                "notify_channels": ["log"]
            }
        }
    
    def register_handler(self, 
                        handler: Callable,
                        alert_types: List[str] = None):
        """Register alert handler"""
        self.alert_handlers.append({
            "handler": handler,
            "alert_types": alert_types or ["critical", "warning", "info"]
        })
    
    def trigger_alert(self,
                     alert_type: str,
                     message: str,
                     details: Optional[Dict] = None):
        """Trigger alert"""
        # Check rate limiting
        if not self._check_rate_limit(alert_type):
            return
        
        # Create alert
        alert = {
            "type": alert_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        # Add to history
        self.alert_history.append(alert)
        
        # Notify handlers
        for handler_info in self.alert_handlers:
            if alert_type in handler_info["alert_types"]:
                try:
                    handler_info["handler"](alert)
                except Exception as e:
                    print(f"Alert handler error: {e}")
        
        # Print alert
        print(f"[{alert_type.upper()}] {message}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge alert"""
        if 0 <= alert_id < len(self.alert_history):
            self.alert_history[alert_id]["acknowledged"] = True
    
    def get_active_alerts(self) -> List[Dict]:
        """Get unacknowledged alerts"""
        return [alert for alert in self.alert_history 
                if not alert["acknowledged"]]
    
    def get_alert_stats(self,
                       time_window_hours: int = 24) -> Dict:
        """Get alert statistics"""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_alerts = []
        for alert in self.alert_history:
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time >= cutoff:
                    recent_alerts.append(alert)
            except:
                pass
        
        # Calculate statistics
        stats = {
            "total_alerts": len(recent_alerts),
            "by_type": {},
            "acknowledged": len([a for a in recent_alerts if a["acknowledged"]]),
            "unacknowledged": len([a for a in recent_alerts if not a["acknowledged"]])
        }
        
        for alert in recent_alerts:
            alert_type = alert["type"]
            stats["by_type"][alert_type] = stats["by_type"].get(alert_type, 0) + 1
        
        return stats
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if alert is within rate limits"""
        config = self.alert_configs.get(alert_type, {})
        
        # Check cooldown
        cooldown = config.get("cooldown_seconds", 60)
        cutoff = datetime.now() - timedelta(seconds=cooldown)
        
        recent_same_type = [
            a for a in self.alert_history
            if a["type"] == alert_type and 
            datetime.fromisoformat(a["timestamp"]) > cutoff
        ]
        
        if recent_same_type:
            return False
        
        # Check hourly limit
        hourly_limit = config.get("max_alerts_per_hour", 10)
        hour_cutoff = datetime.now() - timedelta(hours=1)
        
        hourly_same_type = [
            a for a in self.alert_history
            if a["type"] == alert_type and 
            datetime.fromisoformat(a["timestamp"]) > hour_cutoff
        ]
        
        return len(hourly_same_type) < hourly_limit
    
    # Built-in alert handlers
    @staticmethod
    def log_handler(alert: Dict):
        """Log handler for alerts"""
        with open("alerts.log", "a") as f:
            f.write(json.dumps(alert) + "\n")
    
    @staticmethod
    def email_handler(alert: Dict):
        """Email handler for alerts (stub)"""
        # In production, implement actual email sending
        print(f"Would send email for alert: {alert['message']}")
    
    @staticmethod
    def slack_handler(alert: Dict):
        """Slack handler for alerts (stub)"""
        # In production, implement Slack webhook
        print(f"Would send Slack message for alert: {alert['message']}")