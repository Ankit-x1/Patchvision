"""
Performance monitoring and health checks for PatchVision
"""

import time
import psutil
import threading
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
from ..interfaces import PerformanceMonitor, HealthCheckInterface


@dataclass
class MetricRecord:
    """Single metric record"""
    timestamp: float
    component: str
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any]


@dataclass
class HealthStatus:
    """Health status for a component"""
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    last_check: float
    response_time: float
    error_count: int
    metadata: Dict[str, Any]


class PerformanceMonitorImpl(PerformanceMonitor):
    """Implementation of performance monitoring"""
    
    def __init__(self, log_file: str = "performance_metrics.json"):
        self.log_file = Path(log_file)
        self.metrics: List[MetricRecord] = []
        self.active_monitors: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # System metrics
        self.process = psutil.Process()
        
    def start_monitoring(self, component_name: str) -> None:
        """Start monitoring a component"""
        with self._lock:
            self.active_monitors[component_name] = {
                'start_time': time.time(),
                'start_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
                'start_cpu': self.process.cpu_percent()
            }
            self.logger.info(f"Started monitoring component: {component_name}")
    
    def record_metric(self, metric_name: str, value: float, **metadata) -> None:
        """Record a performance metric"""
        with self._lock:
            record = MetricRecord(
                timestamp=time.time(),
                component=metadata.get('component', 'unknown'),
                metric_name=metric_name,
                value=value,
                unit=metadata.get('unit', 'ms'),
                metadata=metadata
            )
            self.metrics.append(record)
            
            # Keep only last 10000 records in memory
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-10000:]
    
    def get_metrics(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            if component_name:
                component_metrics = [m for m in self.metrics if m.component == component_name]
                return self._analyze_component_metrics(component_name, component_metrics)
            else:
                return self._analyze_all_metrics()
    
    def generate_report(self, output_path: str) -> bool:
        """Generate performance report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': self.get_metrics(),
                'system_info': self._get_system_info(),
                'detailed_metrics': [asdict(m) for m in self.metrics[-1000:]]  # Last 1000 metrics
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Performance report saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return False
    
    def _analyze_component_metrics(self, component_name: str, metrics: List[MetricRecord]) -> Dict[str, Any]:
        """Analyze metrics for a specific component"""
        if not metrics:
            return {'error': f'No metrics found for component {component_name}'}
        
        # Group by metric name
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
        
        analysis = {
            'component': component_name,
            'metric_count': len(metrics),
            'time_range': {
                'start': min(m.timestamp for m in metrics),
                'end': max(m.timestamp for m in metrics)
            }
        }
        
        # Analyze each metric group
        for metric_name, values in metric_groups.items():
            analysis[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1] if values else None
            }
        
        return analysis
    
    def _analyze_all_metrics(self) -> Dict[str, Any]:
        """Analyze all metrics"""
        if not self.metrics:
            return {'error': 'No metrics available'}
        
        # Group by component
        components = {}
        for metric in self.metrics:
            if metric.component not in components:
                components[metric.component] = []
            components[metric.component].append(metric)
        
        analysis = {
            'total_metrics': len(self.metrics),
            'components': {},
            'system_usage': self._get_current_system_usage()
        }
        
        # Analyze each component
        for component, metrics in components.items():
            analysis['components'][component] = self._analyze_component_metrics(component, metrics)
        
        return analysis
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024,  # MB
            'disk_usage': psutil.disk_usage('/').percent if psutil.disk_usage('/') else None
        }
    
    def _get_current_system_usage(self) -> Dict[str, Any]:
        """Get current system usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'process_memory_mb': self.process.memory_info().rss / 1024 / 1024
        }


class HealthCheckImpl(HealthCheckInterface):
    """Implementation of health checks"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: List[HealthStatus] = []
        self.logger = logging.getLogger(__name__)
    
    def check_health(self, component_name: str) -> Dict[str, Any]:
        """Check health of a component"""
        start_time = time.time()
        
        try:
            if component_name not in self.health_checks:
                return {
                    'component': component_name,
                    'status': 'unknown',
                    'error': f'No health check registered for {component_name}'
                }
            
            # Execute health check
            check_result = self.health_checks[component_name]()
            response_time = time.time() - start_time
            
            # Determine status
            if isinstance(check_result, dict):
                status = check_result.get('status', 'unknown')
                metadata = check_result.get('metadata', {})
                error_count = check_result.get('error_count', 0)
            else:
                status = 'healthy' if check_result else 'critical'
                metadata = {}
                error_count = 0 if check_result else 1
            
            health_status = HealthStatus(
                component=component_name,
                status=status,
                last_check=time.time(),
                response_time=response_time,
                error_count=error_count,
                metadata=metadata
            )
            
            # Store in history
            self.health_history.append(health_status)
            
            # Keep only last 1000 records
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-1000:]
            
            return asdict(health_status)
            
        except Exception as e:
            self.logger.error(f"Health check failed for {component_name}: {e}")
            return {
                'component': component_name,
                'status': 'critical',
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def register_health_check(self, component_name: str, check_func: Callable) -> None:
        """Register health check for component"""
        self.health_checks[component_name] = check_func
        self.logger.info(f"Registered health check for component: {component_name}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        # Check all registered components
        component_health = {}
        for component_name in self.health_checks.keys():
            component_health[component_name] = self.check_health(component_name)
        
        # Determine overall status
        statuses = [h.get('status', 'unknown') for h in component_health.values()]
        
        if 'critical' in statuses:
            overall_status = 'critical'
        elif 'warning' in statuses:
            overall_status = 'warning'
        elif 'healthy' in statuses:
            overall_status = 'healthy'
        else:
            overall_status = 'unknown'
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'components': component_health,
            'system_metrics': self._get_system_health_metrics()
        }
    
    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}


# Built-in health checks
def create_gpu_health_check():
    """Create GPU health check"""
    def check_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                # Check GPU memory
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
                
                usage_percent = (memory_allocated / memory_total) * 100
                
                if usage_percent > 90:
                    return {
                        'status': 'warning',
                        'metadata': {
                            'memory_allocated_mb': memory_allocated,
                            'memory_reserved_mb': memory_reserved,
                            'memory_total_mb': memory_total,
                            'usage_percent': usage_percent
                        }
                    }
                else:
                    return {
                        'status': 'healthy',
                        'metadata': {
                            'memory_allocated_mb': memory_allocated,
                            'memory_total_mb': memory_total,
                            'usage_percent': usage_percent
                        }
                    }
            else:
                return {'status': 'healthy', 'metadata': {'gpu_available': False}}
        except ImportError:
            return {'status': 'healthy', 'metadata': {'gpu_available': False}}
        except Exception as e:
            return {'status': 'critical', 'error': str(e)}
    
    return check_gpu


def create_memory_health_check():
    """Create memory health check"""
    def check_memory():
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                return {
                    'status': 'critical',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'available_mb': memory.available / 1024 / 1024
                    }
                }
            elif usage_percent > 80:
                return {
                    'status': 'warning',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'available_mb': memory.available / 1024 / 1024
                    }
                }
            else:
                return {
                    'status': 'healthy',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'available_mb': memory.available / 1024 / 1024
                    }
                }
        except Exception as e:
            return {'status': 'critical', 'error': str(e)}
    
    return check_memory


def create_disk_health_check():
    """Create disk health check"""
    def check_disk():
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent
            
            if usage_percent > 95:
                return {
                    'status': 'critical',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'free_gb': disk.free / 1024 / 1024 / 1024
                    }
                }
            elif usage_percent > 85:
                return {
                    'status': 'warning',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'free_gb': disk.free / 1024 / 1024 / 1024
                    }
                }
            else:
                return {
                    'status': 'healthy',
                    'metadata': {
                        'usage_percent': usage_percent,
                        'free_gb': disk.free / 1024 / 1024 / 1024
                    }
                }
        except Exception as e:
            return {'status': 'critical', 'error': str(e)}
    
    return check_disk


def create_monitoring_system(log_file: str = "performance_metrics.json") -> tuple:
    """Create complete monitoring system"""
    performance_monitor = PerformanceMonitorImpl(log_file)
    health_checker = HealthCheckImpl()
    
    # Register built-in health checks
    health_checker.register_health_check('gpu', create_gpu_health_check())
    health_checker.register_health_check('memory', create_memory_health_check())
    health_checker.register_health_check('disk', create_disk_health_check())
    
    return performance_monitor, health_checker
