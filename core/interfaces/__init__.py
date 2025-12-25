"""
Abstract interfaces for major PatchVision components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np


class ModelWrapper(ABC):
    """Abstract interface for model wrappers"""
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load model from path"""
        pass
    
    @abstractmethod
    def predict(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Run inference on inputs"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        pass
    
    @abstractmethod
    def save_model(self, save_path: str, **kwargs) -> bool:
        """Save model to path"""
        pass


class BatchProcessor(ABC):
    """Abstract interface for batch processing"""
    
    @abstractmethod
    def process_batch(self, batch_data: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """Process a batch of data"""
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """Get optimal batch size"""
        pass
    
    @abstractmethod
    def configure_batch(self, batch_size: int, **kwargs) -> None:
        """Configure batch processing parameters"""
        pass


class PerformanceMonitor(ABC):
    """Abstract interface for performance monitoring"""
    
    @abstractmethod
    def start_monitoring(self, component_name: str) -> None:
        """Start monitoring a component"""
        pass
    
    @abstractmethod
    def record_metric(self, metric_name: str, value: float, **metadata) -> None:
        """Record a performance metric"""
        pass
    
    @abstractmethod
    def get_metrics(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        pass
    
    @abstractmethod
    def generate_report(self, output_path: str) -> bool:
        """Generate performance report"""
        pass


class ErrorHandler(ABC):
    """Abstract interface for error handling"""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error with context"""
        pass
    
    @abstractmethod
    def register_recovery_action(self, error_type: str, action: callable) -> None:
        """Register a recovery action for error type"""
        pass
    
    @abstractmethod
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get error history"""
        pass


class DataValidator(ABC):
    """Abstract interface for data validation"""
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> bool:
        """Validate data and return result"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules"""
        pass
    
    @abstractmethod
    def add_validation_rule(self, rule_name: str, rule: callable) -> None:
        """Add a custom validation rule"""
        pass


class ConfigurationInterface(ABC):
    """Abstract interface for configuration management"""
    
    @abstractmethod
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], output_path: str) -> bool:
        """Save configuration to file"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        pass


class PipelineInterface(ABC):
    """Abstract interface for processing pipelines"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize pipeline with configuration"""
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process data through pipeline"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        pass


class OptimizationInterface(ABC):
    """Abstract interface for optimization strategies"""
    
    @abstractmethod
    def optimize(self, data: Any, **kwargs) -> Any:
        """Optimize data or model"""
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information"""
        pass


class StorageInterface(ABC):
    """Abstract interface for data storage"""
    
    @abstractmethod
    def store(self, key: str, data: Any, **metadata) -> bool:
        """Store data with key"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str, **kwargs) -> Any:
        """Retrieve data by key"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass
    
    @abstractmethod
    def list_keys(self, **filters) -> List[str]:
        """List available keys"""
        pass


class LoggingInterface(ABC):
    """Abstract interface for logging"""
    
    @abstractmethod
    def log(self, level: str, message: str, **metadata) -> None:
        """Log message with level"""
        pass
    
    @abstractmethod
    def set_level(self, level: str) -> None:
        """Set logging level"""
        pass
    
    @abstractmethod
    def add_handler(self, handler: Any) -> None:
        """Add log handler"""
        pass


class MetricsInterface(ABC):
    """Abstract interface for metrics collection"""
    
    @abstractmethod
    def increment_counter(self, metric_name: str, value: int = 1, **tags) -> None:
        """Increment a counter metric"""
        pass
    
    @abstractmethod
    def record_histogram(self, metric_name: str, value: float, **tags) -> None:
        """Record a histogram metric"""
        pass
    
    @abstractmethod
    def record_gauge(self, metric_name: str, value: float, **tags) -> None:
        """Record a gauge metric"""
        pass
    
    @abstractmethod
    def flush_metrics(self) -> None:
        """Flush metrics to storage"""
        pass


class HealthCheckInterface(ABC):
    """Abstract interface for health checks"""
    
    @abstractmethod
    def check_health(self, component_name: str) -> Dict[str, Any]:
        """Check health of a component"""
        pass
    
    @abstractmethod
    def register_health_check(self, component_name: str, check_func: callable) -> None:
        """Register health check for component"""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        pass


# Factory classes for creating instances
class ComponentFactory:
    """Factory for creating component instances"""
    
    @staticmethod
    def create_model_wrapper(framework: str) -> ModelWrapper:
        """Create model wrapper for specified framework"""
        if framework == 'pytorch':
            from ..models.pytorch_wrapper import PyTorchWrapper
            return PyTorchWrapper()
        elif framework == 'tensorflow':
            from ..models.tensorflow_wrapper import TensorFlowWrapper
            return TensorFlowWrapper()
        elif framework == 'numpy':
            from ..models.numpy_wrapper import NumPyWrapper
            return NumPyWrapper()
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def create_batch_processor(processor_type: str) -> BatchProcessor:
        """Create batch processor for specified type"""
        if processor_type == 'optimized':
            from ..processors.engine import OptimizedProcessor
            return OptimizedProcessor()
        elif processor_type == 'standard':
            from ..processors.standard import StandardProcessor
            return StandardProcessor()
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")
    
    @staticmethod
    def create_performance_monitor(monitor_type: str) -> PerformanceMonitor:
        """Create performance monitor for specified type"""
        if monitor_type == 'basic':
            from ..analytics.monitor import BasicMonitor
            return BasicMonitor()
        elif monitor_type == 'advanced':
            from ..analytics.advanced_monitor import AdvancedMonitor
            return AdvancedMonitor()
        else:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")


# Registry for component registration
class ComponentRegistry:
    """Registry for component registration and discovery"""
    
    def __init__(self):
        self._components = {}
        self._interfaces = {}
    
    def register_component(self, name: str, component_class: type, interface: type) -> None:
        """Register a component class"""
        self._components[name] = component_class
        self._interfaces[name] = interface
    
    def get_component(self, name: str) -> type:
        """Get component class by name"""
        if name not in self._components:
            raise ValueError(f"Component not registered: {name}")
        return self._components[name]
    
    def get_interface(self, name: str) -> type:
        """Get interface for component"""
        if name not in self._interfaces:
            raise ValueError(f"Component not registered: {name}")
        return self._interfaces[name]
    
    def list_components(self, interface_type: Optional[type] = None) -> List[str]:
        """List registered components"""
        if interface_type is None:
            return list(self._components.keys())
        
        return [name for name, iface in self._interfaces.items() if iface == interface_type]
    
    def validate_component(self, component: Any, expected_interface: type) -> bool:
        """Validate component implements expected interface"""
        return isinstance(component, expected_interface)


# Global registry instance
component_registry = ComponentRegistry()
