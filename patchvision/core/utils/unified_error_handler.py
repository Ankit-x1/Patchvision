"""
Unified error handling framework for PatchVision
"""

import logging
import traceback
import json
import time
from typing import Dict, Any, Optional, Callable, List, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from ..interfaces import ErrorHandler


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    IO = "io"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors"""
    component: str
    operation: str
    input_data: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ErrorRecord:
    """Complete error record"""
    timestamp: float
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    context: ErrorContext
    traceback: str
    recovery_attempts: int
    resolved: bool
    resolution_time: Optional[float] = None


class UnifiedErrorHandler(ErrorHandler):
    """Unified error handling framework"""
    
    def __init__(self, log_file: str = "unified_errors.log"):
        self.log_file = Path(log_file)
        self.error_records: List[ErrorRecord] = []
        self.recovery_actions: Dict[str, List[Callable]] = {}
        self.error_callbacks: List[Callable] = []
        self.error_stats: Dict[str, Any] = {
            'total_errors': 0,
            'by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'by_category': {category.value: 0 for category in ErrorCategory},
            'by_component': {}
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('patchvision_errors')
        logger.setLevel(logging.DEBUG)
        
        # File handler for all errors
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for critical errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error with context"""
        # Generate error ID
        error_id = f"ERR_{int(time.time() * 1000)}"
        
        # Determine error category and severity
        category = self._categorize_error(error)
        severity = self._determine_severity(error, context)
        
        # Create error context
        error_context = ErrorContext(
            component=context.get('component', 'unknown'),
            operation=context.get('operation', 'unknown'),
            input_data=context.get('input_data'),
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            metadata=context.get('metadata', {})
        )
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=time.time(),
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(error),
            exception_type=type(error).__name__,
            context=error_context,
            traceback=traceback.format_exc(),
            recovery_attempts=0,
            resolved=False
        )
        
        # Store error record
        self.error_records.append(error_record)
        
        # Update statistics
        self._update_stats(error_record)
        
        # Log error
        self._log_error(error_record)
        
        # Attempt recovery
        recovery_success = self._attempt_recovery(error, context, error_record)
        
        # Notify callbacks
        self._notify_callbacks(error_record)
        
        return recovery_success
    
    def register_recovery_action(self, error_type: str, action: Callable) -> None:
        """Register a recovery action for error type"""
        if error_type not in self.recovery_actions:
            self.recovery_actions[error_type] = []
        
        self.recovery_actions[error_type].append(action)
        self.logger.info(f"Registered recovery action for {error_type}")
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback for notifications"""
        self.error_callbacks.append(callback)
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get error history"""
        return [asdict(record) for record in self.error_records]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return self.error_stats.copy()
    
    def generate_error_report(self, output_path: str) -> bool:
        """Generate comprehensive error report"""
        try:
            report = {
                'generated_at': time.time(),
                'summary': self.error_stats,
                'recent_errors': [asdict(r) for r in self.error_records[-100:]],  # Last 100 errors
                'error_patterns': self._analyze_error_patterns(),
                'recommendations': self._generate_recommendations()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Error report generated: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate error report: {e}")
            return False
    
    def clear_error_history(self, older_than_hours: Optional[int] = None) -> None:
        """Clear error history"""
        if older_than_hours is None:
            self.error_records.clear()
            self.error_stats = {
                'total_errors': 0,
                'by_severity': {severity.value: 0 for severity in ErrorSeverity},
                'by_category': {category.value: 0 for category in ErrorCategory},
                'by_component': {}
            }
        else:
            cutoff_time = time.time() - (older_than_hours * 3600)
            self.error_records = [
                r for r in self.error_records 
                if r.timestamp > cutoff_time
            ]
            # Recalculate stats
            self._recalculate_stats()
        
        self.logger.info("Error history cleared")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Validation errors
        if any(keyword in error_type.lower() for keyword in ['valueerror', 'typeerror']):
            return ErrorCategory.VALIDATION
        
        # IO errors
        if any(keyword in error_type.lower() for keyword in ['file', 'path', 'directory']):
            return ErrorCategory.IO
        
        # Memory errors
        if any(keyword in error_message for keyword in ['memory', 'cuda', 'gpu', 'oom']):
            return ErrorCategory.MEMORY
        
        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK
        
        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        # Processing errors (default)
        return ErrorCategory.PROCESSING
    
    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if any(keyword in error_type.lower() for keyword in ['critical', 'fatal']):
            return ErrorSeverity.CRITICAL
        
        # Memory errors are usually critical
        if 'memory' in str(error).lower() or 'cuda' in str(error).lower():
            return ErrorSeverity.CRITICAL
        
        # High severity
        if any(keyword in error_type.lower() for keyword in ['runtime', 'attribute']):
            return ErrorSeverity.HIGH
        
        # Medium severity (default)
        return ErrorSeverity.MEDIUM
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any], error_record: ErrorRecord) -> bool:
        """Attempt error recovery"""
        error_type = type(error).__name__
        
        if error_type in self.recovery_actions:
            for i, action in enumerate(self.recovery_actions[error_type]):
                try:
                    self.logger.info(f"Attempting recovery action {i+1} for {error_type}")
                    result = action(error, context, i+1)
                    
                    if result:
                        error_record.resolved = True
                        error_record.resolution_time = time.time() - error_record.timestamp
                        error_record.recovery_attempts = i + 1
                        self.logger.info(f"Recovery successful for {error_type}")
                        return True
                        
                except Exception as recovery_error:
                    self.logger.error(f"Recovery action failed: {recovery_error}")
                    continue
        
        return False
    
    def _notify_callbacks(self, error_record: ErrorRecord) -> None:
        """Notify all error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error_record)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")
    
    def _update_stats(self, error_record: ErrorRecord) -> None:
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        self.error_stats['by_severity'][error_record.severity.value] += 1
        self.error_stats['by_category'][error_record.category.value] += 1
        
        component = error_record.context.component
        if component not in self.error_stats['by_component']:
            self.error_stats['by_component'][component] = 0
        self.error_stats['by_component'][component] += 1
    
    def _recalculate_stats(self) -> None:
        """Recalculate statistics from error records"""
        self.error_stats = {
            'total_errors': len(self.error_records),
            'by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'by_category': {category.value: 0 for category in ErrorCategory},
            'by_component': {}
        }
        
        for record in self.error_records:
            self._update_stats(record)
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error with appropriate level"""
        log_message = (
            f"[{error_record.error_id}] "
            f"{error_record.context.component}.{error_record.context.operation}: "
            f"{error_record.message}"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        if not self.error_records:
            return {}
        
        # Most common errors
        error_counts = {}
        for record in self.error_records:
            key = f"{record.exception_type}:{record.context.operation}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        # Error frequency by time
        recent_errors = [r for r in self.error_records if time.time() - r.timestamp < 3600]  # Last hour
        error_rate = len(recent_errors) / 3600  # Errors per second
        
        return {
            'most_common': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'error_rate_per_hour': error_rate * 3600,
            'recovery_success_rate': sum(1 for r in self.error_records if r.resolved) / len(self.error_records) * 100
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        # Check for high error rates
        if self.error_stats['total_errors'] > 100:
            recommendations.append("Consider implementing more robust error handling and validation")
        
        # Check for memory errors
        memory_errors = self.error_stats['by_category'][ErrorCategory.MEMORY.value]
        if memory_errors > 10:
            recommendations.append("Review memory usage and implement better memory management")
        
        # Check for validation errors
        validation_errors = self.error_stats['by_category'][ErrorCategory.VALIDATION.value]
        if validation_errors > 20:
            recommendations.append("Strengthen input validation and add pre-processing checks")
        
        # Check for IO errors
        io_errors = self.error_stats['by_category'][ErrorCategory.IO.value]
        if io_errors > 15:
            recommendations.append("Implement better file handling and path validation")
        
        return recommendations


# Decorators for automatic error handling
def handle_errors(error_handler: UnifiedErrorHandler, component_name: str):
    """Decorator for automatic error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'component': component_name,
                    'operation': func.__name__,
                    'input_data': {'args_count': len(args), 'kwargs': list(kwargs.keys())}
                }
                if not error_handler.handle_error(e, context):
                    # Re-raise if recovery failed
                    raise e
        return wrapper
    return decorator


def create_unified_error_handler(log_file: str = "unified_errors.log") -> UnifiedErrorHandler:
    """Create configured unified error handler"""
    return UnifiedErrorHandler(log_file)


# Global error handler instance
global_error_handler = None


def get_global_error_handler() -> UnifiedErrorHandler:
    """Get global error handler instance"""
    global global_error_handler
    if global_error_handler is None:
        global_error_handler = create_unified_error_handler()
    return global_error_handler


def handle_global_error(error: Exception, context: Dict[str, Any]) -> bool:
    """Handle error using global error handler"""
    handler = get_global_error_handler()
    return handler.handle_error(error, context)
