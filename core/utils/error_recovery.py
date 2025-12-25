"""
Error recovery mechanisms for PatchVision
"""

import time
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryAction:
    """Recovery action configuration"""
    name: str
    action: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class ErrorRecoveryManager:
    """
    Comprehensive error recovery system for PatchVision
    """
    
    def __init__(self, log_file: str = "error_recovery.log"):
        self.log_file = Path(log_file)
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.retry_counts: Dict[str, int] = {}
        
        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_action(self, 
                           error_type: str, 
                           recovery_action: RecoveryAction):
        """Register a recovery action for specific error type"""
        if error_type not in self.recovery_actions:
            self.recovery_actions[error_type] = []
        
        self.recovery_actions[error_type].append(recovery_action)
        self.logger.info(f"Registered recovery action '{recovery_action.name}' for error type '{error_type}'")
    
    def handle_error(self, 
                   error: Exception, 
                   context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle error with appropriate recovery actions
        
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error
        self.logger.error(f"Error occurred: {error_type} - {error_message}")
        if context:
            self.logger.error(f"Context: {context}")
        
        # Record error in history
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message,
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.error_history.append(error_record)
        
        # Check retry count
        error_key = f"{error_type}_{id(context) if context else 'no_context'}"
        if error_key in self.retry_counts:
            self.retry_counts[error_key] += 1
        else:
            self.retry_counts[error_key] = 1
        
        # Find recovery actions
        if error_type in self.recovery_actions:
            for recovery_action in self.recovery_actions[error_type]:
                if self.retry_counts[error_key] <= recovery_action.max_retries:
                    self.logger.info(f"Attempting recovery action: {recovery_action.name}")
                    
                    try:
                        # Wait before retry
                        if self.retry_counts[error_key] > 1:
                            time.sleep(recovery_action.retry_delay)
                        
                        # Execute recovery action
                        result = recovery_action.action(error, context, self.retry_counts[error_key])
                        
                        if result:
                            self.logger.info(f"Recovery action '{recovery_action.name}' successful")
                            # Reset retry count on success
                            del self.retry_counts[error_key]
                            return True
                        
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery action failed: {recovery_error}")
                        continue
        
        self.logger.error(f"No recovery actions available for {error_type} or all retries exhausted")
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        # Count error types
        error_counts = {}
        for error in self.error_history:
            error_type = error['type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate rates
        total_errors = len(self.error_history)
        error_rates = {k: v/total_errors for k, v in error_counts.items()}
        
        return {
            'total_errors': total_errors,
            'error_counts': error_counts,
            'error_rates': error_rates,
            'most_common': max(error_counts.items(), key=lambda x: x[1]) if error_counts else None
        }
    
    def save_error_history(self, filename: str):
        """Save error history to file"""
        with open(filename, 'w') as f:
            json.dump(self.error_history, f, indent=2)
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.retry_counts.clear()
        self.logger.info("Error history cleared")


# Default recovery actions for common PatchVision errors
def gpu_memory_recovery(error: Exception, context: Dict, retry_count: int) -> bool:
    """Recover from GPU memory errors"""
    import gc
    import torch
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Reduce batch size if context contains it
    if context and 'batch_size' in context:
        new_batch_size = max(1, context['batch_size'] // 2)
        context['batch_size'] = new_batch_size
        print(f"Reduced batch size to {new_batch_size} for GPU memory recovery")
    
    return True


def model_loading_recovery(error: Exception, context: Dict, retry_count: int) -> bool:
    """Recover from model loading errors"""
    if "model_path" in context:
        model_path = Path(context["model_path"])
        
        # Try alternative paths
        alternatives = [
            model_path.parent / f"{model_path.stem}_backup{model_path.suffix}",
            model_path.parent / "models" / model_path.name,
            Path("models") / model_path.name
        ]
        
        for alt_path in alternatives:
            if alt_path.exists():
                context["model_path"] = str(alt_path)
                print(f"Using alternative model path: {alt_path}")
                return True
    
    return False


def camera_stream_recovery(error: Exception, context: Dict, retry_count: int) -> bool:
    """Recover from camera stream errors"""
    import cv2
    
    # Try to reinitialize camera
    if "camera_id" in context:
        camera_id = context["camera_id"]
        
        # Release current camera
        if "cap" in context and context["cap"] is not None:
            context["cap"].release()
        
        # Try to reconnect
        for attempt in range(3):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                context["cap"] = cap
                print(f"Camera reconnected on attempt {attempt + 1}")
                return True
            time.sleep(1.0)
    
    return False


def create_default_recovery_manager() -> ErrorRecoveryManager:
    """Create recovery manager with default actions"""
    manager = ErrorRecoveryManager()
    
    # Register default recovery actions
    manager.register_recovery_action(
        "CUDAoutofMemoryError",
        RecoveryAction("gpu_memory_recovery", gpu_memory_recovery, max_retries=2)
    )
    
    manager.register_recovery_action(
        "FileNotFoundError",
        RecoveryAction("model_loading_recovery", model_loading_recovery, max_retries=3)
    )
    
    manager.register_recovery_action(
        "cv2.error",
        RecoveryAction("camera_stream_recovery", camera_stream_recovery, max_retries=3)
    )
    
    manager.register_recovery_action(
        "ValueError",
        RecoveryAction("dimension_fix", lambda e, c, r: True, max_retries=1)
    )
    
    return manager


# Decorator for automatic error recovery
def with_recovery(recovery_manager: ErrorRecoveryManager, context: Optional[Dict] = None):
    """Decorator for automatic error recovery"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not recovery_manager.handle_error(e, context):
                        # Re-raise if recovery failed
                        raise e
                    # Continue loop if recovery succeeded
        return wrapper
    return decorator
