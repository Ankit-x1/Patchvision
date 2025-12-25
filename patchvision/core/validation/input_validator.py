"""
Comprehensive input validation pipeline for PatchVision
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json


@dataclass
class ValidationResult:
    """Validation result with detailed information"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ValidationConfig:
    """Configuration for validation pipeline"""
    strict_mode: bool = False
    check_nan_inf: bool = True
    check_value_ranges: bool = True
    check_shapes: bool = True
    check_types: bool = True
    max_batch_size: int = 1024
    min_image_size: Tuple[int, int] = (32, 32)
    max_image_size: Tuple[int, int] = (4096, 4096)
    supported_formats: List[str] = None


class InputValidator:
    """
    Comprehensive input validation pipeline
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    def validate_image_input(self, image: np.ndarray, context: Optional[Dict] = None) -> ValidationResult:
        """Validate image input with comprehensive checks"""
        errors = []
        warnings = []
        metadata = {}
        
        # Check if image is numpy array
        if not isinstance(image, np.ndarray):
            errors.append("Input must be a numpy array")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check dimensions
        if len(image.shape) < 2 or len(image.shape) > 3:
            errors.append(f"Image must have 2 or 3 dimensions, got {len(image.shape)}")
        
        if len(image.shape) == 2:
            h, w = image.shape
            c = 1
        else:
            h, w, c = image.shape
            
        # Check minimum size
        if h < self.config.min_image_size[0] or w < self.config.min_image_size[1]:
            errors.append(f"Image too small: {h}x{w}, minimum {self.config.min_image_size}")
            
        # Check maximum size
        if h > self.config.max_image_size[0] or w > self.config.max_image_size[1]:
            warnings.append(f"Large image: {h}x{w}, may impact performance")
            
        # Check data type
        if self.config.check_types:
            valid_dtypes = [np.dtype(np.uint8), np.dtype(np.float32), np.dtype(np.float16)]
            if not any(image.dtype == valid_dtype for valid_dtype in valid_dtypes):
                errors.append(f"Unsupported dtype: {image.dtype}, expected uint8, float32, or float16")
        
        # Check for NaN/Inf values
        if self.config.check_nan_inf:
            if np.any(np.isnan(image)):
                errors.append("Image contains NaN values")
            if np.any(np.isinf(image)):
                errors.append("Image contains infinite values")
        
        # Check value ranges
        if self.config.check_value_ranges:
            if image.dtype == np.uint8:
                if np.min(image) < 0 or np.max(image) > 255:
                    errors.append(f"uint8 image values out of range [0, 255]: [{np.min(image)}, {np.max(image)}]")
            elif image.dtype in [np.float32, np.float16]:
                if np.min(image) < -1.0 or np.max(image) > 1.0:
                    warnings.append(f"Normalized image values out of range [-1, 1]: [{np.min(image)}, {np.max(image)}]")
        
        metadata.update({
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image))
        })
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_batch_input(self, batch: List[np.ndarray], context: Optional[Dict] = None) -> ValidationResult:
        """Validate batch of inputs"""
        errors = []
        warnings = []
        metadata = {}
        
        if not isinstance(batch, list):
            errors.append("Batch must be a list")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check batch size
        if len(batch) > self.config.max_batch_size:
            errors.append(f"Batch too large: {len(batch)}, maximum {self.config.max_batch_size}")
        
        # Validate each item in batch
        item_errors = []
        for i, item in enumerate(batch):
            result = self.validate_image_input(item, {**context, 'batch_index': i} if context else {'batch_index': i})
            if not result.is_valid:
                item_errors = [f"Item {i}: {err}" for err in result.errors]
                item_errors.extend(result.warnings)
                errors.extend(item_errors)
            warnings.extend([f"Item {i}: {warn}" for warn in result.warnings])
        
        metadata.update({
            'batch_size': len(batch),
            'total_items': len(batch),
            'total_size_mb': sum(item.nbytes for item in batch) / (1024 * 1024)
        })
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_model_input(self, model: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate model input"""
        errors = []
        warnings = []
        metadata = {}
        
        if model is None:
            errors.append("Model is None")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check model type
        try:
            import torch
            import tensorflow as tf
        except ImportError:
            torch = None
            tf = None
        
        # PyTorch model validation
        if torch is not None and hasattr(model, 'parameters'):
            try:
                param_count = sum(p.numel() for p in model.parameters())
                metadata.update({
                    'framework': 'pytorch',
                    'parameter_count': param_count,
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                })
            except Exception as e:
                warnings.append(f"Could not analyze PyTorch model: {e}")
        
        # TensorFlow model validation
        elif tf is not None and hasattr(model, 'variables'):
            try:
                var_count = len(model.variables)
                metadata.update({
                    'framework': 'tensorflow',
                    'variable_count': var_count
                })
            except Exception as e:
                warnings.append(f"Could not analyze TensorFlow model: {e}")
        
        # NumPy model validation
        elif isinstance(model, dict):
            required_keys = ['weights', 'config']
            missing_keys = [key for key in required_keys if key not in model]
            if missing_keys:
                errors.append(f"Missing model keys: {missing_keys}")
            
            if 'weights' in model:
                weights = model['weights']
                if not isinstance(weights, np.ndarray):
                    errors.append("Model weights must be numpy array")
                else:
                    metadata.update({
                        'framework': 'numpy',
                        'weight_shape': weights.shape,
                        'weight_size_mb': weights.nbytes / (1024 * 1024)
                    })
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_config_input(self, config: Dict[str, Any], context: Optional[Dict] = None) -> ValidationResult:
        """Validate configuration input"""
        errors = []
        warnings = []
        metadata = {}
        
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check required configuration sections
        required_sections = ['core', 'processors', 'patches', 'projections']
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            errors.append(f"Missing config sections: {missing_sections}")
        
        # Validate core configuration
        if 'core' in config:
            core_config = config['core']
            if not isinstance(core_config, dict):
                errors.append("Core configuration must be a dictionary")
            else:
                # Check batch size
                if 'batch_size' in core_config:
                    batch_size = core_config['batch_size']
                    if not isinstance(batch_size, int) or batch_size <= 0:
                        errors.append(f"Invalid batch_size: {batch_size}")
                    elif batch_size > self.config.max_batch_size:
                        warnings.append(f"Large batch_size: {batch_size}, may cause memory issues")
        
        metadata.update({
            'config_keys': list(config.keys()),
            'config_size': len(str(config))
        })
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_file_input(self, file_path: Union[str, Path], context: Optional[Dict] = None) -> ValidationResult:
        """Validate file input"""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return ValidationResult(False, errors, warnings, metadata)
        
        # Check file extension
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.npy']
        if file_path.suffix.lower() not in supported_extensions:
            warnings.append(f"Unsupported file extension: {file_path.suffix}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # 100MB limit
            warnings.append(f"Large file: {file_size_mb:.1f}MB, may impact performance")
        
        metadata.update({
            'file_path': str(file_path),
            'file_size_mb': file_size_mb,
            'file_extension': file_path.suffix
        })
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def validate_and_raise(self, data: Any, validation_type: str, context: Optional[Dict] = None) -> Any:
        """Validate data and raise exception if invalid"""
        if validation_type == 'image':
            result = self.validate_image_input(data, context)
        elif validation_type == 'batch':
            result = self.validate_batch_input(data, context)
        elif validation_type == 'model':
            result = self.validate_model_input(data, context)
        elif validation_type == 'config':
            result = self.validate_config_input(data, context)
        elif validation_type == 'file':
            result = self.validate_file_input(data, context)
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        if not result.is_valid:
            error_msg = "; ".join(result.errors)
            if self.config.strict_mode:
                raise ValueError(f"Validation failed: {error_msg}")
            else:
                self.logger.error(f"Validation failed: {error_msg}")
                for warning in result.warnings:
                    self.logger.warning(f"Validation warning: {warning}")
        
        return data
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        return {
            'validator_config': {
                'strict_mode': self.config.strict_mode,
                'checks_enabled': {
                    'nan_inf': self.config.check_nan_inf,
                    'value_ranges': self.config.check_value_ranges,
                    'shapes': self.config.check_shapes,
                    'types': self.config.check_types
                }
            }
        }


def create_validation_pipeline(strict_mode: bool = False) -> InputValidator:
    """Create configured validation pipeline"""
    config = ValidationConfig(strict_mode=strict_mode)
    return InputValidator(config)


# Decorator for automatic validation
def validate_input(validator: InputValidator, validation_type: str, context: Optional[Dict] = None):
    """Decorator for automatic input validation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract the first argument for validation
            if args:
                validator.validate_and_raise(args[0], validation_type, context)
            return func(*args, **kwargs)
        return wrapper
    return decorator
