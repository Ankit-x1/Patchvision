"""
Input and model validation for PatchVision
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """
    Input validation with comprehensive checking
    """
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def validate_image_input(self, 
                           data: np.ndarray,
                           expected_shape: Optional[Tuple[int, ...]] = None,
                           min_value: float = 0.0,
                           max_value: float = 255.0) -> bool:
        """
        Validate image input data
        """
        if not isinstance(data, np.ndarray):
            raise ValidationError("Input must be numpy array")
        
        if data.size == 0:
            raise ValidationError("Input array is empty")
        
        # Check shape
        if len(data.shape) not in [2, 3, 4]:
            raise ValidationError(f"Invalid input shape: {data.shape}. Expected 2D, 3D, or 4D array")
        
        # Check expected shape if provided
        if expected_shape:
            if len(data.shape) != len(expected_shape):
                raise ValidationError(f"Shape mismatch: expected {expected_shape}, got {data.shape}")
            
            for i, (actual, exp) in enumerate(zip(data.shape, expected_shape)):
                if exp != -1 and actual != exp:  # -1 means flexible dimension
                    raise ValidationError(f"Shape dimension {i} mismatch: expected {exp}, got {actual}")
        
        # Check data type
        if not np.issubdtype(data.dtype, np.floating) and not np.issubdtype(data.dtype, np.integer):
            raise ValidationError(f"Invalid data type: {data.dtype}. Expected numeric type")
        
        # Check value range
        data_min, data_max = data.min(), data.max()
        if data_min < min_value or data_max > max_value:
            if self.strict:
                raise ValidationError(f"Values out of range [{min_value}, {max_value}]: [{data_min}, {data_max}]")
            else:
                warnings.warn(f"Values out of range [{min_value}, {max_value}]: [{data_min}, {data_max}]")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(data)):
            raise ValidationError("Input contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValidationError("Input contains infinite values")
        
        return True
    
    def validate_batch_input(self,
                           data: Union[np.ndarray, List[np.ndarray]],
                           batch_size: Optional[int] = None) -> bool:
        """
        Validate batch input data
        """
        if isinstance(data, list):
            if not data:
                raise ValidationError("Batch list is empty")
            
            if batch_size and len(data) != batch_size:
                raise ValidationError(f"Batch size mismatch: expected {batch_size}, got {len(data)}")
            
            # Validate each item in batch
            first_shape = None
            for i, item in enumerate(data):
                self.validate_image_input(item)
                
                if first_shape is None:
                    first_shape = item.shape
                elif item.shape != first_shape:
                    raise ValidationError(f"Inconsistent shapes in batch: item {i} has shape {item.shape}, expected {first_shape}")
        
        elif isinstance(data, np.ndarray):
            if len(data.shape) < 3:
                raise ValidationError("Batch array must have at least 3 dimensions")
            
            if batch_size and data.shape[0] != batch_size:
                raise ValidationError(f"Batch size mismatch: expected {batch_size}, got {data.shape[0]}")
        
        return True
    
    def validate_tensor_input(self,
                            data: np.ndarray,
                            expected_dims: int,
                            min_dims: int = 1,
                            max_dims: int = 10) -> bool:
        """
        Validate tensor input for neural networks
        """
        if not isinstance(data, np.ndarray):
            raise ValidationError("Tensor input must be numpy array")
        
        if len(data.shape) < min_dims or len(data.shape) > max_dims:
            raise ValidationError(f"Invalid tensor dimensions: {len(data.shape)}. Expected between {min_dims} and {max_dims}")
        
        if expected_dims != -1 and len(data.shape) != expected_dims:
            raise ValidationError(f"Dimension mismatch: expected {expected_dims}, got {len(data.shape)}")
        
        # Check for valid tensor values
        if np.any(np.isnan(data)):
            raise ValidationError("Tensor contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValidationError("Tensor contains infinite values")
        
        return True
    
    def validate_model_output(self,
                            output: np.ndarray,
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            output_type: str = "classification") -> bool:
        """
        Validate model output
        """
        if not isinstance(output, np.ndarray):
            raise ValidationError("Model output must be numpy array")
        
        if output.size == 0:
            raise ValidationError("Model output is empty")
        
        # Check shape
        if expected_shape:
            if output.shape != expected_shape:
                raise ValidationError(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Validate based on output type
        if output_type == "classification":
            if len(output.shape) != 2:
                raise ValidationError(f"Classification output should be 2D, got {len(output.shape)}D")
            
            if np.any(output < 0) or np.any(output > 1):
                if self.strict:
                    raise ValidationError("Classification outputs should be between 0 and 1 (probabilities)")
                else:
                    warnings.warn("Classification outputs should be between 0 and 1 (probabilities)")
        
        elif output_type == "regression":
            if len(output.shape) != 2:
                raise ValidationError(f"Regression output should be 2D, got {len(output.shape)}D")
        
        return True


class ModelValidator:
    """
    Model validation and compatibility checking
    """
    
    def __init__(self):
        self.supported_formats = ['pytorch', 'tensorflow', 'numpy', 'onnx']
    
    def validate_model(self, model: Any) -> Dict[str, Any]:
        """
        Validate model and return metadata
        """
        metadata = {
            'format': self._detect_model_format(model),
            'is_valid': False,
            'issues': [],
            'parameters': 0,
            'input_shape': None,
            'output_shape': None
        }
        
        try:
            if metadata['format'] == 'pytorch':
                metadata.update(self._validate_pytorch_model(model))
            elif metadata['format'] == 'tensorflow':
                metadata.update(self._validate_tensorflow_model(model))
            elif metadata['format'] == 'numpy':
                metadata.update(self._validate_numpy_model(model))
            else:
                metadata['issues'].append(f"Unsupported model format: {metadata['format']}")
            
            metadata['is_valid'] = len(metadata['issues']) == 0
            
        except Exception as e:
            metadata['issues'].append(f"Validation error: {str(e)}")
        
        return metadata
    
    def _detect_model_format(self, model: Any) -> str:
        """Detect model format"""
        try:
            import torch
            if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
                return 'pytorch'
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if hasattr(model, 'save') or hasattr(model, 'get_weights'):
                return 'tensorflow'
        except ImportError:
            pass
        
        if isinstance(model, dict):
            return 'numpy'
        
        return 'unknown'
    
    def _validate_pytorch_model(self, model: Any) -> Dict[str, Any]:
        """Validate PyTorch model"""
        import torch
        
        info = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info['parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        
        # Check if model is in correct mode
        if model.training:
            info['issues'].append("Model is in training mode, should be in eval mode for inference")
        
        # Try to get input/output shapes (this is model-specific)
        try:
            # This is a simplified check - real implementation would need model-specific logic
            info['input_shape'] = "unknown"
            info['output_shape'] = "unknown"
        except:
            pass
        
        return info
    
    def _validate_tensorflow_model(self, model: Any) -> Dict[str, Any]:
        """Validate TensorFlow model"""
        import tensorflow as tf
        
        info = {}
        
        # Count parameters
        total_params = model.count_params()
        info['parameters'] = total_params
        
        # Get input/output shapes
        try:
            if hasattr(model, 'input_shape'):
                info['input_shape'] = model.input_shape
            if hasattr(model, 'output_shape'):
                info['output_shape'] = model.output_shape
        except:
            pass
        
        return info
    
    def _validate_numpy_model(self, model: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate NumPy model"""
        info = {}
        
        if not isinstance(model, dict):
            info['issues'].append("NumPy model must be a dictionary")
            return info
        
        # Count parameters
        total_params = sum(np.prod(arr.shape) for arr in model.values() if isinstance(arr, np.ndarray))
        info['parameters'] = int(total_params)
        
        # Check for required keys
        required_keys = ['conv1_weights', 'fc_weights']
        for key in required_keys:
            if key not in model:
                info['issues'].append(f"Missing required key: {key}")
        
        return info
    
    def validate_compatibility(self, 
                            model: Any,
                            input_shape: Tuple[int, ...]) -> bool:
        """
        Validate model compatibility with input shape
        """
        try:
            metadata = self.validate_model(model)
            
            if not metadata['is_valid']:
                return False
            
            # This is a simplified compatibility check
            # Real implementation would try to run a forward pass
            return True
            
        except Exception:
            return False
