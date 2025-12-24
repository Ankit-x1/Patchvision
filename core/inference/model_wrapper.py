"""
Model wrapper for unified interface across frameworks
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    warnings.warn("TensorFlow not available")


class ModelWrapper:
    """
    Unified interface for different deep learning frameworks
    """
    
    def __init__(self, model: Any, framework: str = "auto"):
        self.model = model
        self.framework = self._detect_framework(model, framework)
        self.is_training = False
        self._setup_model()
    
    def _detect_framework(self, model: Any, framework: str) -> str:
        """Detect model framework"""
        if framework != "auto":
            return framework
        
        if HAS_TORCH and isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            return "pytorch"
        elif HAS_TF and hasattr(model, 'predict'):
            return "tensorflow"
        elif isinstance(model, dict):
            return "numpy"
        else:
            return "numpy"
    
    def _setup_model(self):
        """Setup model based on framework"""
        if self.framework == "pytorch":
            self.model.eval()
        elif self.framework == "tensorflow":
            # TensorFlow models are ready by default
            pass
    
    def predict(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Unified prediction interface"""
        if self.framework == "pytorch":
            return self._predict_pytorch(inputs, **kwargs)
        elif self.framework == "tensorflow":
            return self._predict_tensorflow(inputs, **kwargs)
        else:
            return self._predict_numpy(inputs, **kwargs)
    
    def _predict_pytorch(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """PyTorch prediction"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor
            if len(inputs.shape) == 3:  # Single sample
                inputs = inputs[None, ...]
            
            # Handle channel format
            if inputs.shape[1] not in [1, 3]:  # Likely channel-last
                inputs = np.transpose(inputs, (0, 3, 1, 2))
            
            inputs_tensor = torch.from_numpy(inputs).float()
            
            # Forward pass
            outputs = self.model(inputs_tensor)
            
            # Convert back to numpy
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            return outputs.cpu().numpy()
    
    def _predict_tensorflow(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """TensorFlow prediction"""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        # TensorFlow handles channel-last format
        if len(inputs.shape) == 3:  # Single sample
            inputs = inputs[None, ...]
        
        return self.model(inputs, training=False, **kwargs).numpy()
    
    def _predict_numpy(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """NumPy-based prediction"""
        if not isinstance(self.model, dict):
            raise ValueError("NumPy model must be a dictionary with weights")
        
        if len(inputs.shape) == 3:  # Single sample
            inputs = inputs[None, ...]
        
        batch_size = inputs.shape[0]
        num_classes = self.model['fc_weights'].shape[1]
        outputs = np.zeros((batch_size, num_classes))
        
        # Simple forward pass
        for i in range(batch_size):
            x = inputs[i]
            
            # Global average pooling
            x = np.mean(x, axis=(1, 2))
            
            # Fully connected layer
            outputs[i] = np.dot(x, self.model['fc_weights'].T) + self.model['fc_bias']
        
        return outputs
    
    def train(self):
        """Set model to training mode"""
        self.is_training = True
        if self.framework == "pytorch":
            self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.is_training = False
        if self.framework == "pytorch":
            self.model.eval()
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters"""
        if self.framework == "pytorch":
            return {name: param.cpu().numpy() 
                    for name, param in self.model.named_parameters()}
        elif self.framework == "tensorflow":
            return {weight.name: weight.numpy() 
                    for weight in self.model.weights}
        else:
            return self.model.copy()
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set model parameters"""
        if self.framework == "pytorch":
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in parameters:
                        param.copy_(torch.from_numpy(parameters[name]))
        elif self.framework == "tensorflow":
            for weight in self.model.weights:
                if weight.name in parameters:
                    weight.assign(parameters[weight.name])
        else:
            self.model.update(parameters)
    
    def save(self, path: str):
        """Save model"""
        if self.framework == "pytorch":
            torch.save(self.model.state_dict(), path)
        elif self.framework == "tensorflow":
            self.model.save(path)
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model"""
        if self.framework == "pytorch":
            self.model.load_state_dict(torch.load(path))
        elif self.framework == "tensorflow":
            self.model = tf.keras.models.load_model(path)
        else:
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'framework': self.framework,
            'is_training': self.is_training,
            'parameters_count': 0
        }
        
        if self.framework == "pytorch":
            info['parameters_count'] = sum(
                param.numel() for param in self.model.parameters()
            )
        elif self.framework == "tensorflow":
            info['parameters_count'] = sum(
                np.prod(weight.shape) for weight in self.model.weights
            )
        elif isinstance(self.model, dict):
            info['parameters_count'] = sum(
                np.prod(self.model[key].shape) 
                for key in self.model.keys() 
                if 'weights' in key
            )
        
        return info
