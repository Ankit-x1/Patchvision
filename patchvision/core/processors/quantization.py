"""
Quantization support for PatchVision models
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.quantization as tq
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available, some quantization features disabled")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    warnings.warn("TensorFlow not available, some quantization features disabled")


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    method: str = "dynamic"  # dynamic, static, qat
    precision: str = "int8"   # int8, fp16, bfloat16
    calibration_data: Optional[np.ndarray] = None
    per_channel: bool = False
    symmetric: bool = True
    clip_range: Optional[Tuple[float, float]] = None


class ModelQuantizer:
    """
    Advanced model quantization with multiple precision support
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_stats = {}
        self.scales = {}
        self.zero_points = {}
    
    def quantize_model(self, 
                      model: Any,
                      framework: str = "auto") -> Any:
        """
        Quantize model based on configuration
        """
        if framework == "auto":
            framework = self._detect_framework(model)
        
        if framework == "pytorch":
            return self._quantize_pytorch(model)
        elif framework == "tensorflow":
            return self._quantize_tensorflow(model)
        elif framework == "numpy":
            return self._quantize_numpy(model)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect model framework"""
        if HAS_TORCH and isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            return "pytorch"
        elif HAS_TF and hasattr(model, 'save'):
            return "tensorflow"
        elif isinstance(model, (np.ndarray, dict)):
            return "numpy"
        else:
            return "numpy"
    
    def _quantize_pytorch(self, model: nn.Module) -> nn.Module:
        """Quantize PyTorch model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        if self.config.method == "dynamic":
            return self._dynamic_quantize_pytorch(model)
        elif self.config.method == "static":
            return self._static_quantize_pytorch(model)
        elif self.config.method == "qat":
            return self._qat_quantize_pytorch(model)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
    
    def _dynamic_quantize_pytorch(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization for PyTorch"""
        if self.config.precision == "int8":
            quantized_model = tq.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif self.config.precision == "fp16":
            quantized_model = model.half()
        else:
            raise ValueError(f"Unsupported precision: {self.config.precision}")
        
        return quantized_model
    
    def _static_quantize_pytorch(self, model: nn.Module) -> nn.Module:
        """Static quantization for PyTorch"""
        model.eval()
        
        # Prepare model for quantization
        model.qconfig = tq.get_default_qconfig('fbgemm')
        tq.prepare(model, inplace=True)
        
        # Calibrate if data provided
        if self.config.calibration_data is not None:
            self._calibrate_pytorch(model, self.config.calibration_data)
        
        # Convert to quantized model
        quantized_model = tq.convert(model, inplace=False)
        return quantized_model
    
    def _qat_quantize_pytorch(self, model: nn.Module) -> nn.Module:
        """Quantization-aware training for PyTorch"""
        model.train()
        
        # Prepare for QAT
        model.qconfig = tq.get_default_qat_qconfig('fbgemm')
        tq.prepare_qat(model, inplace=True)
        
        return model
    
    def _calibrate_pytorch(self, model: nn.Module, data: np.ndarray):
        """Calibrate model for static quantization"""
        model.eval()
        with torch.no_grad():
            for sample in data:
                if isinstance(sample, np.ndarray):
                    sample = torch.from_numpy(sample)
                model(sample)
    
    def _quantize_tensorflow(self, model: Any) -> Any:
        """Quantize TensorFlow model"""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        if self.config.precision == "int8":
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if self.config.calibration_data is not None:
                def representative_dataset():
                    for data in self.config.calibration_data:
                        yield [data.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            return quantized_model
        
        elif self.config.precision == "fp16":
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            quantized_model = converter.convert()
            return quantized_model
        
        else:
            raise ValueError(f"Unsupported precision: {self.config.precision}")
    
    def _quantize_numpy(self, model: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
        """Quantize numpy arrays"""
        if isinstance(model, np.ndarray):
            return self._quantize_numpy_array(model)
        elif isinstance(model, dict):
            return {k: self._quantize_numpy_array(v) for k, v in model.items()}
        else:
            return model
    
    def _quantize_numpy_array(self, arr: np.ndarray) -> np.ndarray:
        """Quantize individual numpy array"""
        if self.config.precision == "int8":
            return self._quantize_to_int8(arr)
        elif self.config.precision == "fp16":
            return arr.astype(np.float16)
        elif self.config.precision == "bfloat16":
            return arr.astype(np.bfloat16) if hasattr(np, 'bfloat16') else arr.astype(np.float16)
        else:
            return arr
    
    def _quantize_to_int8(self, arr: np.ndarray) -> np.ndarray:
        """Quantize array to INT8"""
        if self.config.calibration_data is not None:
            # Use calibration data for better quantization
            min_val, max_val = self._compute_calibration_range(arr)
        else:
            min_val, max_val = arr.min(), arr.max()
        
        # Apply clip range if specified
        if self.config.clip_range:
            min_val = max(min_val, self.config.clip_range[0])
            max_val = min(max_val, self.config.clip_range[1])
        
        # Compute scale and zero point
        scale = (max_val - min_val) / 255.0
        zero_point = int(round(-min_val / scale)) if scale > 0 else 0
        
        # Quantize
        quantized = np.round(arr / scale + zero_point)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        # Store quantization parameters
        self.scales[str(id(arr))] = scale
        self.zero_points[str(id(arr))] = zero_point
        
        return quantized
    
    def _compute_calibration_range(self, arr: np.ndarray) -> Tuple[float, float]:
        """Compute calibration range from calibration data"""
        if isinstance(self.config.calibration_data, np.ndarray):
            return self.config.calibration_data.min(), self.config.calibration_data.max()
        else:
            return arr.min(), arr.max()
    
    def dequantize_model(self, quantized_model: Any, framework: str = "auto") -> Any:
        """Dequantize model back to original precision"""
        if framework == "auto":
            framework = self._detect_framework(quantized_model)
        
        if framework == "numpy":
            return self._dequantize_numpy(quantized_model)
        else:
            warnings.warn(f"Dequantization not implemented for framework: {framework}")
            return quantized_model
    
    def _dequantize_numpy(self, model: Union[np.ndarray, Dict]) -> Union[np.ndarray, Dict]:
        """Dequantize numpy arrays"""
        if isinstance(model, np.ndarray):
            return self._dequantize_numpy_array(model)
        elif isinstance(model, dict):
            return {k: self._dequantize_numpy_array(v) for k, v in model.items()}
        else:
            return model
    
    def _dequantize_numpy_array(self, arr: np.ndarray) -> np.ndarray:
        """Dequantize individual numpy array"""
        if arr.dtype == np.uint8:
            arr_id = str(id(arr))
            if arr_id in self.scales and arr_id in self.zero_points:
                scale = self.scales[arr_id]
                zero_point = self.zero_points[arr_id]
                return (arr.astype(np.float32) - zero_point) * scale
            else:
                # Fallback dequantization
                return arr.astype(np.float32) / 255.0
        elif arr.dtype == np.float16:
            return arr.astype(np.float32)
        else:
            return arr
    
    def save_quantization_params(self, file_path: str):
        """Save quantization parameters"""
        params = {
            'scales': self.scales,
            'zero_points': self.zero_points,
            'config': {
                'method': self.config.method,
                'precision': self.config.precision,
                'per_channel': self.config.per_channel,
                'symmetric': self.config.symmetric
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def load_quantization_params(self, file_path: str):
        """Load quantization parameters"""
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        self.scales = params['scales']
        self.zero_points = params['zero_points']
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics"""
        return {
            'num_quantized_arrays': len(self.scales),
            'precision': self.config.precision,
            'method': self.config.method,
            'memory_reduction': self._estimate_memory_reduction()
        }
    
    def _estimate_memory_reduction(self) -> float:
        """Estimate memory reduction from quantization"""
        if self.config.precision == "int8":
            return 0.75  # 75% reduction from float32 to int8
        elif self.config.precision == "fp16":
            return 0.5   # 50% reduction from float32 to fp16
        else:
            return 0.0


class BatchQuantizer:
    """
    Batch processing quantization for large datasets
    """
    
    def __init__(self, config: QuantizationConfig, batch_size: int = 32):
        self.config = config
        self.batch_size = batch_size
        self.quantizer = ModelQuantizer(config)
    
    def quantize_batch(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Quantize a batch of arrays"""
        quantized = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            quantized_batch = [self.quantizer._quantize_numpy_array(arr) for arr in batch]
            quantized.extend(quantized_batch)
        
        return quantized
    
    def calibrate_and_quantize(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Calibrate on subset then quantize all data"""
        # Use first batch for calibration
        calibration_data = data[:self.batch_size]
        self.config.calibration_data = np.concatenate(calibration_data)
        
        # Quantize all data
        return self.quantize_batch(data)
