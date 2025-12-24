"""
Model serialization utilities for PatchVision
"""

import json
import pickle
import gzip
import hashlib
import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
import warnings
from dataclasses import dataclass
import time


@dataclass
class SerializationMetadata:
    """Metadata for serialized models"""
    format: str
    version: str
    created_at: float
    model_hash: str
    compression: bool
    framework: str
    size_bytes: int


class ModelSerializer:
    """
    Advanced model serialization with multiple formats and compression
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.supported_formats = ['pickle', 'json', 'numpy', 'torch', 'tensorflow']
        self.serializer_version = "1.0"
    
    def serialize(self, 
                  model: Any,
                  file_path: str,
                  format: str = "auto",
                  compress: bool = True,
                  metadata: Optional[Dict[str, Any]] = None) -> SerializationMetadata:
        """
        Serialize model to file
        
        Args:
            model: Model to serialize
            file_path: Output file path
            format: Serialization format
            compress: Whether to compress the output
            metadata: Additional metadata
            
        Returns:
            Serialization metadata
        """
        if format == "auto":
            format = self._detect_format(model)
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        # Prepare metadata
        model_hash = self._compute_model_hash(model)
        start_time = time.time()
        
        # Serialize based on format
        if format == "pickle":
            data = self._serialize_pickle(model)
        elif format == "json":
            data = self._serialize_json(model)
        elif format == "numpy":
            data = self._serialize_numpy(model)
        elif format == "torch":
            data = self._serialize_torch(model)
        elif format == "tensorflow":
            data = self._serialize_tensorflow(model)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Apply compression if requested
        if compress:
            data = gzip.compress(data, compresslevel=self.compression_level)
            file_path += ".gz"
        
        # Write to file
        with open(file_path, 'wb') as f:
            f.write(data)
        
        # Create metadata
        file_size = Path(file_path).stat().st_size
        serialization_metadata = SerializationMetadata(
            format=format,
            version=self.serializer_version,
            created_at=time.time(),
            model_hash=model_hash,
            compression=compress,
            framework=self._detect_framework(model),
            size_bytes=file_size
        )
        
        # Save metadata
        metadata_path = file_path + ".meta"
        metadata_dict = {
            **asdict(serialization_metadata),
            'custom_metadata': metadata or {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return serialization_metadata
    
    def deserialize(self, 
                    file_path: str,
                    format: Optional[str] = None,
                    verify_hash: bool = True) -> Tuple[Any, SerializationMetadata]:
        """
        Deserialize model from file
        
        Args:
            file_path: Input file path
            format: Serialization format (if None, auto-detect)
            verify_hash: Whether to verify model hash
            
        Returns:
            Tuple of (model, metadata)
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            # Try with .gz extension
            gz_path = Path(str(file_path) + ".gz")
            if gz_path.exists():
                file_path = gz_path
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load metadata
        metadata_path = Path(str(file_path) + ".meta")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = SerializationMetadata(**{k: v for k, v in metadata_dict.items() 
                                           if k in SerializationMetadata.__annotations__})
        
        # Determine format
        if format is None:
            format = metadata.format
        
        # Read file data
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Decompress if needed
        if metadata.compression:
            data = gzip.decompress(data)
        
        # Deserialize based on format
        if format == "pickle":
            model = self._deserialize_pickle(data)
        elif format == "json":
            model = self._deserialize_json(data)
        elif format == "numpy":
            model = self._deserialize_numpy(data)
        elif format == "torch":
            model = self._deserialize_torch(data)
        elif format == "tensorflow":
            model = self._deserialize_tensorflow(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Verify hash if requested
        if verify_hash:
            current_hash = self._compute_model_hash(model)
            if current_hash != metadata.model_hash:
                warnings.warn(f"Model hash mismatch: expected {metadata.model_hash}, got {current_hash}")
        
        return model, metadata
    
    def _detect_format(self, model: Any) -> str:
        """Auto-detect best serialization format"""
        try:
            import torch
            if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
                return "torch"
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if hasattr(model, 'save') or hasattr(model, 'get_weights'):
                return "tensorflow"
        except ImportError:
            pass
        
        if isinstance(model, (np.ndarray, dict)):
            return "numpy"
        elif isinstance(model, (int, float, str, list, tuple)):
            return "json"
        else:
            return "pickle"
    
    def _detect_framework(self, model: Any) -> str:
        """Detect model framework"""
        try:
            import torch
            if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
                return "pytorch"
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if hasattr(model, 'save') or hasattr(model, 'get_weights'):
                return "tensorflow"
        except ImportError:
            pass
        
        if isinstance(model, np.ndarray):
            return "numpy"
        else:
            return "generic"
    
    def _compute_model_hash(self, model: Any) -> str:
        """Compute hash of model for integrity checking"""
        try:
            if isinstance(model, np.ndarray):
                return hashlib.sha256(model.tobytes()).hexdigest()
            elif isinstance(model, dict):
                hash_input = json.dumps(model, sort_keys=True).encode()
                return hashlib.sha256(hash_input).hexdigest()
            elif isinstance(model, (list, tuple)):
                hash_input = str(model).encode()
                return hashlib.sha256(hash_input).hexdigest()
            else:
                # For complex objects, use pickle serialization
                serialized = pickle.dumps(model)
                return hashlib.sha256(serialized).hexdigest()
        except Exception:
            return "unknown"
    
    def _serialize_pickle(self, model: Any) -> bytes:
        """Serialize using pickle"""
        return pickle.dumps(model)
    
    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize using pickle"""
        return pickle.loads(data)
    
    def _serialize_json(self, model: Any) -> bytes:
        """Serialize using JSON"""
        if isinstance(model, (int, float, str, list, tuple, dict)):
            return json.dumps(model).encode()
        else:
            raise ValueError("JSON serialization only supports basic Python types")
    
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize using JSON"""
        return json.loads(data.decode())
    
    def _serialize_numpy(self, model: Any) -> bytes:
        """Serialize numpy arrays"""
        if isinstance(model, np.ndarray):
            return model.tobytes()
        elif isinstance(model, dict):
            # Save as npz file in memory
            buffer = np.savez_compressed(None, **model)
            return buffer.getvalue()
        else:
            raise ValueError("NumPy serialization only supports arrays and dicts")
    
    def _deserialize_numpy(self, data: bytes) -> Any:
        """Deserialize numpy arrays"""
        try:
            # Try to load as npz
            from io import BytesIO
            buffer = BytesIO(data)
            loaded = np.load(buffer)
            
            if len(loaded.files) == 1:
                return loaded[loaded.files[0]]
            else:
                return {key: loaded[key] for key in loaded.files}
        except:
            # Try as single array
            return np.frombuffer(data, dtype=np.float32)
    
    def _serialize_torch(self, model: Any) -> bytes:
        """Serialize PyTorch model"""
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                # Save state dict
                return pickle.dumps(model.state_dict())
            elif isinstance(model, torch.jit.ScriptModule):
                # Save scripted model
                buffer = BytesIO()
                torch.save(model, buffer)
                return buffer.getvalue()
            else:
                # Save as tensor
                return pickle.dumps(model)
        except ImportError:
            raise ImportError("PyTorch not available")
    
    def _deserialize_torch(self, data: bytes) -> Any:
        """Deserialize PyTorch model"""
        try:
            import torch
            from io import BytesIO
            
            # Try to load as scripted model first
            try:
                buffer = BytesIO(data)
                return torch.load(buffer)
            except:
                # Try to load as state dict
                return pickle.loads(data)
        except ImportError:
            raise ImportError("PyTorch not available")
    
    def _serialize_tensorflow(self, model: Any) -> bytes:
        """Serialize TensorFlow model"""
        try:
            import tensorflow as tf
            from io import BytesIO
            
            buffer = BytesIO()
            model.save(buffer)
            return buffer.getvalue()
        except ImportError:
            raise ImportError("TensorFlow not available")
    
    def _deserialize_tensorflow(self, data: bytes) -> Any:
        """Deserialize TensorFlow model"""
        try:
            import tensorflow as tf
            from io import BytesIO
            
            buffer = BytesIO(data)
            return tf.keras.models.load_model(buffer)
        except ImportError:
            raise ImportError("TensorFlow not available")
    
    def get_serialization_info(self, file_path: str) -> SerializationMetadata:
        """Get serialization info without loading the model"""
        metadata_path = Path(str(file_path) + ".meta")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return SerializationMetadata(**{k: v for k, v in metadata_dict.items() 
                                       if k in SerializationMetadata.__annotations__})
    
    def verify_integrity(self, file_path: str) -> bool:
        """Verify model integrity"""
        try:
            metadata = self.get_serialization_info(file_path)
            model, _ = self.deserialize(file_path, verify_hash=True)
            return True
        except Exception as e:
            print(f"Integrity check failed: {e}")
            return False
    
    def copy_serialized_model(self, source_path: str, target_path: str) -> bool:
        """Copy serialized model with metadata"""
        try:
            source = Path(source_path)
            target = Path(target_path)
            
            # Copy model file
            if source.exists():
                target.write_bytes(source.read_bytes())
            
            # Copy metadata file
            source_meta = Path(str(source) + ".meta")
            if source_meta.exists():
                target_meta = Path(str(target) + ".meta")
                target_meta.write_text(source_meta.read_text())
            
            return True
        except Exception as e:
            print(f"Failed to copy model: {e}")
            return False
