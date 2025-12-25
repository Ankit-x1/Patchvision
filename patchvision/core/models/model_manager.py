"""
Model loading and saving infrastructure
"""

import os
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
import warnings

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available, some features disabled")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    warnings.warn("TensorFlow not available, some features disabled")


class ModelManager:
    """
    Comprehensive model loading and saving infrastructure
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.metadata_file = self.model_dir / "model_registry.json"
        self.model_registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load model registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save model registry: {e}")
    
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   framework: str = "auto",
                   metadata: Optional[Dict[str, Any]] = None,
                   overwrite: bool = False) -> str:
        """
        Save model with metadata and versioning
        
        Args:
            model: Model object to save
            model_name: Name for the model
            framework: Framework type ('pytorch', 'tensorflow', 'numpy', 'auto')
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing model
            
        Returns:
            Model path
        """
        if framework == "auto":
            framework = self._detect_framework(model)
        
        model_path = self.model_dir / model_name
        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model {model_name} already exists")
        
        model_path.mkdir(exist_ok=True)
        
        # Save model based on framework
        if framework == "pytorch":
            self._save_pytorch_model(model, model_path)
        elif framework == "tensorflow":
            self._save_tensorflow_model(model, model_path)
        elif framework == "numpy":
            self._save_numpy_model(model, model_path)
        elif framework == "pickle":
            self._save_pickle_model(model, model_path)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Create metadata
        model_metadata = {
            "name": model_name,
            "framework": framework,
            "created_at": str(np.datetime64('now')),
            "model_hash": self._compute_model_hash(model),
            "metadata": metadata or {}
        }
        
        # Save metadata
        with open(model_path / "metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update registry
        self.model_registry[model_name] = model_metadata
        self._save_registry()
        
        return str(model_path)
    
    def load_model(self, 
                   model_name: str,
                   framework: Optional[str] = None,
                   device: Optional[str] = None) -> Any:
        """
        Load model from disk
        
        Args:
            model_name: Name of the model to load
            framework: Framework type (if None, auto-detect)
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        if framework is None:
            framework = metadata["framework"]
        
        # Load model based on framework
        if framework == "pytorch":
            return self._load_pytorch_model(model_path, device)
        elif framework == "tensorflow":
            return self._load_tensorflow_model(model_path)
        elif framework == "numpy":
            return self._load_numpy_model(model_path)
        elif framework == "pickle":
            return self._load_pickle_model(model_path)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.model_registry.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        if model_name not in self.model_registry:
            raise KeyError(f"Model {model_name} not found")
        return self.model_registry[model_name].copy()
    
    def delete_model(self, model_name: str) -> bool:
        """Delete model from disk"""
        model_path = self.model_dir / model_name
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            
            if model_name in self.model_registry:
                del self.model_registry[model_name]
                self._save_registry()
            return True
        return False
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect model framework"""
        if HAS_TORCH and isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            return "pytorch"
        elif HAS_TF and hasattr(model, 'save'):
            return "tensorflow"
        elif isinstance(model, (np.ndarray, dict)):
            return "numpy"
        else:
            return "pickle"
    
    def _compute_model_hash(self, model: Any) -> str:
        """Compute hash of model for integrity checking"""
        try:
            if HAS_TORCH and isinstance(model, nn.Module):
                # Hash model parameters
                state_dict = model.state_dict()
                hash_input = "".join([str(v) for v in state_dict.values()])
            elif isinstance(model, np.ndarray):
                hash_input = model.tobytes()
            elif isinstance(model, dict):
                hash_input = str(model).encode()
            else:
                hash_input = str(model).encode()
            
            return hashlib.sha256(hash_input).hexdigest()
        except Exception:
            return "unknown"
    
    def _save_pytorch_model(self, model: nn.Module, model_path: Path):
        """Save PyTorch model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        # Save model state dict
        torch.save(model.state_dict(), model_path / "model_state.pth")
        
        # Save full model if possible
        try:
            torch.save(model, model_path / "model_full.pth")
        except Exception as e:
            warnings.warn(f"Could not save full model: {e}")
    
    def _load_pytorch_model(self, model_path: Path, device: Optional[str] = None) -> nn.Module:
        """Load PyTorch model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")
        
        # Try to load full model first
        full_model_path = model_path / "model_full.pth"
        if full_model_path.exists():
            model = torch.load(full_model_path, map_location=device)
        else:
            # Load state dict only (requires model architecture)
            state_dict_path = model_path / "model_state.pth"
            if not state_dict_path.exists():
                raise FileNotFoundError("No PyTorch model files found")
            
            # This is a simplified version - in practice, you'd need the model architecture
            raise NotImplementedError("Loading from state_dict requires model architecture")
        
        return model
    
    def _save_tensorflow_model(self, model: Any, model_path: Path):
        """Save TensorFlow model"""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        model.save(str(model_path / "tf_model"))
    
    def _load_tensorflow_model(self, model_path: Path) -> Any:
        """Load TensorFlow model"""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        tf_model_path = model_path / "tf_model"
        if not tf_model_path.exists():
            raise FileNotFoundError("TensorFlow model not found")
        
        return tf.keras.models.load_model(str(tf_model_path))
    
    def _save_numpy_model(self, model: Union[np.ndarray, Dict], model_path: Path):
        """Save numpy-based model"""
        if isinstance(model, np.ndarray):
            np.save(model_path / "model.npy", model)
        elif isinstance(model, dict):
            np.savez_compressed(model_path / "model.npz", **model)
        else:
            raise ValueError("Unsupported numpy model type")
    
    def _load_numpy_model(self, model_path: Path) -> Union[np.ndarray, Dict]:
        """Load numpy-based model"""
        array_path = model_path / "model.npy"
        dict_path = model_path / "model.npz"
        
        if array_path.exists():
            return np.load(array_path)
        elif dict_path.exists():
            data = np.load(dict_path)
            return {key: data[key] for key in data.files}
        else:
            raise FileNotFoundError("NumPy model not found")
    
    def _save_pickle_model(self, model: Any, model_path: Path):
        """Save model using pickle"""
        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
    
    def _load_pickle_model(self, model_path: Path) -> Any:
        """Load pickled model"""
        with open(model_path / "model.pkl", 'rb') as f:
            return pickle.load(f)
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify model integrity using hash"""
        try:
            metadata = self.get_model_info(model_name)
            stored_hash = metadata.get("model_hash")
            
            if stored_hash == "unknown":
                return True  # Can't verify
            
            model = self.load_model(model_name)
            current_hash = self._compute_model_hash(model)
            
            return stored_hash == current_hash
        except Exception:
            return False
