"""
Configuration settings for PatchVision
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class PatchVisionConfig:
    """Main configuration for PatchVision"""
    
    # Model settings
    model_path: str = "models"
    model_name: str = "default"
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    use_fp16: bool = True
    use_quantization: bool = False
    
    # Processing settings
    patch_size: int = 16
    stride: int = 8
    num_heads: int = 8
    hidden_dim: int = 512
    
    # Performance settings
    num_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_profiling: bool = False
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_debug: bool = False
    
    # Validation settings
    strict_validation: bool = False
    check_inputs: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PatchVisionConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save(self, file_path: str):
        """Save configuration to file"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'PatchVisionConfig':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> PatchVisionConfig:
    """Get default configuration"""
    return PatchVisionConfig()


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    env_config = {}
    
    # Map environment variables to config keys
    env_mappings = {
        'PATCHVISION_DEVICE': 'device',
        'PATCHVISION_BATCH_SIZE': 'batch_size',
        'PATCHVISION_MODEL_PATH': 'model_path',
        'PATCHVISION_LOG_LEVEL': 'log_level',
        'PATCHVISION_NUM_WORKERS': 'num_workers',
        'PATCHVISION_MEMORY_LIMIT': 'memory_limit_gb',
        'PATCHVISION_USE_FP16': 'use_fp16',
        'PATCHVISION_USE_QUANTIZATION': 'use_quantization'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion
            if config_key in ['batch_size', 'num_workers']:
                env_config[config_key] = int(value)
            elif config_key in ['memory_limit_gb']:
                env_config[config_key] = float(value)
            elif config_key in ['use_fp16', 'use_quantization', 'strict_validation', 'check_inputs']:
                env_config[config_key] = value.lower() in ['true', '1', 'yes']
            else:
                env_config[config_key] = value
    
    return env_config


def load_config_with_env(config_file: Optional[str] = None) -> PatchVisionConfig:
    """Load configuration with environment variable overrides"""
    # Start with default config
    config = get_default_config()
    
    # Load from file if provided
    if config_file and os.path.exists(config_file):
        config = PatchVisionConfig.load(config_file)
    
    # Override with environment variables
    env_config = get_env_config()
    for key, value in env_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
