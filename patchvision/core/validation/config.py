"""
Configuration validation for PatchVision
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ValidationConfig:
    """Configuration for validation settings"""
    strict_mode: bool = False
    check_nan_inf: bool = True
    check_value_ranges: bool = True
    check_shapes: bool = True
    check_types: bool = True
    max_batch_size: int = 1024
    max_tensor_size: int = 1000000000  # 1B elements
    allowed_dtypes: List[str] = None
    
    def __post_init__(self):
        if self.allowed_dtypes is None:
            self.allowed_dtypes = ['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8']


class ConfigValidator:
    """
    Configuration validation and loading
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = ValidationConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    print(f"Warning: Unknown config key: {key}")
        
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self, config_file: str):
        """Save configuration to file"""
        try:
            config_data = asdict(self.config)
            
            # Create directory if needed
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.config.max_batch_size <= 0:
            issues.append("max_batch_size must be positive")
        
        if self.config.max_tensor_size <= 0:
            issues.append("max_tensor_size must be positive")
        
        if not self.config.allowed_dtypes:
            issues.append("allowed_dtypes cannot be empty")
        
        return issues
    
    def get_config(self) -> ValidationConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config key: {key}")


def load_default_config() -> ValidationConfig:
    """Load default validation configuration"""
    return ValidationConfig()


def create_config_file(config_file: str, **kwargs):
    """Create a configuration file with custom settings"""
    validator = ConfigValidator()
    validator.update_config(**kwargs)
    validator.save_config(config_file)
