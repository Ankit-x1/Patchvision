"""
Configuration manager for PatchVision
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .settings import PatchVisionConfig, load_config_with_env


class ConfigManager:
    """
    Central configuration management for PatchVision
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config = load_config_with_env(self.config_file)
        self._watchers = []
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations"""
        possible_locations = [
            "patchvision.json",
            "config.json",
            "configs/patchvision.json",
            os.path.expanduser("~/.patchvision/config.json"),
            "/etc/patchvision/config.json"
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                return location
        
        return None
    
    def get_config(self) -> PatchVisionConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Notify watchers
        for watcher in self._watchers:
            watcher(self.config)
    
    def save_config(self, file_path: Optional[str] = None):
        """Save configuration to file"""
        save_path = file_path or self.config_file or "patchvision.json"
        self.config.save(save_path)
        
        if file_path:
            self.config_file = file_path
    
    def reload_config(self):
        """Reload configuration from file"""
        if self.config_file:
            self.config = load_config_with_env(self.config_file)
            
            # Notify watchers
            for watcher in self._watchers:
                watcher(self.config)
    
    def add_watcher(self, callback):
        """Add configuration change watcher"""
        self._watchers.append(callback)
    
    def remove_watcher(self, callback):
        """Remove configuration change watcher"""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        config_dict = self.config.to_dict()
        
        # Simple section mapping
        section_mappings = {
            'model': ['model_path', 'model_name', 'batch_size', 'max_sequence_length'],
            'hardware': ['device', 'use_fp16', 'use_quantization'],
            'processing': ['patch_size', 'stride', 'num_heads', 'hidden_dim'],
            'performance': ['num_workers', 'memory_limit_gb', 'enable_profiling'],
            'logging': ['log_level', 'log_file', 'enable_debug'],
            'validation': ['strict_validation', 'check_inputs']
        }
        
        if section in section_mappings:
            return {key: config_dict[key] for key in section_mappings[section] if key in config_dict}
        else:
            return {}
    
    def validate_config(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate batch size
        if self.config.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        # Validate memory limit
        if self.config.memory_limit_gb <= 0:
            issues.append("memory_limit_gb must be positive")
        
        # Validate device
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if self.config.device not in valid_devices:
            issues.append(f"device must be one of {valid_devices}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.config.log_level not in valid_log_levels:
            issues.append(f"log_level must be one of {valid_log_levels}")
        
        # Validate patch size
        if self.config.patch_size <= 0:
            issues.append("patch_size must be positive")
        
        # Validate stride
        if self.config.stride <= 0:
            issues.append("stride must be positive")
        
        if self.config.stride > self.config.patch_size:
            issues.append("stride should not be larger than patch_size")
        
        return issues


# Global config manager instance
_global_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def load_config(config_file: Optional[str] = None) -> PatchVisionConfig:
    """Load configuration"""
    manager = get_config_manager(config_file)
    return manager.get_config()


def save_config(config: PatchVisionConfig, file_path: str):
    """Save configuration"""
    config.save(file_path)


def get_config() -> PatchVisionConfig:
    """Get current configuration"""
    manager = get_config_manager()
    return manager.get_config()


def update_config(**kwargs):
    """Update global configuration"""
    manager = get_config_manager()
    manager.update_config(**kwargs)
