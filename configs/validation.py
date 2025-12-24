#!/usr/bin/env python3
"""
Configuration validation for PatchVision
"""

import yaml
import jsonschema
from pathlib import Path

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "description": {"type": "string"},
        "core": {
            "type": "object",
            "properties": {
                "patches": {
                    "type": "object",
                    "properties": {
                        "default_size": {"type": "integer", "minimum": 4, "maximum": 128},
                        "default_stride": {"type": "integer", "minimum": 2, "maximum": 64},
                        "strategies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True
                        }
                    },
                    "required": ["default_size", "default_stride"]
                }
            },
            "required": ["patches"]
        }
    },
    "required": ["version", "core"]
}

def validate_config(config_path: str):
    """Validate configuration file against schema"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        print(f"✓ Configuration {config_path} is valid")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def validate_all_configs():
    """Validate all configuration files"""
    config_dir = Path("configs")
    
    if not config_dir.exists():
        print("Config directory not found")
        return False
    
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    
    if not config_files:
        print("No configuration files found")
        return False
    
    results = []
    for config_file in config_files:
        results.append(validate_config(config_file))
    
    return all(results)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Validate specific config
        config_file = sys.argv[1]
        success = validate_config(config_file)
        sys.exit(0 if success else 1)
    else:
        # Validate all configs
        success = validate_all_configs()
        sys.exit(0 if success else 1)