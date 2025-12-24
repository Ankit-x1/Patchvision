"""
Model versioning system for PatchVision
"""

import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import shutil


@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    created_at: str
    description: str
    model_hash: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    metrics: Dict[str, float] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = {}
        if self.config is None:
            self.config = {}


class ModelVersionManager:
    """
    Advanced model versioning with branching and tagging
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.versions_file = self.model_dir / "versions.json"
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load version registry"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    versions = {}
                    for model_name, version_data in data.items():
                        versions[model_name] = {}
                        for version_str, version_info in version_data.items():
                            versions[model_name][version_str] = ModelVersion(**version_info)
                    return versions
            except Exception as e:
                print(f"Warning: Failed to load versions: {e}")
        return {}
    
    def _save_versions(self):
        """Save version registry"""
        try:
            data = {}
            for model_name, version_dict in self.versions.items():
                data[model_name] = {}
                for version_str, version_obj in version_dict.items():
                    data[model_name][version_str] = asdict(version_obj)
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save versions: {e}")
    
    def create_version(self,
                      model_name: str,
                      model_path: str,
                      description: str = "",
                      parent_version: Optional[str] = None,
                      tags: List[str] = None,
                      metrics: Dict[str, float] = None,
                      config: Dict[str, Any] = None) -> str:
        """
        Create a new version of a model
        
        Args:
            model_name: Name of the model
            model_path: Path to the model files
            description: Version description
            parent_version: Parent version for branching
            tags: Version tags
            metrics: Performance metrics
            config: Model configuration
            
        Returns:
            Version string
        """
        # Generate version number
        if model_name not in self.versions:
            self.versions[model_name] = {}
            version_str = "1.0.0"
        else:
            version_str = self._generate_version_number(model_name)
        
        # Compute model hash
        model_hash = self._compute_directory_hash(model_path)
        
        # Create version object
        version = ModelVersion(
            version=version_str,
            created_at=str(time.time()),
            description=description,
            model_hash=model_hash,
            parent_version=parent_version,
            tags=tags or [],
            metrics=metrics or {},
            config=config or {}
        )
        
        # Copy model files to version directory
        version_dir = self.model_dir / model_name / "versions" / version_str
        version_dir.mkdir(parents=True, exist_ok=True)
        
        if Path(model_path).is_dir():
            shutil.copytree(model_path, version_dir / "model", dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, version_dir / "model")
        
        # Save version metadata
        with open(version_dir / "version.json", 'w') as f:
            json.dump(asdict(version), f, indent=2)
        
        # Update registry
        self.versions[model_name][version_str] = version
        self._save_versions()
        
        return version_str
    
    def get_version(self, model_name: str, version: str) -> ModelVersion:
        """Get version metadata"""
        if model_name not in self.versions or version not in self.versions[model_name]:
            raise KeyError(f"Version {version} of model {model_name} not found")
        return self.versions[model_name][version]
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model"""
        if model_name not in self.versions:
            return []
        return sorted(self.versions[model_name].keys(), reverse=True)
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of a model"""
        versions = self.list_versions(model_name)
        return versions[0] if versions else None
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific version"""
        if model_name not in self.versions or version not in self.versions[model_name]:
            return False
        
        # Remove from registry
        del self.versions[model_name][version]
        
        # Remove files
        version_dir = self.model_dir / model_name / "versions" / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Clean up empty model directories
        if not self.versions[model_name]:
            del self.versions[model_name]
            model_dir = self.model_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
        
        self._save_versions()
        return True
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "hash_match": v1.model_hash == v2.model_hash,
            "metrics_diff": {},
            "config_diff": {},
            "time_diff": float(v2.created_at) - float(v1.created_at)
        }
        
        # Compare metrics
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            val1 = v1.metrics.get(metric, None)
            val2 = v2.metrics.get(metric, None)
            if val1 is not None and val2 is not None:
                comparison["metrics_diff"][metric] = val2 - val1
            else:
                comparison["metrics_diff"][metric] = {"v1": val1, "v2": val2}
        
        # Compare configs
        all_keys = set(v1.config.keys()) | set(v2.config.keys())
        for key in all_keys:
            val1 = v1.config.get(key)
            val2 = v2.config.get(key)
            if val1 != val2:
                comparison["config_diff"][key] = {"v1": val1, "v2": val2}
        
        return comparison
    
    def find_versions_by_tag(self, model_name: str, tag: str) -> List[str]:
        """Find versions with specific tag"""
        if model_name not in self.versions:
            return []
        
        matching_versions = []
        for version_str, version_obj in self.versions[model_name].items():
            if tag in version_obj.tags:
                matching_versions.append(version_str)
        
        return matching_versions
    
    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get complete version history"""
        if model_name not in self.versions:
            return []
        
        history = []
        versions = sorted(self.versions[model_name].items(), 
                         key=lambda x: float(x[1].created_at))
        
        for version_str, version_obj in versions:
            history.append({
                "version": version_str,
                "created_at": version_obj.created_at,
                "description": version_obj.description,
                "parent": version_obj.parent_version,
                "tags": version_obj.tags,
                "metrics": version_obj.metrics
            })
        
        return history
    
    def rollback_to_version(self, model_name: str, target_version: str) -> str:
        """Rollback to a previous version"""
        if model_name not in self.versions or target_version not in self.versions[model_name]:
            raise KeyError(f"Target version {target_version} not found")
        
        target_version_obj = self.versions[model_name][target_version]
        version_dir = self.model_dir / model_name / "versions" / target_version
        
        # Create new version based on target
        new_version = self.create_version(
            model_name=model_name,
            model_path=str(version_dir / "model"),
            description=f"Rollback to version {target_version}",
            parent_version=target_version,
            tags=["rollback"],
            metrics=target_version_obj.metrics,
            config=target_version_obj.config
        )
        
        return new_version
    
    def _generate_version_number(self, model_name: str) -> str:
        """Generate next version number using semantic versioning"""
        versions = list(self.versions[model_name].keys())
        if not versions:
            return "1.0.0"
        
        # Sort versions and get latest
        versions.sort(key=lambda x: [int(i) for i in x.split('.')])
        latest = versions[-1]
        
        # Increment patch version
        parts = latest.split('.')
        patch = int(parts[2]) + 1
        return f"{parts[0]}.{parts[1]}.{patch}"
    
    def _compute_directory_hash(self, path: str) -> str:
        """Compute hash of directory contents"""
        hash_md5 = hashlib.md5()
        path_obj = Path(path)
        
        if path_obj.is_file():
            with open(path_obj, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            for file_path in sorted(path_obj.rglob('*')):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(path_obj))
                    hash_md5.update(relative_path.encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
