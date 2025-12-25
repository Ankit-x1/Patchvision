"""
Data integrity checks and validation for PatchVision
"""

import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import time


@dataclass
class IntegrityCheck:
    """Single integrity check result"""
    check_name: str
    passed: bool
    details: Dict[str, Any]
    timestamp: float


@dataclass
class DataIntegrityReport:
    """Complete data integrity report"""
    file_path: str
    file_hash: str
    checks: List[IntegrityCheck]
    overall_status: str  # 'valid', 'corrupted', 'suspicious'
    metadata: Dict[str, Any]


class DataIntegrityValidator:
    """
    Comprehensive data integrity validation system
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
        
    def validate_file_integrity(self, file_path: Union[str, Path]) -> DataIntegrityReport:
        """Validate complete file integrity"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return DataIntegrityReport(
                file_path=str(file_path),
                file_hash="",
                checks=[],
                overall_status="corrupted",
                metadata={"error": "File does not exist"}
            )
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Run integrity checks
        checks = []
        
        # Basic file checks
        checks.append(self._check_file_accessibility(file_path))
        checks.append(self._check_file_size(file_path))
        checks.append(self._check_file_extension(file_path))
        
        # Content-based checks based on file type
        if file_path.suffix.lower() in ['.npy']:
            checks.extend(self._check_numpy_file(file_path))
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            checks.extend(self._check_image_file(file_path))
        elif file_path.suffix.lower() in ['.json']:
            checks.extend(self._check_json_file(file_path))
        
        # Determine overall status
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            if any("corrupted" in c.check_name for c in failed_checks):
                overall_status = "corrupted"
            else:
                overall_status = "suspicious"
        else:
            overall_status = "valid"
        
        return DataIntegrityReport(
            file_path=str(file_path),
            file_hash=file_hash,
            checks=checks,
            overall_status=overall_status,
            metadata={
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "last_modified": file_path.stat().st_mtime,
                "strict_mode": self.strict_mode
            }
        )
    
    def validate_array_integrity(self, array: np.ndarray, context: Optional[Dict] = None) -> List[IntegrityCheck]:
        """Validate numpy array integrity"""
        checks = []
        
        # Check for NaN values
        nan_check = IntegrityCheck(
            check_name="nan_check",
            passed=not np.any(np.isnan(array)),
            details={
                "nan_count": int(np.sum(np.isnan(array))),
                "total_elements": array.size,
                "shape": array.shape
            },
            timestamp=time.time()
        )
        checks.append(nan_check)
        
        # Check for infinite values
        inf_check = IntegrityCheck(
            check_name="inf_check",
            passed=not np.any(np.isinf(array)),
            details={
                "inf_count": int(np.sum(np.isinf(array))),
                "total_elements": array.size,
                "shape": array.shape
            },
            timestamp=time.time()
        )
        checks.append(inf_check)
        
        # Check for valid numeric ranges
        range_check = IntegrityCheck(
            check_name="range_check",
            passed=self._check_array_ranges(array),
            details={
                "min_value": float(np.min(array)),
                "max_value": float(np.max(array)),
                "mean_value": float(np.mean(array)),
                "std_value": float(np.std(array)),
                "dtype": str(array.dtype)
            },
            timestamp=time.time()
        )
        checks.append(range_check)
        
        # Check array consistency
        consistency_check = IntegrityCheck(
            check_name="consistency_check",
            passed=self._check_array_consistency(array),
            details={
                "is_contiguous": array.flags['C_CONTIGUOUS'],
                "is_writable": array.flags['WRITEABLE'],
                "memory_layout": str(array.flags)
            },
            timestamp=time.time()
        )
        checks.append(consistency_check)
        
        # Check for suspicious patterns
        pattern_check = IntegrityCheck(
            check_name="pattern_check",
            passed=self._check_suspicious_patterns(array),
            details={
                "has_constant_values": self._has_constant_values(array),
                "has_repeating_patterns": self._has_repeating_patterns(array),
                "entropy_estimate": self._estimate_array_entropy(array)
            },
            timestamp=time.time()
        )
        checks.append(pattern_check)
        
        return checks
    
    def validate_batch_integrity(self, batch_data: List[np.ndarray], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate batch data integrity"""
        batch_checks = []
        batch_stats = {
            "batch_size": len(batch_data),
            "total_elements": 0,
            "total_memory_mb": 0,
            "consistent_shapes": True,
            "consistent_dtypes": True,
            "shape_distribution": {},
            "dtype_distribution": {}
        }
        
        if not batch_data:
            return {
                "batch_valid": False,
                "error": "Empty batch",
                "checks": [],
                "stats": batch_stats
            }
        
        # Get reference shape and dtype
        ref_shape = batch_data[0].shape
        ref_dtype = batch_data[0].dtype
        
        for i, array in enumerate(batch_data):
            # Individual array checks
            array_checks = self.validate_array_integrity(array, {"batch_index": i})
            batch_checks.extend(array_checks)
            
            # Update batch statistics
            batch_stats["total_elements"] += array.size
            batch_stats["total_memory_mb"] += array.nbytes / (1024 * 1024)
            
            # Check shape consistency
            if array.shape != ref_shape:
                batch_stats["consistent_shapes"] = False
                shape_key = str(array.shape)
                batch_stats["shape_distribution"][shape_key] = batch_stats["shape_distribution"].get(shape_key, 0) + 1
            
            # Check dtype consistency
            if array.dtype != ref_dtype:
                batch_stats["consistent_dtypes"] = False
                dtype_key = str(array.dtype)
                batch_stats["dtype_distribution"][dtype_key] = batch_stats["dtype_distribution"].get(dtype_key, 0) + 1
        
        # Overall batch validation
        batch_valid = (
            batch_stats["consistent_shapes"] and
            batch_stats["consistent_dtypes"] and
            all(check.passed for check in batch_checks)
        )
        
        return {
            "batch_valid": batch_valid,
            "checks": batch_checks,
            "stats": batch_stats,
            "reference_shape": ref_shape,
            "reference_dtype": str(ref_dtype)
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _check_file_accessibility(self, file_path: Path) -> IntegrityCheck:
        """Check if file is accessible"""
        try:
            # Test read access
            with open(file_path, 'rb') as f:
                f.read(1)  # Read first byte
            return IntegrityCheck(
                check_name="file_accessibility",
                passed=True,
                details={"readable": True, "writable": file_path.stat().st_mode & 0o200 != 0}
            )
        except Exception as e:
            return IntegrityCheck(
                check_name="file_accessibility",
                passed=False,
                details={"error": str(e), "readable": False, "writable": False}
            )
    
    def _check_file_size(self, file_path: Path) -> IntegrityCheck:
        """Check file size"""
        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Check if file is suspiciously small or large
            suspicious = size_mb < 0.001 or size_mb > 1000  # < 1KB or > 1GB
            
            return IntegrityCheck(
                check_name="file_size",
                passed=not suspicious,
                details={
                    "size_bytes": size_bytes,
                    "size_mb": size_mb,
                    "suspicious": suspicious
                }
            )
        except Exception as e:
            return IntegrityCheck(
                check_name="file_size",
                passed=False,
                details={"error": str(e)}
            )
    
    def _check_file_extension(self, file_path: Path) -> IntegrityCheck:
        """Check file extension"""
        allowed_extensions = {'.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.json', '.yaml', '.yml'}
        extension = file_path.suffix.lower()
        
        return IntegrityCheck(
            check_name="file_extension",
            passed=extension in allowed_extensions,
            details={
                "extension": extension,
                "allowed": extension in allowed_extensions
            }
        )
    
    def _check_numpy_file(self, file_path: Path) -> List[IntegrityCheck]:
        """Check numpy file integrity"""
        checks = []
        
        try:
            # Try to load numpy file
            array = np.load(file_path, allow_pickle=False)
            
            # Validate loaded array
            array_checks = self.validate_array_integrity(array, {"file": str(file_path)})
            checks.extend(array_checks)
            
            # Check numpy-specific integrity
            version_check = IntegrityCheck(
                check_name="numpy_version",
                passed=True,  # Would need version comparison
                details={
                    "numpy_version": np.__version__,
                    "file_format": "numpy"
                }
            )
            checks.append(version_check)
            
        except Exception as e:
            checks.append(IntegrityCheck(
                check_name="numpy_load",
                passed=False,
                details={"error": str(e)}
            ))
        
        return checks
    
    def _check_image_file(self, file_path: Path) -> List[IntegrityCheck]:
        """Check image file integrity"""
        checks = []
        
        try:
            # Try to read image header
            with open(file_path, 'rb') as f:
                header = f.read(100)  # Read first 100 bytes
            
            # Basic image format checks
            extension = file_path.suffix.lower()
            
            if extension in ['.jpg', '.jpeg']:
                valid = header.startswith(b'\xff\xd8\xff')
                format_name = "JPEG"
            elif extension == '.png':
                valid = header.startswith(b'\x89PNG\r\n\x1a\n')
                format_name = "PNG"
            elif extension == '.bmp':
                valid = header.startswith(b'BM')
                format_name = "BMP"
            else:
                valid = True  # Assume valid for other formats
                format_name = extension[1:].upper()
            
            checks.append(IntegrityCheck(
                check_name="image_header",
                passed=valid,
                details={
                    "format": format_name,
                    "header_valid": valid,
                    "extension_matches": extension in ['.jpg', '.jpeg', '.png', '.bmp']
                }
            ))
            
        except Exception as e:
            checks.append(IntegrityCheck(
                check_name="image_header",
                passed=False,
                details={"error": str(e)}
            ))
        
        return checks
    
    def _check_json_file(self, file_path: Path) -> List[IntegrityCheck]:
        """Check JSON file integrity"""
        checks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse JSON
            data = json.loads(content)
            
            checks.append(IntegrityCheck(
                check_name="json_syntax",
                passed=True,
                details={
                    "valid_json": True,
                    "root_type": type(data).__name__
                }
            ))
            
            # Check for common JSON issues
            if isinstance(data, dict):
                checks.append(IntegrityCheck(
                    check_name="json_structure",
                    passed=True,
                    details={
                        "has_keys": len(data.keys()) > 0,
                        "key_count": len(data.keys())
                    }
                ))
            
        except json.JSONDecodeError as e:
            checks.append(IntegrityCheck(
                check_name="json_syntax",
                passed=False,
                details={"error": str(e), "line": getattr(e, 'lineno', None)}
            ))
        except Exception as e:
            checks.append(IntegrityCheck(
                check_name="json_access",
                passed=False,
                details={"error": str(e)}
            ))
        
        return checks
    
    def _check_array_ranges(self, array: np.ndarray) -> bool:
        """Check if array values are in valid ranges"""
        if array.size == 0:
            return True
        
        # Check based on dtype
        if array.dtype == np.uint8:
            return np.min(array) >= 0 and np.max(array) <= 255
        elif array.dtype == np.float32 or array.dtype == np.float16:
            return not np.any(np.abs(array) > 1e6)  # Reasonable float range
        elif array.dtype in [np.int32, np.int64]:
            return not np.any(np.abs(array) > 1e9)  # Reasonable int range
        
        return True  # Assume valid for other types
    
    def _check_array_consistency(self, array: np.ndarray) -> bool:
        """Check array memory consistency"""
        # Check for memory alignment issues
        if not array.flags['C_CONTIGUOUS'] and not array.flags['F_CONTIGUOUS']:
            return False
        
        # Check for unusual strides
        if array.ndim > 1:
            expected_stride = array.dtype.itemsize
            for i, stride in enumerate(array.strides[1:]):
                if stride % expected_stride != 0 and stride != 0:
                    return False
        
        return True
    
    def _check_suspicious_patterns(self, array: np.ndarray) -> bool:
        """Check for suspicious patterns in array"""
        if array.size < 10:  # Too small to detect patterns
            return True
        
        # Check for all zeros or all same values
        if np.all(array == 0) or np.all(array == array.flat[0]):
            return False
        
        # Check for suspicious regularity
        unique_ratio = len(np.unique(array)) / array.size
        if unique_ratio < 0.01:  # Less than 1% unique values
            return False
        
        return True
    
    def _has_constant_values(self, array: np.ndarray) -> bool:
        """Check if array has constant values"""
        unique_values = np.unique(array)
        return len(unique_values) == 1
    
    def _has_repeating_patterns(self, array: np.ndarray) -> bool:
        """Check for repeating patterns"""
        if array.size < 20:
            return False
        
        # Simple pattern detection for 1D arrays
        if array.ndim == 1:
            first_half = array[:len(array)//2]
            second_half = array[len(array)//2:]
            return np.array_equal(first_half, second_half)
        
        return False
    
    def _estimate_array_entropy(self, array: np.ndarray) -> float:
        """Estimate entropy of array values"""
        if array.size == 0:
            return 0.0
        
        # Discretize values for entropy calculation
        flat = array.flatten()
        if flat.dtype in [np.float32, np.float16]:
            # Convert to discrete values
            discretized = np.floor(flat * 255).astype(np.uint8)
        else:
            discretized = flat
        
        # Calculate histogram
        hist, _ = np.histogram(discretized, bins=256, range=(0, 256))
        hist = hist[hist > 0] / (hist.sum() + 1e-8)
        
        # Calculate entropy
        if hist.size > 0:
            return -np.sum(hist * np.log2(hist + 1e-8))
        
        return 0.0


def create_integrity_validator(strict_mode: bool = False) -> DataIntegrityValidator:
    """Create configured integrity validator"""
    return DataIntegrityValidator(strict_mode)


# Decorator for automatic integrity checking
def validate_integrity(validator: DataIntegrityValidator, check_type: str = "array"):
    """Decorator for automatic integrity validation"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate result based on type
            if check_type == "array" and isinstance(result, np.ndarray):
                checks = validator.validate_array_integrity(result)
                failed_checks = [c for c in checks if not c.passed]
                if failed_checks and validator.strict_mode:
                    raise ValueError(f"Integrity check failed: {[c.check_name for c in failed_checks]}")
            
            elif check_type == "batch" and isinstance(result, list):
                batch_result = validator.validate_batch_integrity(result)
                if not batch_result["batch_valid"] and validator.strict_mode:
                    raise ValueError(f"Batch integrity check failed: {batch_result['error']}")
            
            return result
        return wrapper
    return decorator
