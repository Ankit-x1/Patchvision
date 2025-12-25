"""
Performance profiling hooks for PatchVision
"""

import time
import functools
import threading
from typing import Dict, Any, Callable, Optional, List
from collections import defaultdict
import numpy as np


class ProfilerHook:
    """
    Simple performance profiler for function calls
    """
    
    def __init__(self):
        self.call_times = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.lock = threading.Lock()
        self.enabled = True
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function calls"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    with self.lock:
                        self.call_times[name].append(duration)
                        self.call_counts[name] += 1
            
            return wrapper
        return decorator
    
    def profile_method(self, cls_name: str = None):
        """Decorator to profile class methods"""
        def decorator(func):
            name = func_name or f"{cls_name}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    with self.lock:
                        self.call_times[name].append(duration)
                        self.call_counts[name] += 1
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics"""
        stats = {}
        
        with self.lock:
            for func_name, times in self.call_times.items():
                if times:
                    stats[func_name] = {
                        'avg_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'std_time': np.std(times),
                        'total_time': np.sum(times),
                        'call_count': self.call_counts[func_name]
                    }
        
        return stats
    
    def reset(self):
        """Reset profiling data"""
        with self.lock:
            self.call_times.clear()
            self.call_counts.clear()
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
    
    def print_summary(self):
        """Print profiling summary"""
        stats = self.get_stats()
        
        if not stats:
            print("No profiling data available")
            return
        
        print("\n=== Profiling Summary ===")
        for func_name, data in sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
            print(f"{func_name}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Avg time: {data['avg_time']*1000:.2f} ms")
            print(f"  Total time: {data['total_time']:.2f} s")
            print()


# Global profiler instance
_global_profiler = ProfilerHook()


def profile_function(name: Optional[str] = None):
    """Global function profiling decorator"""
    return _global_profiler.profile_function(name)


def profile_method(cls_name: Optional[str] = None):
    """Global method profiling decorator"""
    return _global_profiler.profile_method(cls_name)


def get_profiling_stats() -> Dict[str, Dict[str, float]]:
    """Get global profiling statistics"""
    return _global_profiler.get_stats()


def reset_profiling():
    """Reset global profiling data"""
    _global_profiler.reset()


def enable_profiling():
    """Enable global profiling"""
    _global_profiler.enable()


def disable_profiling():
    """Disable global profiling"""
    _global_profiler.disable()


def print_profiling_summary():
    """Print global profiling summary"""
    _global_profiler.print_summary()


class MemoryProfiler:
    """
    Simple memory usage profiler
    """
    
    def __init__(self):
        self.memory_snapshots = []
        self.enabled = True
    
    def snapshot(self, label: str = ""):
        """Take memory snapshot"""
        if not self.enabled:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'label': label,
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'timestamp': time.time()
            }
            
            self.memory_snapshots.append(snapshot)
            
        except ImportError:
            pass  # psutil not available
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024)
            }
        except ImportError:
            return {}
    
    def get_memory_trend(self) -> List[Dict[str, float]]:
        """Get memory usage trend"""
        return self.memory_snapshots.copy()


# Global memory profiler
_global_memory_profiler = MemoryProfiler()


def memory_snapshot(label: str = ""):
    """Take global memory snapshot"""
    _global_memory_profiler.snapshot(label)


def get_memory_usage() -> Dict[str, float]:
    """Get global memory usage"""
    return _global_memory_profiler.get_memory_usage()


def get_memory_trend() -> List[Dict[str, float]]:
    """Get global memory trend"""
    return _global_memory_profiler.get_memory_trend()
