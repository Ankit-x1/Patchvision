import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class PatchStrategy:
    """Base patch extraction strategy"""
    name: str
    extractor: Callable
    priority: int = 1
    
class PatchStrategies:
    """
    Collection of industrial patch strategies
    """
    
    @staticmethod
    def sliding_window(image: np.ndarray, 
                      size: int = 16,
                      stride: int = 8) -> np.ndarray:
        """Classic sliding window"""
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                patches.append(image[y:y+size, x:x+size])
                
        return np.array(patches)
    
    @staticmethod
    def interest_point_based(image: np.ndarray,
                           keypoints: np.ndarray,
                           size: int = 32) -> np.ndarray:
        """Extract patches around interest points"""
        patches = []
        h, w = image.shape[:2]
        
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            x1 = max(0, x - size//2)
            y1 = max(0, y - size//2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            
            patches.append(image[y1:y2, x1:x2])
            
        return np.array(patches)
    
    @staticmethod
    def random_sampling(image: np.ndarray,
                       num_patches: int = 100,
                       size_range: Tuple[int, int] = (8, 32)) -> np.ndarray:
        """Random patch sampling for efficiency"""
        h, w = image.shape[:2]
        patches = []
        
        for _ in range(num_patches):
            size = np.random.randint(size_range[0], size_range[1])
            x = np.random.randint(0, w - size)
            y = np.random.randint(0, h - size)
            patches.append(image[y:y+size, x:x+size])
            
        return np.array(patches)
    
    @staticmethod
    def edge_aware(image: np.ndarray,
                  edge_map: np.ndarray,
                  size: int = 16,
                  threshold: float = 0.1) -> np.ndarray:
        """Extract patches near edges"""
        h, w = image.shape[:2]
        patches = []
        
        # Find edge locations
        edge_locations = np.argwhere(edge_map > threshold)
        
        for y, x in edge_locations[:100]:  # Limit patches
            x1 = max(0, x - size//2)
            y1 = max(0, y - size//2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            
            patches.append(image[y1:y2, x1:x2])
            
        return np.array(patches)