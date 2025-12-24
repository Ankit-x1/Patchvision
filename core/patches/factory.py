import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
class PatchFactory:
    """
    Industrial-grade patch processor with adaptive strategies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'patch_size': 16,
            'stride': 8,
            'mode': 'adaptive',
            'min_contrast': 0.1,
            'max_overlap': 0.3
        }
        
    def adaptive_patching(self, 
                         image: np.ndarray,
                         saliency_map: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Generate patches based on content importance
        """
        h, w = image.shape[:2]
        patch_size = self.config['patch_size']
        stride = self.config['stride']
        
        patches = []
        patch_id = 0
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Compute patch importance
                importance = 1.0
                if saliency_map is not None:
                    saliency_patch = saliency_map[y:y+patch_size, x:x+patch_size]
                    importance = float(saliency_patch.mean())
                
                # Dynamic resolution based on importance
                resolution = 'high' if importance > 0.5 else 'standard'
                
                patches.append({
                    'id': patch_id,
                    'coordinates': (x, y, patch_size, patch_size),
                    'data': patch,
                    'importance': importance,
                    'resolution': resolution,
                    'metadata': {
                        'contrast': float(np.std(patch)),
                        'entropy': self._compute_entropy(patch)
                    }
                })
                patch_id += 1
                
        return patches
    
    def hierarchical_patching(self,
                            image: np.ndarray,
                            levels: int = 3) -> Dict[int, List[Dict]]:
        """
        Multi-scale hierarchical patching
        """
        pyramid = {}
        
        for level in range(levels):
            scale = 2 ** level
            patch_size = self.config['patch_size'] // scale
            
            # Downsample
            if scale > 1:
                h, w = image.shape[:2]
                resized = cv2.resize(image, (w//scale, h//scale))
            else:
                resized = image
                
            patches = self.adaptive_patching(resized)
            pyramid[level] = patches
            
        return pyramid
    
    def temporal_patching(self,
                         video: np.ndarray,
                         temporal_stride: int = 2) -> List[Dict]:
        """
        3D spatio-temporal patches for video
        """
        t, h, w, c = video.shape
        patches_3d = []
        
        for frame_start in range(0, t - 3, temporal_stride):
            temporal_patch = video[frame_start:frame_start+3]
            
            # Extract spatial patches from temporal volume
            for y in range(0, h - 16, 8):
                for x in range(0, w - 16, 8):
                    patch = temporal_patch[:, y:y+16, x:x+16, :]
                    
                    # Compute motion information
                    motion = np.std(patch, axis=0).mean()
                    
                    patches_3d.append({
                        'spatial': (x, y, 16, 16),
                        'temporal': (frame_start, 3),
                        'data': patch,
                        'motion': motion
                    })
                    
        return patches_3d
    
    @staticmethod
    def _compute_entropy(patch: np.ndarray) -> float:
        """Compute patch information entropy for both grayscale and color images"""
        if len(patch.shape) == 3:
            # Color image - compute entropy for each channel and average
            entropies = []
            for channel in range(patch.shape[2]):
                hist = cv2.calcHist([patch], [channel], None, [256], [0, 256])
                hist = hist[hist > 0] / hist.sum()
                if hist.size > 0:
                    entropy = -np.sum(hist * np.log2(hist))
                    entropies.append(entropy)
            return np.mean(entropies) if entropies else 0.0
        else:
            # Grayscale image
            hist = cv2.calcHist([patch], [0], None, [256], [0, 256])
            hist = hist[hist > 0] / hist.sum()
            return -np.sum(hist * np.log2(hist)) if hist.size > 0 else 0.0