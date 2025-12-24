import numpy as np
from typing import Optional, Tuple, Dict, List
import math

class SparseAttention:
    """
    Sparse attention mechanism for long sequences
    """
    
    def __init__(self,
                 block_size: int = 64,
                 num_rand_blocks: int = 3,
                 use_hash: bool = True):
        self.block_size = block_size
        self.num_rand_blocks = num_rand_blocks
        self.use_hash = use_hash
        
    def __call__(self,
                 Q: np.ndarray,
                 K: np.ndarray,
                 V: np.ndarray,
                 seq_len: int) -> np.ndarray:
        """
        Sparse attention computation
        """
        batch_size, _, dim = Q.shape
        
        # Divide into blocks
        num_blocks = seq_len // self.block_size
        
        # Local attention within blocks
        local_attention = self._local_attention(Q, K, V, num_blocks)
        
        # Random block attention
        random_attention = self._random_attention(Q, K, V, num_blocks)
        
        # Combine
        attention = local_attention + 0.5 * random_attention
        
        return attention
    
    def _local_attention(self,
                        Q: np.ndarray,
                        K: np.ndarray,
                        V: np.ndarray,
                        num_blocks: int) -> np.ndarray:
        """Attention within local blocks"""
        batch_size, seq_len, dim = Q.shape
        output = np.zeros_like(Q)
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            
            # Extract block
            Q_block = Q[:, start:end]
            K_block = K[:, start:end]
            V_block = V[:, start:end]
            
            # Compute block attention
            scores = np.matmul(Q_block, K_block.transpose(0, 2, 1))
            scores = scores / math.sqrt(dim)
            attention = np.softmax(scores, axis=-1)
            
            # Apply
            block_output = np.matmul(attention, V_block)
            output[:, start:end] = block_output
            
        return output
    
    def _random_attention(self,
                         Q: np.ndarray,
                         K: np.ndarray,
                         V: np.ndarray,
                         num_blocks: int) -> np.ndarray:
        """Attention to random blocks"""
        batch_size, seq_len, dim = Q.shape
        output = np.zeros_like(Q)
        
        for block_idx in range(num_blocks):
            # Select random blocks
            random_blocks = np.random.choice(
                num_blocks, 
                self.num_rand_blocks, 
                replace=False
            )
            
            Q_block = Q[:, block_idx*self.block_size:(block_idx+1)*self.block_size]
            
            # Gather keys and values from random blocks
            K_blocks = []
            V_blocks = []
            
            for rb in random_blocks:
                start = rb * self.block_size
                end = start + self.block_size
                K_blocks.append(K[:, start:end])
                V_blocks.append(V[:, start:end])
                
            K_random = np.concatenate(K_blocks, axis=1)
            V_random = np.concatenate(V_blocks, axis=1)
            
            # Compute attention
            scores = np.matmul(Q_block, K_random.transpose(0, 2, 1))
            scores = scores / math.sqrt(dim)
            attention = np.softmax(scores, axis=-1)
            
            # Apply
            block_output = np.matmul(attention, V_random)
            output[:, block_idx*self.block_size:(block_idx+1)*self.block_size] = block_output
            
        return output

class CrossScaleAttention:
    """
    Attention across different scales/resolutions
    """
    
    def __init__(self, scales: List[int] = [1, 2, 4, 8]):
        self.scales = scales
        
    def __call__(self,
                 multi_scale_features: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute attention across scales
        """
        # Align features to common resolution
        aligned = self._align_scales(multi_scale_features)
        
        # Concatenate along feature dimension
        features_concat = np.concatenate(list(aligned.values()), axis=-1)
        
        # Cross-scale attention
        batch_size, seq_len, total_dim = features_concat.shape
        Q = self._make_projection(features_concat, total_dim // 4)
        K = self._make_projection(features_concat, total_dim // 4)
        V = self._make_projection(features_concat, total_dim // 4)
        
        # Attention
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        scores = scores / math.sqrt(total_dim // 4)
        attention = np.softmax(scores, axis=-1)
        
        # Apply
        output = np.matmul(attention, V)
        
        return output
    
    def _align_scales(self, 
                     features: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Align features from different scales"""
        # Find target shape (from smallest scale)
        target_shape = None
        for scale, feat in features.items():
            if target_shape is None or feat.shape[1] < target_shape[1]:
                target_shape = feat.shape
                
        aligned = {}
        for scale, feat in features.items():
            if feat.shape == target_shape:
                aligned[scale] = feat
            else:
                # Simple resizing
                aligned[scale] = self._resize_sequence(feat, target_shape[1])
                
        return aligned
    
    @staticmethod
    def _resize_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
        """Resize sequence length"""
        current_len = seq.shape[1]
        
        if current_len == target_len:
            return seq
            
        batch_size, _, dim = seq.shape
        
        if current_len < target_len:
            # Upsample
            factor = target_len // current_len
            return np.repeat(seq, factor, axis=1)[:, :target_len]
        else:
            # Downsample
            factor = current_len // target_len
            return seq[:, ::factor, :][:, :target_len]
            
    @staticmethod
    def _make_projection(x: np.ndarray, dim: int) -> np.ndarray:
        """Simple linear projection"""
        weights = np.random.randn(x.shape[-1], dim) * 0.02
        return np.matmul(x, weights)