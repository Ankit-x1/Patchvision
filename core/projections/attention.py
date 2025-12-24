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
        """Vectorized attention within local blocks"""
        batch_size, seq_len, dim = Q.shape
        output = np.zeros_like(Q)
        
        # Vectorized block processing
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            
            # Extract blocks using slicing
            Q_block = Q[:, start:end]
            K_block = K[:, start:end]
            V_block = V[:, start:end]
            
            # Vectorized attention computation
            scores = np.einsum('bqd,bkd->bqk', Q_block, K_block) / math.sqrt(dim)
            attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention = attention / np.sum(attention, axis=-1, keepdims=True)
            
            # Vectorized application
            block_output = np.einsum('bqk,bkd->bqd', attention, V_block)
            output[:, start:end] = block_output
            
        return output
    
    def _random_attention(self,
                         Q: np.ndarray,
                         K: np.ndarray,
                         V: np.ndarray,
                         num_blocks: int) -> np.ndarray:
        """Vectorized attention to random blocks"""
        batch_size, seq_len, dim = Q.shape
        output = np.zeros_like(Q)
        
        # Pre-generate all random block selections for vectorization
        all_random_blocks = np.array([
            np.random.choice(num_blocks, self.num_rand_blocks, replace=False)
            for _ in range(num_blocks)
        ])
        
        for block_idx in range(num_blocks):
            Q_block = Q[:, block_idx*self.block_size:(block_idx+1)*self.block_size]
            
            # Vectorized gathering of random blocks
            random_blocks = all_random_blocks[block_idx]
            
            # Collect all K and V blocks at once
            K_blocks_list = []
            V_blocks_list = []
            
            for rb in random_blocks:
                start = rb * self.block_size
                end = start + self.block_size
                K_blocks_list.append(K[:, start:end])
                V_blocks_list.append(V[:, start:end])
                
            # Vectorized concatenation
            K_random = np.concatenate(K_blocks_list, axis=1)
            V_random = np.concatenate(V_blocks_list, axis=1)
            
            # Vectorized attention computation using einsum
            scores = np.einsum('bqd,bkd->bqk', Q_block, K_random) / math.sqrt(dim)
            
            # Numerically stable softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            # Vectorized application
            block_output = np.einsum('bqk,bkd->bqd', attention, V_random)
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
            
        batch_size, current_len, dim = seq.shape
        
        if current_len < target_len:
            # Upsample with proper handling of odd dimensions
            factor = target_len / current_len
            if factor.is_integer():
                return np.repeat(seq, int(factor), axis=1)[:, :target_len]
            else:
                # Use interpolation for non-integer factors
                indices = np.linspace(0, current_len-1, target_len)
                return np.take(seq, indices.astype(int), axis=1)
        else:
            # Downsample with proper handling of odd dimensions
            factor = current_len / target_len
            if factor.is_integer():
                return seq[:, ::int(factor), :][:, :target_len]
            else:
                # Use strided slicing for non-integer factors
                indices = np.round(np.linspace(0, current_len-1, target_len)).astype(int)
                return seq[np.arange(batch_size)[:, None], indices]
            
    @staticmethod
    def _make_projection(x: np.ndarray, dim: int) -> np.ndarray:
        """Simple linear projection"""
        weights = np.random.randn(x.shape[-1], dim) * 0.02
        return np.matmul(x, weights)