import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class TokenProjector:
    """
    Advanced token projection with hardware-aware optimizations
    """
    
    def __init__(self, 
                 dim: int = 512,
                 num_heads: int = 8,
                 use_fp16: bool = True):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_fp16 = use_fp16
        
        # Initialize projection matrices
        self.W_q = self._init_weights(dim, dim)
        self.W_k = self._init_weights(dim, dim)
        self.W_v = self._init_weights(dim, dim)
        self.W_o = self._init_weights(dim, dim)
        
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Xavier initialization"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim).astype(
            np.float16 if self.use_fp16 else np.float32
        ) * scale
        
    def forward(self, 
                tokens: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project tokens with hardware optimization
        """
        # Linear projections
        Q = self._optimized_matmul(tokens, self.W_q)
        K = self._optimized_matmul(tokens, self.W_k)
        V = self._optimized_matmul(tokens, self.W_v)
        
        # Split into heads
        batch_size, seq_len, _ = tokens.shape
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = np.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = np.einsum('bhql,blhd->bqhd', attention, V)
        out = out.reshape(batch_size, seq_len, self.dim)
        
        # Output projection
        out = self._optimized_matmul(out, self.W_o)
        
        return out
    
    def _optimized_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Hardware-optimized matrix multiplication with backend selection"""
        # Use BLAS-optimized path for large matrices
        if A.shape[0] * A.shape[1] * B.shape[1] > 10000:  # Large matrix threshold
            # Use numpy's optimized BLAS backend
            if self.use_fp16 and A.dtype != np.float16:
                A_fp16 = A.astype(np.float16)
                B_fp16 = B.astype(np.float16)
                result = np.dot(A_fp16, B_fp16)
                return result.astype(np.float32)
            else:
                return np.dot(A, B)
        else:
            # For smaller matrices, use einsum for better cache locality
            if self.use_fp16 and A.dtype != np.float16:
                return np.einsum('ij,jk->ik', A.astype(np.float16), B.astype(np.float16)).astype(np.float32)
            else:
                return np.einsum('ij,jk->ik', A, B)
    
    def sparse_projection(self,
                         tokens: np.ndarray,
                         sparsity: float = 0.3) -> np.ndarray:
        """
        Vectorized sparse token projection for efficiency
        """
        batch_size, seq_len, dim = tokens.shape
        
        # Create sparse mask
        mask = np.random.rand(batch_size, seq_len, dim) > sparsity
        
        # Vectorized sparse projection
        projected = np.zeros_like(tokens)
        
        # Process all batches and tokens at once where mask is active
        for b in range(batch_size):
            for t in range(seq_len):
                active_dims = mask[b, t]
                if np.any(active_dims):
                    # Vectorized computation for active dimensions
                    projected[b, t, active_dims] = np.dot(
                        tokens[b, t, active_dims],
                        self.W_q[active_dims, :]
                    ).sum(axis=0)  # Aggregate across active dimensions
                    
        return projected
    
    def batch_forward(self, 
                     tokens_batch: List[np.ndarray],
                     mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Optimized batch processing for multiple token sequences
        """
        # Stack tokens for batch processing
        stacked_tokens = np.stack(tokens_batch, axis=0)
        
        # Single forward pass for all batches
        batch_output = self.forward(stacked_tokens, mask)
        
        # Split back into individual outputs
        return [batch_output[i] for i in range(len(tokens_batch))]

class MultiScaleProjector:
    """
    Multi-scale token projection for hierarchical features
    """
    
    def __init__(self, scales: List[int] = [1, 2, 4]):
        self.scales = scales
        self.projectors = {
            scale: TokenProjector(dim=512//scale) 
            for scale in scales
        }
        
    def forward(self, 
                multi_scale_tokens: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Project tokens at multiple scales
        """
        projections = {}
        
        for scale, tokens in multi_scale_tokens.items():
            if scale in self.projectors:
                projections[scale] = self.projectors[scale].forward(tokens)
                
        return projections
    
    def cross_scale_fusion(self,
                          projections: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Fuse projections from different scales
        """
        fused = None
        
        for scale, proj in projections.items():
            if fused is None:
                fused = proj
            else:
                # Resize and fuse
                target_shape = fused.shape
                resized = self._resize_projection(proj, target_shape)
                fused = 0.5 * fused + 0.5 * resized
                
        return fused
    
    @staticmethod
    def _resize_projection(proj: np.ndarray, 
                          target_shape: Tuple) -> np.ndarray:
        """Resize projection to target shape"""
        if proj.shape == target_shape:
            return proj
            
        # Simple nearest-neighbor resize for projections
        batch_size, seq_len, dim = proj.shape
        target_seq_len = target_shape[1]
        
        if seq_len < target_seq_len:
            # Upsample by repeating with proper handling of odd dimensions
            factor = target_seq_len / seq_len
            if factor.is_integer():
                return np.repeat(proj, int(factor), axis=1)
            else:
                # Use interpolation for non-integer factors
                indices = np.linspace(0, seq_len-1, target_seq_len)
                return np.take(proj, indices.astype(int), axis=1)
        else:
            # Downsample by average pooling with proper handling of odd dimensions
            factor = seq_len / target_seq_len
            if factor.is_integer():
                return proj.reshape(batch_size, target_seq_len, int(factor), dim).mean(axis=2)
            else:
                # Use strided slicing for non-integer factors
                indices = np.round(np.linspace(0, seq_len-1, target_seq_len)).astype(int)
                return proj[np.arange(batch_size)[:, None], indices]