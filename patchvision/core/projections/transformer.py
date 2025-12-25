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
        # Linear projections - ensure proper matrix multiplication
        batch_size, seq_len, token_dim = tokens.shape
        if token_dim != self.dim:
            # Project tokens to correct dimension first
            tokens_proj = np.einsum('bsd,dk->bsk', tokens, 
                                np.random.randn(token_dim, self.dim) * np.sqrt(2.0/(token_dim + self.dim)))
        else:
            tokens_proj = tokens
            
        Q = self._optimized_matmul(tokens_proj, self.W_q)
        K = self._optimized_matmul(tokens_proj, self.W_k)
        V = self._optimized_matmul(tokens_proj, self.W_v)
        
        # Split into heads
        batch_size, seq_len, _ = tokens.shape
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Manual softmax implementation with numerical stability
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        attention = exp_scores / (sum_exp + 1e-8)  # Add small epsilon for numerical stability
        
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
                result = np.matmul(A.astype(np.float16), B.astype(np.float16))
                return result.astype(np.float32)
            else:
                return np.matmul(A, B)
        else:
            # For smaller matrices, use einsum for better cache locality
            if self.use_fp16 and A.dtype != np.float16:
                return np.matmul(A.astype(np.float16), B.astype(np.float16)).astype(np.float32)
            else:
                return np.matmul(A, B)
    
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


class PositionalEmbedding:
    """
    Positional embeddings for transformer models
    """
    
    def __init__(self, d_model: int = 512, max_seq_len: int = 5000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embeddings = self._create_embeddings()
    
    def _create_embeddings(self) -> np.ndarray:
        """Create sinusoidal positional embeddings"""
        position = np.arange(self.max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         (-np.log(10000.0) / self.d_model))
        
        embeddings = np.zeros((self.max_seq_len, self.d_model))
        embeddings[:, 0::2] = np.sin(position * div_term)
        embeddings[:, 1::2] = np.cos(position * div_term)
        
        return embeddings
    
    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional embeddings for sequence length"""
        return self.embeddings[:seq_len]
    
    def add_to_input(self, x: np.ndarray) -> np.ndarray:
        """Add positional embeddings to input"""
        seq_len = x.shape[1]
        pos_emb = self(seq_len)
        return x + pos_emb


class TransformerEncoderLayer:
    """
    Single transformer encoder layer
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        
        # Initialize weights
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        self.W_o = self._init_weights(d_model, d_model)
        
        # Feed-forward weights
        self.W_ff1 = self._init_weights(d_model, d_ff)
        self.W_ff2 = self._init_weights(d_ff, d_model)
        
        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        # Dropout parameters (simplified)
        self.dropout_rate = 0.1
    
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize weights"""
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * scale
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through encoder layer"""
        # Multi-head attention
        attn_output = self._multi_head_attention(x, x, x, mask)
        
        # Add & norm
        x = self._layer_norm(x + attn_output, self.gamma1, self.beta1)
        
        # Feed-forward
        ff_output = self._feed_forward(x)
        
        # Add & norm
        x = self._layer_norm(x + ff_output, self.gamma2, self.beta2)
        
        return x
    
    def _multi_head_attention(self, 
                            Q: np.ndarray, 
                            K: np.ndarray, 
                            V: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Multi-head attention computation"""
        batch_size, seq_len, _ = Q.shape
        
        # Linear projections
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        
        # Apply attention
        output = np.matmul(attention, V)
        
        # Transpose back and reshape
        output = output.transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = np.matmul(output, self.W_o)
        
        return output
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network"""
        # First linear layer + ReLU
        x = np.matmul(x, self.W_ff1)
        x = np.maximum(0, x)  # ReLU
        
        # Second linear layer
        x = np.matmul(x, self.W_ff2)
        
        return x
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-6)
        
        return gamma * (x - mean) / std + beta


class TransformerEncoder:
    """
    Complete transformer encoder
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 n_layers: int = 6,
                 max_seq_len: int = 5000):
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        
        # Encoder layers
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
    
    def forward(self, 
                x: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through encoder"""
        # Add positional embeddings
        x = self.pos_embedding.add_to_input(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x


class TransformerDecoderLayer:
    """
    Single transformer decoder layer
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        
        # Self-attention weights
        self.W_q_self = self._init_weights(d_model, d_model)
        self.W_k_self = self._init_weights(d_model, d_model)
        self.W_v_self = self._init_weights(d_model, d_model)
        self.W_o_self = self._init_weights(d_model, d_model)
        
        # Cross-attention weights
        self.W_q_cross = self._init_weights(d_model, d_model)
        self.W_k_cross = self._init_weights(d_model, d_model)
        self.W_v_cross = self._init_weights(d_model, d_model)
        self.W_o_cross = self._init_weights(d_model, d_model)
        
        # Feed-forward weights
        self.W_ff1 = self._init_weights(d_model, d_ff)
        self.W_ff2 = self._init_weights(d_ff, d_model)
        
        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)
    
    def _init_weights(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize weights"""
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * scale
    
    def forward(self, 
                x: np.ndarray,
                encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through decoder layer"""
        # Self-attention
        self_attn = self._self_attention(x, self_mask)
        x = self._layer_norm(x + self_attn, self.gamma1, self.beta1)
        
        # Cross-attention
        cross_attn = self._cross_attention(x, encoder_output, cross_mask)
        x = self._layer_norm(x + cross_attn, self.gamma2, self.beta2)
        
        # Feed-forward
        ff_output = self._feed_forward(x)
        x = self._layer_norm(x + ff_output, self.gamma3, self.beta3)
        
        return x
    
    def _self_attention(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Self-attention computation"""
        # Simplified self-attention (similar to encoder)
        Q = np.matmul(x, self.W_q_self)
        K = np.matmul(x, self.W_k_self)
        V = np.matmul(x, self.W_v_self)
        
        batch_size, seq_len, _ = Q.shape
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        
        output = np.matmul(attention, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.matmul(output, self.W_o_self)
        
        return output
    
    def _cross_attention(self, 
                        x: np.ndarray,
                        encoder_output: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Cross-attention computation"""
        Q = np.matmul(x, self.W_q_cross)
        K = np.matmul(encoder_output, self.W_k_cross)
        V = np.matmul(encoder_output, self.W_v_cross)
        
        batch_size, seq_len, _ = Q.shape
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        
        output = np.matmul(attention, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.matmul(output, self.W_o_cross)
        
        return output
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network"""
        x = np.matmul(x, self.W_ff1)
        x = np.maximum(0, x)  # ReLU
        x = np.matmul(x, self.W_ff2)
        return x
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-6)
        return gamma * (x - mean) / std + beta


class TransformerDecoder:
    """
    Complete transformer decoder
    """
    
    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 n_layers: int = 6,
                 max_seq_len: int = 5000):
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        
        # Decoder layers
        self.layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
    
    def forward(self,
                x: np.ndarray,
                encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through decoder"""
        # Add positional embeddings
        x = self.pos_embedding.add_to_input(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer.forward(x, encoder_output, self_mask, cross_mask)
        
        return x


class CompleteTransformer:
    """
    Complete transformer with encoder and decoder
    """
    
    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 max_seq_len: int = 5000,
                 vocab_size: int = 10000):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_encoder_layers, max_seq_len)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, n_decoder_layers, max_seq_len)
        
        # Output projection
        self.W_output = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
    
    def forward(self,
                src: np.ndarray,
                tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through complete transformer"""
        # Encode source
        encoder_output = self.encoder.forward(src, src_mask)
        
        # Decode target
        decoder_output = self.decoder.forward(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = np.matmul(decoder_output, self.W_output)
        
        return output
    
    def encode(self, src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode source sequence"""
        return self.encoder.forward(src, mask)
    
    def decode(self,
               tgt: np.ndarray,
               encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode target sequence"""
        return self.decoder.forward(tgt, encoder_output, tgt_mask, src_mask)