import numpy as np
from typing import Optional
import warnings


try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available, GPU optimizations disabled")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

class GPUOptimizer:
    """
    GPU-specific optimizations using tensor cores
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.use_tensor_cores = self._check_tensor_cores()
        
        if HAS_TORCH:
            self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
            
    def _check_tensor_cores(self) -> bool:
        """Check if tensor cores are available"""
        if HAS_TORCH and torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(self.device_id)
            return prop.major >= 7  # Volta or newer
        return False
        
    def optimized_matmul(self, 
                        A: np.ndarray, 
                        B: np.ndarray,
                        use_fp16: bool = True) -> np.ndarray:
        """
        Tensor core optimized matrix multiplication
        """
        if not HAS_TORCH or self.device.type == 'cpu':
            # Fallback to numpy
            return np.matmul(A, B)
            
        # Convert to torch tensors
        A_t = torch.from_numpy(A).to(self.device)
        B_t = torch.from_numpy(B).to(self.device)
        
        if use_fp16 and self.use_tensor_cores:
            A_t = A_t.half()
            B_t = B_t.half()
            
        # Tensor core optimized matmul
        with torch.cuda.amp.autocast(enabled=use_fp16):
            result = torch.matmul(A_t, B_t)
            
        return result.cpu().numpy()
    
    def optimized_conv(self,
                      input: np.ndarray,
                      weights: np.ndarray,
                      stride: int = 1,
                      padding: int = 0,
                      use_winograd: bool = True) -> np.ndarray:
        """
        Optimized convolution using Winograd or implicit GEMM
        """
        if not HAS_TORCH:
            return self._cpu_conv(input, weights, stride, padding)
            
        input_t = torch.from_numpy(input).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)
        
        # Use optimized convolution
        if use_winograd and input_t.shape[1] <= 64:  # Winograd works best for small channels
            result = F.conv2d(input_t, weights_t, 
                            stride=stride, 
                            padding=padding)
        else:
            # Use cuDNN implicit GEMM
            result = F.conv2d(input_t, weights_t,
                            stride=stride,
                            padding=padding)
                            
        return result.cpu().numpy()
    
    def optimized_attention(self,
                           Q: np.ndarray,
                           K: np.ndarray,
                           V: np.ndarray,
                           use_flash: bool = True) -> np.ndarray:
        """
        Flash attention optimization
        """
        if not HAS_TORCH:
            return self._cpu_attention(Q, K, V)
            
        Q_t = torch.from_numpy(Q).to(self.device)
        K_t = torch.from_numpy(K).to(self.device)
        V_t = torch.from_numpy(V).to(self.device)
        
        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's built-in flash attention
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                result = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        else:
            # Standard attention
            scores = torch.matmul(Q_t, K_t.transpose(-2, -1))
            scores = scores / torch.sqrt(torch.tensor(Q_t.shape[-1], device=self.device))
            attention = F.softmax(scores, dim=-1)
            result = torch.matmul(attention, V_t)
            
        return result.cpu().numpy()
    
    def batch_normalization(self,
                           x: np.ndarray,
                           gamma: np.ndarray,
                           beta: np.ndarray,
                           running_mean: np.ndarray,
                           running_var: np.ndarray,
                           eps: float = 1e-5) -> np.ndarray:
        """
        Fused batch normalization
        """
        if not HAS_TORCH:
            return self._cpu_batch_norm(x, gamma, beta, running_mean, running_var, eps)
            
        x_t = torch.from_numpy(x).to(self.device)
        
        # Fused batch norm
        result = F.batch_norm(x_t,
                            torch.from_numpy(running_mean).to(self.device),
                            torch.from_numpy(running_var).to(self.device),
                            torch.from_numpy(gamma).to(self.device),
                            torch.from_numpy(beta).to(self.device),
                            training=False,
                            eps=eps)
                            
        return result.cpu().numpy()
    
    @staticmethod
    def _cpu_conv(input, weights, stride, padding):
        """CPU fallback convolution"""
        from scipy.signal import convolve2d
        
        batch, in_ch, h, w = input.shape
        out_ch, _, kh, kw = weights.shape
        
        output = np.zeros((batch, out_ch, 
                          (h - kh + 2*padding)//stride + 1,
                          (w - kw + 2*padding)//stride + 1))
        
        for b in range(batch):
            for oc in range(out_ch):
                for ic in range(in_ch):
                    output[b, oc] += convolve2d(
                        input[b, ic],
                        weights[oc, ic, ::-1, ::-1],  # Flip for correlation
                        mode='same' if padding > 0 else 'valid'
                    )[::stride, ::stride]
                    
        return output
    
    @staticmethod
    def _cpu_attention(Q, K, V):
        """CPU fallback attention"""
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        scores = scores / np.sqrt(Q.shape[-1])
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        return np.matmul(attention, V)
    
    @staticmethod
    def _cpu_batch_norm(x, gamma, beta, mean, var, eps):
        """CPU fallback batch norm"""
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

class TensorCoreEngine:
    """
    NVIDIA Tensor Core specific optimizations
    """
    
    def __init__(self):
        if HAS_CUPY:
            self.use_cupy = True
        else:
            self.use_cupy = False
            
    def fp16_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """FP16 tensor core matmul"""
        if self.use_cupy:
            A_cp = cp.asarray(A, dtype=cp.float16)
            B_cp = cp.asarray(B, dtype=cp.float16)
            result = cp.matmul(A_cp, B_cp)
            return cp.asnumpy(result)
        else:
            # Fallback with numpy
            return np.matmul(A.astype(np.float16), B.astype(np.float16))
            
    def mixed_precision_training(self, model, data, loss_fn):
        """Mixed precision training wrapper"""
        if HAS_TORCH:
            scaler = torch.cuda.amp.GradScaler()
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = loss_fn(output)
                
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()