import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
import gc
import threading
from collections import defaultdict
import time

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


class GPUMemoryManager:
    """
    Advanced GPU memory management with pooling and OOM handling
    """
    
    def __init__(self, device_id: int = 0, max_pool_size: float = 0.8):
        self.device_id = device_id
        self.max_pool_size = max_pool_size
        self.memory_pool = {}
        self.allocated_tensors = defaultdict(list)
        self.oom_count = 0
        self.memory_stats = {
            'allocations': 0,
            'deallocations': 0,
            'oom_recoveries': 0
        }
        
        if HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self.total_memory = torch.cuda.get_device_properties(device_id).total_memory
            self.max_pool_bytes = int(self.total_memory * max_pool_size)
        else:
            self.device = None
            self.total_memory = 0
            self.max_pool_bytes = 0
            
        self._lock = threading.Lock()
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: str = 'float32') -> torch.Tensor:
        """Allocate tensor with memory pooling"""
        if not HAS_TORCH or self.device is None:
            return torch.zeros(shape, dtype=getattr(torch, dtype))
        
        key = (shape, dtype)
        
        with self._lock:
            # Try to reuse from pool
            if key in self.memory_pool and self.memory_pool[key]:
                tensor = self.memory_pool[key].pop()
                if tensor.shape == shape and tensor.dtype == getattr(torch, dtype):
                    self.memory_stats['allocations'] += 1
                    return tensor
            
            # Check available memory
            if not self._check_memory_availability(shape, dtype):
                self._handle_oom()
                return self.allocate_tensor(shape, dtype)  # Retry after cleanup
            
            # Allocate new tensor
            try:
                tensor = torch.zeros(shape, dtype=getattr(torch, dtype), device=self.device)
                self.allocated_tensors[key].append(tensor)
                self.memory_stats['allocations'] += 1
                return tensor
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.oom_count += 1
                    self._handle_oom()
                    return self.allocate_tensor(shape, dtype)  # Retry after cleanup
                raise
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Deallocate tensor back to pool"""
        if not HAS_TORCH or tensor.device.type != 'cuda':
            return
        
        key = (tensor.shape, str(tensor.dtype).split('.')[-1])
        
        with self._lock:
            if len(self.memory_pool.get(key, [])) < 10:  # Limit pool size per shape
                if key not in self.memory_pool:
                    self.memory_pool[key] = []
                self.memory_pool[key].append(tensor)
            else:
                # Remove from allocated list if pool is full
                if tensor in self.allocated_tensors[key]:
                    self.allocated_tensors[key].remove(tensor)
                del tensor
            
            self.memory_stats['deallocations'] += 1
    
    def _check_memory_availability(self, shape: Tuple[int, ...], dtype: str) -> bool:
        """Check if enough memory is available"""
        if not HAS_TORCH:
            return True
        
        try:
            required_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            current_memory = torch.cuda.memory_allocated(self.device_id)
            available_memory = self.total_memory - current_memory
            
            return available_memory >= required_bytes and (current_memory + required_bytes) < self.max_pool_bytes
        except:
            return False
    
    def _handle_oom(self):
        """Handle out of memory situation"""
        self.memory_stats['oom_recoveries'] += 1
        
        # Clear memory pool
        self.memory_pool.clear()
        
        # Force garbage collection
        gc.collect()
        
        if HAS_TORCH:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Clear unused tensors
            for key, tensors in list(self.allocated_tensors.items()):
                for tensor in tensors[:]:
                    if tensor.numel() == 0 or not tensor.is_pinned():
                        tensors.remove(tensor)
                        del tensor
    
    def get_memory_info(self) -> Dict:
        """Get current memory usage information"""
        if not HAS_TORCH:
            return {'error': 'PyTorch not available'}
        
        return {
            'allocated': torch.cuda.memory_allocated(self.device_id),
            'cached': torch.cuda.memory_reserved(self.device_id),
            'max_allocated': torch.cuda.max_memory_allocated(self.device_id),
            'total': self.total_memory,
            'pool_size': len(self.memory_pool),
            'stats': self.memory_stats.copy()
        }
    
    def cleanup(self):
        """Clean up all allocated memory"""
        with self._lock:
            self.memory_pool.clear()
            
            for tensors in self.allocated_tensors.values():
                for tensor in tensors:
                    del tensor
            self.allocated_tensors.clear()
            
            if HAS_TORCH:
                torch.cuda.empty_cache()


class GPUMemoryPool:
    """
    Specialized memory pool for specific tensor shapes
    """
    
    def __init__(self, max_size_per_shape: int = 20):
        self.pools = defaultdict(list)
        self.max_size_per_shape = max_size_per_shape
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get tensor from pool or allocate new one"""
        key = (shape, dtype)
        
        with self._lock:
            if self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()
                return tensor
        
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if tensor.device.type != 'cuda':
            return
        
        key = (tensor.shape, tensor.dtype)
        
        with self._lock:
            if len(self.pools[key]) < self.max_size_per_shape:
                self.pools[key].append(tensor)
            else:
                del tensor
    
    def clear(self):
        """Clear all pools"""
        with self._lock:
            for tensors in self.pools.values():
                for tensor in tensors:
                    del tensor
            self.pools.clear()


class OOMHandler:
    """
    Out of memory error handler with recovery strategies
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.recovery_strategies = [
            self._clear_cache,
            self._force_gc,
            self._reduce_batch_size,
            self._fallback_to_cpu
        ]
    
    def handle_oom(self, func, *args, **kwargs):
        """Handle OOM error with recovery strategies"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                
                last_exception = e
                
                if attempt < len(self.recovery_strategies):
                    self.recovery_strategies[attempt]()
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError(f"Failed after {self.max_retries} OOM recovery attempts") from last_exception
    
    def _clear_cache(self):
        """Clear CUDA cache"""
        if HAS_TORCH:
            torch.cuda.empty_cache()
    
    def _force_gc(self):
        """Force garbage collection"""
        gc.collect()
    
    def _reduce_batch_size(self):
        """Reduce batch size (placeholder for actual implementation)"""
        pass
    
    def _fallback_to_cpu(self):
        """Fallback to CPU processing"""
        pass