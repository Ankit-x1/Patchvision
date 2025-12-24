import numpy as np
from typing import List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, prange

class CPUOptimizer:
    """
    CPU-specific optimizations with SIMD and multi-threading
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        self.num_threads = num_threads or mp.cpu_count()
        self.pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # Check for AVX512 support
        self.has_avx512 = self._check_avx512()
        
    @staticmethod
    def _check_avx512() -> bool:
        """Check for AVX512 support"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx512' in info.get('flags', [])
        except:
            return False
            
    def optimized_matmul(self, 
                        A: np.ndarray, 
                        B: np.ndarray,
                        use_simd: bool = True) -> np.ndarray:
        """
        SIMD optimized matrix multiplication
        """
        if use_simd and self.has_avx512:
            return self._avx512_matmul(A, B)
        else:
            # Use BLAS optimized numpy
            return np.matmul(A, B)
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _avx512_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Numba-accelerated matmul with AVX512 hints"""
        m, n = A.shape
        n, p = B.shape
        C = np.zeros((m, p))
        
        for i in prange(m):
            for j in prange(p):
                total = 0.0
                for k in range(n):
                    total += A[i, k] * B[k, j]
                C[i, j] = total
                
        return C
    
    def optimized_conv(self,
                      input: np.ndarray,
                      kernel: np.ndarray,
                      stride: int = 1,
                      padding: int = 0) -> np.ndarray:
        """
        Optimized convolution using im2col and GEMM
        """
        return self._im2col_convolution(input, kernel, stride, padding)
    
    @staticmethod
    def _im2col_convolution(input: np.ndarray,
                           kernel: np.ndarray,
                           stride: int,
                           padding: int) -> np.ndarray:
        """
        im2col based convolution (faster on CPU)
        """
        batch, in_ch, h, w = input.shape
        out_ch, _, kh, kw = kernel.shape
        
        # Add padding
        if padding > 0:
            input = np.pad(input, 
                          ((0, 0), (0, 0), 
                           (padding, padding), (padding, padding)),
                          mode='constant')
            
        # Output dimensions
        out_h = (h - kh + 2*padding) // stride + 1
        out_w = (w - kw + 2*padding) // stride + 1
        
        # im2col
        cols = np.zeros((batch, in_ch * kh * kw, out_h * out_w))
        
        for i in range(kh):
            for j in range(kw):
                row = i * kw + j
                for b in range(batch):
                    for c in range(in_ch):
                        cols[b, row::kh*kw, :] = \
                            input[b, c, i:i+out_h*stride:stride, 
                                  j:j+out_w*stride:stride].flatten()
                            
        # Reshape kernel and perform GEMM
        kernel_reshaped = kernel.reshape(out_ch, -1)
        output = np.matmul(kernel_reshaped, cols)
        
        # Reshape to output format
        return output.reshape(batch, out_ch, out_h, out_w)
    
    def parallel_process(self,
                        data: List,
                        func: callable,
                        chunk_size: Optional[int] = None) -> List:
        """
        Parallel processing with load balancing
        """
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.num_threads * 4))
            
        chunks = [data[i:i+chunk_size] 
                 for i in range(0, len(data), chunk_size)]
        
        # Process in parallel
        futures = [self.pool.submit(func, chunk) for chunk in chunks]
        results = [f.result() for f in futures]
        
        # Flatten results
        return [item for sublist in results for item in sublist]
    
    def vectorized_operation(self,
                            operation: str,
                            data: np.ndarray,
                            *args) -> np.ndarray:
        """
        Vectorized operations using numpy
        """
        if operation == 'normalize':
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            return (data - mean) / (std + 1e-8)
            
        elif operation == 'standardize':
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
            
        elif operation == 'pca':
            # Simplified PCA
            cov = np.cov(data.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            idx = eigvals.argsort()[::-1]
            return np.dot(data, eigvecs[:, idx[:args[0]]])
            
        else:
            raise ValueError(f"Unknown operation: {operation}")

class VectorizedProcessor:
    """
    Vectorized operations for CPU
    """
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def vectorized_dot(vectors: np.ndarray, 
                      matrix: np.ndarray) -> np.ndarray:
        """Vectorized dot product"""
        n_vectors = vectors.shape[0]
        n_features = vectors.shape[1]
        n_outputs = matrix.shape[1]
        
        result = np.zeros((n_vectors, n_outputs))
        
        for i in prange(n_vectors):
            for j in prange(n_outputs):
                total = 0.0
                for k in range(n_features):
                    total += vectors[i, k] * matrix[k, j]
                result[i, j] = total
                
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def batched_softmax(x: np.ndarray) -> np.ndarray:
        """Batched softmax with numerical stability"""
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        result = np.zeros_like(x)
        
        for b in prange(batch_size):
            # Find max for numerical stability
            max_val = x[b, 0]
            for i in range(1, seq_len):
                if x[b, i] > max_val:
                    max_val = x[b, i]
                    
            # Compute exponentials
            exp_sum = 0.0
            for i in range(seq_len):
                exp_val = np.exp(x[b, i] - max_val)
                result[b, i] = exp_val
                exp_sum += exp_val
                
            # Normalize
            for i in range(seq_len):
                result[b, i] /= exp_sum
                
        return result