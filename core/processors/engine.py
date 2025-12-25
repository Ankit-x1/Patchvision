import numpy as np
from typing import Dict, List, Union, Optional
import time
import threading
from queue import Queue

class InferenceEngine:
    """
    Battlefield-ready inference engine with real-time optimization
    """
    
    def __init__(self,
                 mode: str = 'auto',
                 batch_size: int = 32,
                 use_quantization: bool = True):
        self.mode = mode
        self.batch_size = batch_size
        self.use_quantization = use_quantization
        self.quantization_table = self._build_quantization_table()
        
        # Performance monitoring
        self.latency_history = []
        self.throughput_history = []
        
        # Dynamic optimization
        self.optimal_batch_size = batch_size
        self.adaptive_threshold = 0.8
        
    def process(self, 
                inputs: Union[np.ndarray, List],
                model: callable) -> np.ndarray:
        """
        Process inputs with dynamic optimization
        """
        start_time = time.time()
        
        # Adaptive batching
        if len(inputs) > self.batch_size * 2:
            # Use dynamic batching
            batches = self._dynamic_batch(inputs)
            results = []
            
            for batch in batches:
                # Quantize if enabled
                if self.use_quantization:
                    batch = self._quantize_batch(batch)
                    
                # Process
                batch_result = model(batch)
                
                # Dequantize if needed
                if self.use_quantization:
                    batch_result = self._dequantize_batch(batch_result)
                    
                results.append(batch_result)
                
            output = np.concatenate(results, axis=0)
        else:
            # Single batch
            if self.use_quantization:
                inputs = self._quantize_batch(inputs)
                
            output = model(inputs)
            
            if self.use_quantization:
                output = self._dequantize_batch(output)
                
        # Update performance metrics
        latency = time.time() - start_time
        throughput = len(inputs) / latency if latency > 0 else 0
        
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        
        # Adaptive optimization
        self._adaptive_optimize(latency, throughput)
        
        return output
    
    def stream_process(self,
                      input_stream: Queue,
                      output_stream: Queue,
                      model: callable,
                      num_workers: int = 4):
        """
        Stream processing with multiple workers
        """
        def worker():
            while True:
                try:
                    batch = input_stream.get(timeout=1)
                    if batch is None:  # Termination signal
                        break
                        
                    result = self.process(batch, model)
                    output_stream.put(result)
                    
                except Queue.Empty:
                    continue
                    
        # Start workers
        workers = []
        for _ in range(num_workers):
            w = threading.Thread(target=worker)
            w.start()
            workers.append(w)
            
        return workers
    
    def _dynamic_batch(self, inputs: List) -> List[List]:
        """
        Create optimal batches based on input characteristics
        """
        batches = []
        current_batch = []
        current_size = 0
        
        for inp in inputs:
            inp_size = inp.nbytes if hasattr(inp, 'nbytes') else len(inp)
            
            if current_size + inp_size > self.optimal_batch_size * 1024:  # KB
                if current_batch:
                    batches.append(current_batch)
                current_batch = [inp]
                current_size = inp_size
            else:
                current_batch.append(inp)
                current_size += inp_size
                
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def _build_quantization_table(self) -> Dict:
        """Build dynamic quantization table"""
        return {
            'scale': 255.0,
            'zero_point': 128,
            'dtype': np.uint8
        }
        
    def _quantize_batch(self, batch: np.ndarray) -> np.ndarray:
        """Quantize batch to uint8"""
        if batch.dtype == np.uint8:
            return batch
            
        scale = self.quantization_table['scale']
        zero_point = self.quantization_table['zero_point']
        
        # Quantize
        quantized = np.clip(batch * scale + zero_point, 0, 255)
        return quantized.astype(np.uint8)
    
    def _dequantize_batch(self, batch: np.ndarray) -> np.ndarray:
        """Dequantize batch to float32"""
        if batch.dtype != np.uint8:
            return batch
            
        scale = self.quantization_table['scale']
        zero_point = self.quantization_table['zero_point']
        
        # Dequantize
        return (batch.astype(np.float32) - zero_point) / scale
    
    def _adaptive_optimize(self, latency: float, throughput: float):
        """
        Adaptively optimize processing parameters
        """
        # Update optimal batch size
        if len(self.latency_history) > 10:
            avg_latency = np.mean(self.latency_history[-10:])
            
            if avg_latency > self.adaptive_threshold:
                # Reduce batch size
                self.optimal_batch_size = max(1, self.optimal_batch_size // 2)
            elif avg_latency < self.adaptive_threshold * 0.5:
                # Increase batch size
                self.optimal_batch_size = min(256, self.optimal_batch_size * 2)

class OptimizedProcessor:
    """
    Hardware-optimized processor with automatic backend selection
    """
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.engines = {
            'cuda': self._init_cuda,
            'mps': self._init_mps,
            'cpu': self._init_cpu
        }
        
    def _detect_backend(self) -> str:
        """Detect available hardware backend"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
            
        return 'cpu'
    
    def process(self, 
                operation: str,
                *args,
                **kwargs):
        """
        Process operation with optimal backend
        """
        engine = self.engines[self.backend]()
        
        if operation == 'matmul':
            return engine.optimized_matmul(*args, **kwargs)
        elif operation == 'conv':
            return engine.optimized_conv(*args, **kwargs)
        elif operation == 'attention':
            return engine.optimized_attention(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _init_cuda(self):
        """Initialize CUDA backend"""
        from .gpu import GPUOptimizer
        return GPUOptimizer(device_id=0)
    
    def _init_mps(self):
        """Initialize MPS backend"""
        from .cpu import CPUProcessor
        processor = CPUProcessor()
        processor.backend = 'mps'
        return processor
    
    def _init_cpu(self):
        """Initialize CPU backend"""
        from .cpu import CPUProcessor
        return CPUProcessor()


class CudaEngine:
    """CUDA-specific optimized operations"""
    
    def __init__(self):
        from .gpu import GPUOptimizer
        self.optimizer = GPUOptimizer()
    
    def optimized_matmul(self, A, B, use_fp16=True):
        return self.optimizer.optimized_matmul(A, B, use_fp16)
    
    def optimized_conv(self, input, weights, stride=1, padding=0, use_winograd=True):
        return self.optimizer.optimized_conv(input, weights, stride, padding, use_winograd)
    
    def optimized_attention(self, Q, K, V, use_flash=True):
        return self.optimizer.optimized_attention(Q, K, V, use_flash)


class MpsEngine:
    """MPS-specific optimized operations"""
    
    def __init__(self):
        try:
            import torch
            self.device = torch.device('mps')
            self.has_torch = True
        except ImportError:
            self.has_torch = False
    
    def optimized_matmul(self, A, B, use_fp16=True):
        if not self.has_torch:
            return np.matmul(A, B)
        
        import torch
        A_t = torch.from_numpy(A).to(self.device)
        B_t = torch.from_numpy(B).to(self.device)
        
        if use_fp16:
            A_t = A_t.half()
            B_t = B_t.half()
        
        result = torch.matmul(A_t, B_t)
        return result.cpu().numpy()
    
    def optimized_conv(self, input, weights, stride=1, padding=0, use_winograd=True):
        if not self.has_torch:
            from scipy.signal import convolve2d
            return self._cpu_conv(input, weights, stride, padding)
        
        import torch
        import torch.nn.functional as F
        
        input_t = torch.from_numpy(input).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)
        
        result = F.conv2d(input_t, weights_t, stride=stride, padding=padding)
        return result.cpu().numpy()
    
    def optimized_attention(self, Q, K, V, use_flash=True):
        if not self.has_torch:
            return self._cpu_attention(Q, K, V)
        
        import torch
        import torch.nn.functional as F
        
        Q_t = torch.from_numpy(Q).to(self.device)
        K_t = torch.from_numpy(K).to(self.device)
        V_t = torch.from_numpy(V).to(self.device)
        
        scores = torch.matmul(Q_t, K_t.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(Q_t.shape[-1], device=self.device))
        attention = F.softmax(scores, dim=-1)
        result = torch.matmul(attention, V_t)
        
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
                        weights[oc, ic, ::-1, ::-1],
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


class CpuEngine:
    """CPU-optimized operations"""
    
    def __init__(self):
        try:
            from .cpu import CPUProcessor
            self.processor = CPUProcessor()
        except ImportError:
            self.processor = None
    
    def optimized_matmul(self, A, B, use_fp16=False):
        if use_fp16:
            return np.matmul(A.astype(np.float16), B.astype(np.float16))
        return np.matmul(A, B)
    
    def optimized_conv(self, input, weights, stride=1, padding=0, use_winograd=False):
        from scipy.signal import convolve2d
        return self._cpu_conv(input, weights, stride, padding)
    
    def optimized_attention(self, Q, K, V, use_flash=False):
        return self._cpu_attention(Q, K, V)
    
    @staticmethod
    def _cpu_conv(input, weights, stride, padding):
        """CPU convolution implementation"""
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
                        weights[oc, ic, ::-1, ::-1],
                        mode='same' if padding > 0 else 'valid'
                    )[::stride, ::stride]
        return output
    
    @staticmethod
    def _cpu_attention(Q, K, V):
        """CPU attention implementation"""
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        scores = scores / np.sqrt(Q.shape[-1])
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / np.sum(attention, axis=-1, keepdims=True)
        return np.matmul(attention, V)