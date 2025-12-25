"""
Batch processing utilities for efficient inference
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue


class BatchProcessor:
    """
    High-performance batch processing with parallel execution
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_workers: Optional[int] = None,
                 prefetch_batches: int = 2):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, (batch_size // 4) + 1)
        self.prefetch_batches = prefetch_batches
        self.processing_times = []
        
        # Thread-safe queues for pipeline
        self.input_queue = Queue(maxsize=prefetch_batches * 2)
        self.output_queue = Queue(maxsize=prefetch_batches * 2)
        self._stop_event = threading.Event()
        
    def process_batches(self, 
                       data: List[np.ndarray],
                       process_func: Callable[[np.ndarray], np.ndarray],
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> List[np.ndarray]:
        """
        Process data in batches with parallel execution
        """
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        results = [None] * total_batches
        
        # Create batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch))
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(self._process_single_batch, batch_idx, batch, process_func): (batch_idx, batch)
                for batch_idx, batch in batches
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx, result = future.result()
                results[batch_idx] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_batches)
        
        return results
    
    def _process_single_batch(self, 
                            batch_idx: int,
                            batch: np.ndarray,
                            process_func: Callable[[np.ndarray], np.ndarray]) -> Tuple[int, np.ndarray]:
        """Process a single batch with timing"""
        start_time = time.time()
        
        try:
            result = process_func(batch)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return batch_idx, result
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            return batch_idx, np.array([])  # Return empty result on error
    
    def process_streaming(self,
                        data_stream: Queue,
                        process_func: Callable[[np.ndarray], np.ndarray],
                        output_stream: Queue) -> None:
        """
        Process data in streaming fashion with prefetching
        """
        def producer():
            """Producer thread for batching input data"""
            batch = []
            while not self._stop_event.is_set():
                try:
                    data_item = data_stream.get(timeout=0.1)
                    batch.append(data_item)
                    
                    if len(batch) >= self.batch_size:
                        self.input_queue.put(batch.copy())
                        batch = []
                        
                except:
                    continue
            
            # Put remaining batch
            if batch:
                self.input_queue.put(batch)
        
        def consumer():
            """Consumer thread for processing batches"""
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while not self._stop_event.is_set():
                    try:
                        batch = self.input_queue.get(timeout=0.1)
                        
                        # Process batch
                        future = executor.submit(process_func, batch)
                        result = future.result()
                        
                        # Put result in output queue
                        output_stream.put(result)
                        
                    except:
                        continue
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        return producer_thread, consumer_thread
    
    def stop_streaming(self):
        """Stop streaming processing"""
        self._stop_event.set()
    
    def adaptive_batch_processing(self,
                               data: List[np.ndarray],
                               process_func: Callable[[np.ndarray], np.ndarray],
                               target_latency: float = 0.1) -> List[np.ndarray]:
        """
        Adaptive batch processing based on latency targets
        """
        # Start with current batch size
        current_batch_size = self.batch_size
        best_batch_size = current_batch_size
        best_throughput = 0
        
        # Test different batch sizes
        test_sizes = [current_batch_size // 2, current_batch_size, current_batch_size * 2]
        
        for test_size in test_sizes:
            if test_size < 1 or test_size > len(data):
                continue
            
            # Time processing for this batch size
            start_time = time.time()
            test_batch = data[:test_size]
            
            try:
                _ = process_func(test_batch)
                processing_time = time.time() - start_time
                throughput = test_size / processing_time
                
                # Check if meets latency target
                if processing_time <= target_latency and throughput > best_throughput:
                    best_batch_size = test_size
                    best_throughput = throughput
                    
            except Exception:
                continue
        
        # Process all data with optimal batch size
        self.batch_size = best_batch_size
        return self.process_batches(data, process_func)
    
    def process_with_overlap(self,
                           data: List[np.ndarray],
                           process_func: Callable[[np.ndarray], np.ndarray],
                           overlap_ratio: float = 0.25) -> List[np.ndarray]:
        """
        Process batches with overlap for better temporal consistency
        """
        overlap_size = int(self.batch_size * overlap_ratio)
        actual_batch_size = self.batch_size - overlap_size
        
        results = []
        
        for i in range(0, len(data), actual_batch_size):
            # Create overlapping batch
            start_idx = max(0, i - overlap_size)
            end_idx = min(len(data), i + self.batch_size)
            
            batch = data[start_idx:end_idx]
            
            # Process batch
            batch_result = process_func(batch)
            
            # Extract only the non-overlapping portion
            if len(batch_result) == len(batch):
                result_start = overlap_size if i > 0 else 0
                result_end = actual_batch_size if i + self.batch_size <= len(data) else len(batch_result) - result_start
                results.append(batch_result[result_start:result_start + result_end])
            else:
                results.append(batch_result)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_batch_time': np.mean(self.processing_times),
            'std_batch_time': np.std(self.processing_times),
            'min_batch_time': np.min(self.processing_times),
            'max_batch_time': np.max(self.processing_times),
            'total_batches': len(self.processing_times),
            'current_batch_size': self.batch_size,
            'avg_throughput': self.batch_size / np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def optimize_batch_size(self,
                          sample_data: List[np.ndarray],
                          process_func: Callable[[np.ndarray], np.ndarray],
                          min_size: int = 1,
                          max_size: int = 256) -> int:
        """
        Find optimal batch size through binary search
        """
        low, high = min_size, max_size
        best_size = self.batch_size
        best_throughput = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            # Test mid batch size
            test_data = sample_data[:mid]
            
            try:
                start_time = time.time()
                _ = process_func(test_data)
                processing_time = time.time() - start_time
                
                if processing_time > 0:
                    throughput = mid / processing_time
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_size = mid
                
                # Adjust search range based on performance
                if processing_time < 0.1:  # Fast processing, try larger batches
                    low = mid + 1
                else:  # Slow processing, try smaller batches
                    high = mid - 1
                    
            except Exception:
                high = mid - 1
        
        self.batch_size = best_size
        return best_size


class MemoryEfficientBatchProcessor:
    """
    Memory-efficient batch processing for large datasets
    """
    
    def __init__(self, max_memory_mb: float = 1024):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
    
    def estimate_batch_memory(self, sample: np.ndarray) -> float:
        """Estimate memory usage for a single sample"""
        return sample.nbytes
    
    def calculate_optimal_batch_size(self, sample: np.ndarray) -> int:
        """Calculate optimal batch size based on memory constraints"""
        sample_memory = self.estimate_batch_memory(sample)
        
        # Reserve 50% of memory for processing overhead
        available_memory = self.max_memory_bytes * 0.5
        
        optimal_size = int(available_memory / sample_memory)
        return max(1, optimal_size)
    
    def process_memory_constrained(self,
                                 data: List[np.ndarray],
                                 process_func: Callable[[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """
        Process data with memory constraints
        """
        if not data:
            return []
        
        # Calculate optimal batch size
        sample = data[0]
        optimal_batch_size = self.calculate_optimal_batch_size(sample)
        
        results = []
        
        # Process in memory-constrained batches
        for i in range(0, len(data), optimal_batch_size):
            batch = data[i:i + optimal_batch_size]
            batch_result = process_func(batch)
            results.append(batch_result)
        
        return results
