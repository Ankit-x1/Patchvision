"""
Performance benchmarking tools for PatchVision
"""

import time
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
import statistics
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Benchmark result data"""
    operation: str
    avg_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float
    samples: int
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """
    Simple performance benchmarking for individual operations
    """
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(self, 
                          func: Callable,
                          operation_name: str,
                          *args, **kwargs) -> BenchmarkResult:
        """Benchmark a single function"""
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        # Calculate throughput (samples per second)
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        result = BenchmarkResult(
            operation=operation_name,
            avg_time=avg_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            throughput=throughput,
            samples=self.benchmark_runs,
            metadata={}
        )
        
        self.results.append(result)
        return result
    
    def benchmark_inference(self, 
                           model: Any,
                           inputs: np.ndarray,
                           operation_name: str = "inference") -> BenchmarkResult:
        """Benchmark model inference"""
        def inference_func():
            return model.predict(inputs)
        
        return self.benchmark_function(inference_func, operation_name)
    
    def benchmark_batch_processing(self,
                                  processor: Any,
                                  data: List[np.ndarray],
                                  operation_name: str = "batch_processing") -> BenchmarkResult:
        """Benchmark batch processing"""
        def batch_func():
            return processor.process_batches(data)
        
        return self.benchmark_function(batch_func, operation_name)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        if not self.results:
            return {}
        
        summary = {
            'total_operations': len(self.results),
            'operations': {}
        }
        
        for result in self.results:
            summary['operations'][result.operation] = {
                'avg_time': result.avg_time,
                'std_time': result.std_time,
                'throughput': result.throughput,
                'samples': result.samples
            }
        
        return summary
    
    def save_results(self, file_path: str):
        """Save benchmark results to file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'operation': result.operation,
                'avg_time': result.avg_time,
                'std_time': result.std_time,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'throughput': result.throughput,
                'samples': result.samples,
                'metadata': result.metadata
            })
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for PatchVision
    """
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark = PerformanceBenchmark()
        self.latency_measurements = []
        self.throughput_measurements = []
    
    def run_latency_benchmark(self, 
                             model: Any,
                             test_data: np.ndarray,
                             num_runs: int = 100) -> Dict[str, float]:
        """Measure inference latency"""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model.predict(test_data)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latency_stats = {
            'avg_latency_ms': statistics.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        self.latency_measurements.append(latency_stats)
        return latency_stats
    
    def run_throughput_benchmark(self,
                                model: Any,
                                batch_sizes: List[int],
                                duration_seconds: float = 30.0) -> Dict[int, float]:
        """Measure throughput at different batch sizes"""
        throughput_results = {}
        
        for batch_size in batch_sizes:
            # Generate test data
            test_data = np.random.randn(batch_size, 224, 224, 3)
            
            start_time = time.time()
            samples_processed = 0
            
            while time.time() - start_time < duration_seconds:
                _ = model.predict(test_data)
                samples_processed += batch_size
            
            actual_duration = time.time() - start_time
            throughput = samples_processed / actual_duration
            throughput_results[batch_size] = throughput
        
        self.throughput_measurements.append(throughput_results)
        return throughput_results
    
    def run_memory_benchmark(self, 
                           model: Any,
                           input_sizes: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Estimate memory usage for different input sizes"""
        memory_results = {}
        
        for size in input_sizes:
            h, w, c = size
            test_input = np.random.randn(1, h, w, c)
            
            # Estimate input size
            input_memory = test_input.nbytes / (1024 * 1024)  # MB
            
            # Rough estimation of model memory usage
            try:
                # Try to get model parameters count
                if hasattr(model, 'get_parameters'):
                    param_count = sum(p.numel() for p in model.get_parameters().values())
                else:
                    param_count = 1000000  # Default estimate
                
                # Assume 4 bytes per parameter (float32)
                model_memory = param_count * 4 / (1024 * 1024)  # MB
                
                total_memory = input_memory + model_memory
                memory_results[f"{h}x{w}x{c}"] = total_memory
                
            except Exception:
                memory_results[f"{h}x{w}x{c}"] = input_memory
        
        return memory_results
    
    def run_comprehensive_benchmark(self, 
                                  model: Any,
                                  test_data: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        results = {
            'latency': self.run_latency_benchmark(model, test_data),
            'throughput': self.run_throughput_benchmark(model, [1, 4, 8, 16, 32]),
            'memory': self.run_memory_benchmark(model, [(224, 224, 3), (512, 512, 3)]),
            'individual_operations': self.benchmark.get_results_summary()
        }
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_models(self, 
                      models: Dict[str, Any],
                      test_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare multiple models"""
        comparison_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.run_latency_benchmark(model, test_data, num_runs=50)
                comparison_results[model_name] = results
            except Exception as e:
                comparison_results[model_name] = {'error': str(e)}
        
        return comparison_results
    
    def generate_report(self) -> str:
        """Generate benchmark report"""
        if not self.latency_measurements and not self.throughput_measurements:
            return "No benchmark data available"
        
        report = ["# PatchVision Benchmark Report\n"]
        
        if self.latency_measurements:
            latest_latency = self.latency_measurements[-1]
            report.append("## Latency Results")
            report.append(f"- Average: {latest_latency['avg_latency_ms']:.2f} ms")
            report.append(f"- P95: {latest_latency['p95_latency_ms']:.2f} ms")
            report.append(f"- P99: {latest_latency['p99_latency_ms']:.2f} ms")
        
        if self.throughput_measurements:
            latest_throughput = self.throughput_measurements[-1]
            report.append("\n## Throughput Results")
            for batch_size, throughput in latest_throughput.items():
                report.append(f"- Batch {batch_size}: {throughput:.1f} samples/sec")
        
        return "\n".join(report)
