"""
Debug visualization utilities for PatchVision
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from pathlib import Path


class DebugVisualizer:
    """
    Simple visualization utilities for debugging and analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_latency_distribution(self, latencies: List[float], title: str = "Latency Distribution"):
        """Plot latency distribution"""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(latencies, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title(f'{title} - Histogram')
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(latencies)
        plt.ylabel('Latency (ms)')
        plt.title(f'{title} - Box Plot')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"latency_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_throughput_comparison(self, throughput_data: Dict[str, float], 
                                 title: str = "Throughput Comparison"):
        """Plot throughput comparison across models or configurations"""
        plt.figure(figsize=(10, 6))
        
        models = list(throughput_data.keys())
        throughputs = list(throughput_data.values())
        
        plt.bar(models, throughputs, color='skyblue')
        plt.xlabel('Model/Configuration')
        plt.ylabel('Throughput (samples/sec)')
        plt.title(title)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"throughput_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_memory_usage(self, memory_data: List[Dict[str, float]], 
                         title: str = "Memory Usage Over Time"):
        """Plot memory usage over time"""
        if not memory_data:
            return
        
        plt.figure(figsize=(12, 6))
        
        timestamps = [d['timestamp'] for d in memory_data]
        rss_memory = [d['rss_mb'] for d in memory_data]
        vms_memory = [d['vms_mb'] for d in memory_data]
        
        # Normalize timestamps to start from 0
        if timestamps:
            start_time = timestamps[0]
            timestamps = [(t - start_time) for t in timestamps]
        
        plt.plot(timestamps, rss_memory, label='RSS', color='blue')
        plt.plot(timestamps, vms_memory, label='VMS', color='red')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"memory_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_performance_heatmap(self, performance_data: Dict[str, Dict[str, float]], 
                                metric: str = 'avg_time'):
        """Plot performance heatmap"""
        if not performance_data:
            return
        
        # Prepare data for heatmap
        operations = list(performance_data.keys())
        metrics_data = []
        
        for op in operations:
            if metric in performance_data[op]:
                metrics_data.append(performance_data[op][metric])
            else:
                metrics_data.append(0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Reshape data for heatmap (operations x 1)
        data = np.array(metrics_data).reshape(-1, 1)
        
        sns.heatmap(data, annot=True, fmt='.4f', 
                   xticklabels=[metric], yticklabels=operations,
                   cmap='YlOrRd')
        
        plt.title(f'Performance Heatmap - {metric}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"heatmap_{metric}_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_batch_size_analysis(self, batch_sizes: List[int], 
                                latencies: List[float], 
                                throughputs: List[float]):
        """Plot batch size analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency vs Batch Size
        ax1.plot(batch_sizes, latencies, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency vs Batch Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Throughput vs Batch Size
        ax2.plot(batch_sizes, throughputs, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"batch_analysis_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_attention_weights(self, attention_weights: np.ndarray, 
                              title: str = "Attention Weights"):
        """Visualize attention weights"""
        plt.figure(figsize=(10, 8))
        
        if len(attention_weights.shape) == 3:  # (batch, seq, seq)
            # Average across batch
            avg_attention = np.mean(attention_weights, axis=0)
        elif len(attention_weights.shape) == 2:  # (seq, seq)
            avg_attention = attention_weights
        else:
            print(f"Unexpected attention weights shape: {attention_weights.shape}")
            return
        
        sns.heatmap(avg_attention, cmap='Blues', 
                   xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"attention_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def plot_feature_maps(self, feature_maps: np.ndarray, 
                         title: str = "Feature Maps",
                         max_channels: int = 16):
        """Visualize feature maps"""
        if len(feature_maps.shape) != 4:  # (batch, channels, height, width)
            print(f"Expected 4D feature maps, got shape: {feature_maps.shape}")
            return
        
        # Take first sample
        sample_maps = feature_maps[0]
        
        # Limit number of channels to display
        num_channels = min(sample_maps.shape[0], max_channels)
        
        # Calculate grid size
        cols = int(np.sqrt(num_channels))
        rows = (num_channels + cols - 1) // cols
        
        plt.figure(figsize=(cols * 2, rows * 2))
        
        for i in range(num_channels):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(sample_maps[i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Channel {i}')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"feature_maps_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def create_performance_dashboard(self, 
                                  latency_data: List[float],
                                  throughput_data: Dict[str, float],
                                  memory_data: List[Dict[str, float]]):
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency distribution
        axes[0, 0].hist(latency_data, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        
        # Throughput comparison
        if throughput_data:
            models = list(throughput_data.keys())
            throughputs = list(throughput_data.values())
            axes[0, 1].bar(models, throughputs, color='skyblue')
            axes[0, 1].set_xlabel('Model/Configuration')
            axes[0, 1].set_ylabel('Throughput (samples/sec)')
            axes[0, 1].set_title('Throughput Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        if memory_data:
            timestamps = [d['timestamp'] for d in memory_data]
            rss_memory = [d['rss_mb'] for d in memory_data]
            
            if timestamps:
                start_time = timestamps[0]
                timestamps = [(t - start_time) for t in timestamps]
            
            axes[1, 0].plot(timestamps, rss_memory, color='red')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('RSS Memory (MB)')
            axes[1, 0].set_title('Memory Usage Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Latency statistics
        if latency_data:
            stats_text = f"""
            Mean: {np.mean(latency_data):.2f} ms
            Std: {np.std(latency_data):.2f} ms
            Min: {np.min(latency_data):.2f} ms
            Max: {np.max(latency_data):.2f} ms
            P95: {np.percentile(latency_data, 95):.2f} ms
            P99: {np.percentile(latency_data, 99):.2f} ms
            """
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Latency Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"dashboard_{int(np.random.random()*10000)}.png")
        plt.close()
    
    def save_comparison_plot(self, 
                           data_dict: Dict[str, List[float]], 
                           title: str = "Performance Comparison",
                           ylabel: str = "Value"):
        """Save comparison plot for multiple metrics"""
        plt.figure(figsize=(12, 6))
        
        for label, values in data_dict.items():
            plt.plot(values, label=label, marker='o', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"comparison_{int(np.random.random()*10000)}.png")
        plt.close()
