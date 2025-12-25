"""
Structured logging system for PatchVision
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sys


class StructuredLogger:
    """
    Structured logging with JSON output and performance tracking
    """
    
    def __init__(self, 
                 name: str = "patchvision",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with simple format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with JSON format if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
        
        self.performance_data = []
    
    def log_inference(self, 
                     model_name: str,
                     input_shape: tuple,
                     latency_ms: float,
                     batch_size: int = 1,
                     metadata: Optional[Dict] = None):
        """Log inference performance"""
        log_data = {
            'event': 'inference',
            'model': model_name,
            'input_shape': input_shape,
            'latency_ms': latency_ms,
            'batch_size': batch_size,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_data))
        self.performance_data.append(log_data)
    
    def log_batch_processing(self,
                           operation: str,
                           batch_size: int,
                           processing_time: float,
                           throughput: float,
                           metadata: Optional[Dict] = None):
        """Log batch processing performance"""
        log_data = {
            'event': 'batch_processing',
            'operation': operation,
            'batch_size': batch_size,
            'processing_time': processing_time,
            'throughput': throughput,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_data))
        self.performance_data.append(log_data)
    
    def log_model_load(self,
                      model_name: str,
                      model_size_mb: float,
                      load_time: float,
                      framework: str,
                      metadata: Optional[Dict] = None):
        """Log model loading performance"""
        log_data = {
            'event': 'model_load',
            'model': model_name,
            'model_size_mb': model_size_mb,
            'load_time': load_time,
            'framework': framework,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_error(self,
                 error_type: str,
                 error_message: str,
                 context: Optional[Dict] = None):
        """Log error with context"""
        log_data = {
            'event': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': time.time(),
            'context': context or {}
        }
        
        self.logger.error(json.dumps(log_data))
    
    def log_debug(self,
                 message: str,
                 data: Optional[Dict] = None):
        """Log debug information"""
        log_data = {
            'event': 'debug',
            'message': message,
            'timestamp': time.time(),
            'data': data or {}
        }
        
        self.logger.debug(json.dumps(log_data))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from logged data"""
        if not self.performance_data:
            return {}
        
        inference_data = [d for d in self.performance_data if d['event'] == 'inference']
        batch_data = [d for d in self.performance_data if d['event'] == 'batch_processing']
        
        summary = {
            'total_logs': len(self.performance_data),
            'inference_count': len(inference_data),
            'batch_count': len(batch_data)
        }
        
        if inference_data:
            latencies = [d['latency_ms'] for d in inference_data]
            summary['avg_latency_ms'] = sum(latencies) / len(latencies)
            summary['min_latency_ms'] = min(latencies)
            summary['max_latency_ms'] = max(latencies)
        
        if batch_data:
            throughputs = [d['throughput'] for d in batch_data]
            summary['avg_throughput'] = sum(throughputs) / len(throughputs)
            summary['max_throughput'] = max(throughputs)
        
        return summary
    
    def save_performance_data(self, file_path: str):
        """Save performance data to file"""
        with open(file_path, 'w') as f:
            json.dump(self.performance_data, f, indent=2)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


def setup_logger(name: str = "patchvision", log_level: str = "INFO") -> StructuredLogger:
    """Setup default logger for PatchVision"""
    log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.json"
    Path("logs").mkdir(exist_ok=True)
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    return StructuredLogger(name, log_file, level_map.get(log_level, logging.INFO))
