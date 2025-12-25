import numpy as np
from typing import Dict, List, Optional, Callable
import threading
import time
import queue
import cv2
from datetime import datetime
import json

class RealTimeStream:
    """
    Real-time sensor data ingestion and processing
    """
    
    def __init__(self, 
                 source_config: Dict,
                 buffer_size: int = 1000):
        self.source_config = source_config
        self.buffer_size = buffer_size
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.processors = []
        self.listeners = []
        self.sources = []  # Initialize sources list
        
    def add_sensor_source(self, 
                         sensor_type: str,
                         config: Dict):
        """Add sensor source to stream"""
        if sensor_type == 'camera':
            source = CameraSource(config)
        elif sensor_type == 'lidar':
            source = LidarSource(config)
        elif sensor_type == 'thermal':
            source = ThermalSource(config)
        elif sensor_type == 'imu':
            source = IMUSource(config)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
            
        self.sources.append(source)
        
    def add_processor(self, processor: Callable):
        """Add data processor"""
        self.processors.append(processor)
        
    def add_listener(self, listener: Callable):
        """Add data listener (callback)"""
        self.listeners.append(listener)
        
    def start(self):
        """Start streaming"""
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop)
        self.thread.start()
        
    def stop(self):
        """Stop streaming"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def get_latest(self, 
                  num_samples: int = 1,
                  block: bool = True) -> List[Dict]:
        """
        Get latest samples from buffer
        """
        samples = []
        
        for _ in range(num_samples):
            try:
                sample = self.data_buffer.get(block=block)
                samples.append(sample)
            except queue.Empty:
                break
                
        return samples
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.running:
            # Read from all sources
            for source in self.sources:
                try:
                    data = source.read()
                    
                    if data is not None:
                        # Process data
                        for processor in self.processors:
                            data = processor(data)
                            
                        # Add metadata
                        data['timestamp'] = datetime.now().isoformat()
                        data['source'] = source.sensor_type
                        
                        # Put in buffer
                        try:
                            self.data_buffer.put_nowait(data)
                        except queue.Full:
                            # Remove oldest if buffer full
                            try:
                                self.data_buffer.get_nowait()
                                self.data_buffer.put_nowait(data)
                            except queue.Empty:
                                pass
                                
                        # Notify listeners
                        for listener in self.listeners:
                            listener(data)
                            
                except Exception as e:
                    print(f"Error reading from source {source.sensor_type}: {e}")
                    
            time.sleep(0.001)  # Small delay to prevent CPU spinning
            
    def save_stream(self, 
                   filename: str,
                   duration: float = 10.0):
        """
        Save stream to file for specified duration
        """
        start_time = time.time()
        data_log = []
        
        def logger(data):
            data_log.append(data)
            
        self.add_listener(logger)
        
        time.sleep(duration)
        
        self.listeners.remove(logger)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data_log, f, indent=2)
            
        return len(data_log)

class SensorIngestor:
    """
    Unified sensor data ingestor
    """
    
    def __init__(self, sync_tolerance: float = 0.01):
        self.sync_tolerance = sync_tolerance
        self.sensor_buffers = {}
        self.sync_queue = queue.Queue()
        
    def ingest(self, 
              sensor_id: str,
              data: Dict,
              timestamp: float):
        """
        Ingest sensor data with timestamp
        """
        if sensor_id not in self.sensor_buffers:
            self.sensor_buffers[sensor_id] = []
            
        self.sensor_buffers[sensor_id].append({
            'data': data,
            'timestamp': timestamp
        })
        
        # Try to synchronize
        self._try_synchronize()
        
    def get_synced_data(self, 
                       timeout: float = 1.0) -> Optional[Dict]:
        """
        Get synchronized multi-sensor data
        """
        try:
            return self.sync_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _try_synchronize(self):
        """Try to synchronize data from all sensors"""
        if len(self.sensor_buffers) < 2:
            return
            
        # Find common timeframe
        all_timestamps = []
        for sensor_id, buffer in self.sensor_buffers.items():
            if buffer:
                all_timestamps.append(buffer[-1]['timestamp'])
                
        if len(all_timestamps) == len(self.sensor_buffers):
            max_time = max(all_timestamps)
            min_time = min(all_timestamps)
            
            if max_time - min_time <= self.sync_tolerance:
                # We have synchronized data
                synced_data = {}
                
                for sensor_id, buffer in self.sensor_buffers.items():
                    # Find closest timestamp
                    closest = min(buffer, 
                                 key=lambda x: abs(x['timestamp'] - max_time))
                    synced_data[sensor_id] = closest['data']
                    
                # Add synchronization timestamp
                synced_data['sync_timestamp'] = max_time
                
                # Put in queue
                self.sync_queue.put(synced_data)
                
                # Clear old data
                self._clean_buffers(max_time)

    def _clean_buffers(self, current_time: float):
        """Clean old data from buffers"""
        cutoff_time = current_time - 1.0  # Keep 1 second of history
        
        for sensor_id in self.sensor_buffers:
            self.sensor_buffers[sensor_id] = [
                item for item in self.sensor_buffers[sensor_id]
                if item['timestamp'] > cutoff_time
            ]

# Sensor source implementations
class CameraSource:
    def __init__(self, config: Dict):
        self.config = config
        self.sensor_type = 'camera'  # Add sensor type identifier
        self.camera_id = config.get('camera_id', 0)
        self.resolution = config.get('resolution', (640, 480))
        self.fps = config.get('fps', 30)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
    def read(self) -> Optional[Dict]:
        ret, frame = self.cap.read()
        if ret:
            return {
                'type': 'image',
                'data': frame,
                'shape': frame.shape
            }
        return None

class LidarSource:
    """
    Example LiDAR sensor source - requires hardware-specific driver
    This is a plugin example. For production, integrate with your LiDAR SDK.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.sensor_type = 'lidar'
        # TODO: Initialize actual LiDAR connection
        # Example: self.lidar = YourLidarSDK.connect(config)
        
    def read(self) -> Optional[Dict]:
        # Simulate LiDAR data
        num_points = 10000
        points = np.random.randn(num_points, 3) * 10
        intensity = np.random.rand(num_points)
        
        return {
            'type': 'point_cloud',
            'points': points,
            'intensity': intensity
        }

class ThermalSource:
    """
    Example Thermal camera source - requires hardware-specific driver
    This is a plugin example. For production, integrate with your thermal camera SDK.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.sensor_type = 'thermal'
        # TODO: Initialize actual thermal camera connection
        # Example: self.thermal = YourThermalSDK.connect(config)
        
    def read(self) -> Optional[Dict]:
        # Simulate thermal data
        resolution = self.config.get('resolution', (320, 240))
        thermal = np.random.rand(*resolution) * 100 + 20  # 20-120°C
        
        return {
            'type': 'thermal',
            'data': thermal,
            'unit': 'celsius'
        }

class IMUSource:
    """
    Example IMU sensor source - requires hardware-specific driver
    This is a plugin example. For production, integrate with your IMU SDK.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.sensor_type = 'imu'
        # TODO: Initialize actual IMU connection
        # Example: self.imu = YourIMUSDK.connect(config)
        
    def read(self) -> Optional[Dict]:
        # Simulate IMU data
        return {
            'type': 'imu',
            'acceleration': np.random.randn(3).tolist(),
            'gyroscope': np.random.randn(3).tolist(),
            'magnetometer': np.random.randn(3).tolist()
        }