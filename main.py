#!/usr/bin/env python3
"""
PatchVision - Main Entry Point
Industrial Vision Processing Framework
"""

import argparse
import sys
import yaml
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="PatchVision Industrial Framework")
    parser.add_argument("--config", type=str, default="configs/industrial.yaml",
                       help="Configuration file path")
    parser.add_argument("--mode", type=str, default="inference",
                       choices=["train", "inference", "visualize", "serve"],
                       help="Operation mode")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"PatchVision v1.0.0")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    
    if args.mode == "inference":
        from core.processors.engine import InferenceEngine
        
        print("Starting inference mode...")
        
        # Initialize error recovery system
        from core.utils.error_recovery import create_default_recovery_manager
        recovery_manager = create_default_recovery_manager()
        
        # Initialize model manager
        from core.models.model_manager import ModelManager
        model_manager = ModelManager()
        
        # Initialize inference engine
        engine = InferenceEngine(
            mode=args.device,
            batch_size=config['core']['processors']['batch_size']
        )
        
        # Load default model if available
        if model_manager.model_registry:
            print(f"Found {len(model_manager.model_registry)} registered models")
            # Load latest model
            latest_model = list(model_manager.model_registry.keys())[0]
            try:
                model_path = model_manager.model_registry[latest_model]['path']
                print(f"Loading model: {latest_model}")
                # Model loading would be integrated here
                print(f"Model loaded from: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model {latest_model}: {e}")
        else:
            print("No registered models found, using default inference engine")
        
        # Process based on input type
        if args.input:
            # Process file or directory
            process_files(args.input, engine, config)
        else:
            print("Starting real-time stream...")
            from data.realtime_stream import RealTimeStream
            
            # Initialize real-time stream with camera source
            stream_config = {
                'camera': {
                    'camera_id': 0,
                    'resolution': (640, 480),
                    'fps': 30
                }
            }
            
            stream = RealTimeStream(stream_config)
            stream.add_sensor_source('camera', stream_config['camera'])
            
            # Add PatchVision processing pipeline
            from core.patches.factory import PatchFactory
            from core.projections.transfomer import TokenProjector
            
            patch_factory = PatchFactory()
            projector = TokenProjector(dim=128)
            
            def process_frame(data):
                if data['type'] == 'image':
                    # Process frame through PatchVision pipeline
                    frame = data['data']
                    patches = patch_factory.adaptive_patching(frame)
                    
                    # Convert patches to tokens
                    patch_array = np.array([p['data'] for p in patches])
                    if len(patch_array) > 0:
                        patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
                        tokens = projector.forward(patch_array)
                        data['tokens'] = tokens
                        data['num_patches'] = len(patches)
                    
                return data
            
            stream.add_processor(process_frame)
            
            # Start streaming
            stream.start()
            print("Real-time stream started. Press Ctrl+C to stop.")
            
            try:
                while True:
                    # Get latest frames
                    frames = stream.get_latest(num_samples=1)
                    for frame_data in frames:
                        if 'tokens' in frame_data:
                            print(f"Processed frame: {frame_data['num_patches']} patches, "
                                  f"tokens shape: {frame_data['tokens'].shape}")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopping stream...")
                stream.stop()
            
    elif args.mode == "visualize":
        print("Visualization mode not implemented yet")
        # TODO: Implement visualization
        
    elif args.mode == "serve":
        print("API server mode not implemented yet")
        # TODO: Implement API server
    
    else:
        print(f"Mode {args.mode} not implemented yet")
        sys.exit(1)

def process_files(input_path, engine, config):
    """Process files for inference with benchmarking"""
    input_path = Path(input_path)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    if input_path.is_file():
        # Process single file
        print(f"Processing file: {input_path}")
        
        # Benchmark file processing
        def process_single_file():
            # Add file processing logic here
            return f"Processed {input_path}"
        
        result = benchmark.benchmark_function(process_single_file, "file_processing")
        print(f"File processing: {result.avg_time:.4f}s ± {result.std_time:.4f}s")
        
    elif input_path.is_dir():
        # Process directory
        print(f"Processing directory: {input_path}")
        files = list(input_path.glob("*"))
        print(f"Found {len(files)} files")
        
        # Benchmark batch processing
        def process_directory():
            # Add batch processing logic here
            return f"Processed {len(files)} files"
        
        result = benchmark.benchmark_function(process_directory, "batch_processing")
        print(f"Batch processing: {result.avg_time:.4f}s ± {result.std_time:.4f}s")
        
        # Save benchmark results
        benchmark.save_results("benchmark_results.json")
        print("Benchmark results saved to benchmark_results.json")
        
    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()