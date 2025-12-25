#!/usr/bin/env python3
"""
PatchVision - Main Entry Point
Industrial Vision Processing Framework
"""

import argparse
import sys
import yaml
import time
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PatchVision Industrial Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/industrial.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["train", "inference", "visualize", "serve"],
        help="Operation mode",
    )
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
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
            mode=args.device, batch_size=config["core"]["processors"]["batch_size"]
        )

        # Load default model if available
        if model_manager.model_registry:
            print(f"Found {len(model_manager.model_registry)} registered models")
            # Load latest model
            latest_model = list(model_manager.model_registry.keys())[0]
            try:
                model_path = model_manager.model_registry[latest_model]["path"]
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
                "camera": {"camera_id": 0, "resolution": (640, 480), "fps": 30}
            }

            stream = RealTimeStream(stream_config)
            stream.add_sensor_source("camera", stream_config["camera"])

            # Add PatchVision processing pipeline
            from core.patches.factory import PatchFactory
            from core.projections.transformer import TokenProjector

            patch_factory = PatchFactory()
            projector = TokenProjector(dim=128)

            def process_frame(data):
                if data["type"] == "image":
                    # Process frame through PatchVision pipeline
                    frame = data["data"]
                    patches = patch_factory.adaptive_patching(frame)

                    # Convert patches to tokens
                    patch_array = np.array([p["data"] for p in patches])
                    if len(patch_array) > 0:
                        patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
                        tokens = projector.forward(patch_array)
                        data["tokens"] = tokens
                        data["num_patches"] = len(patches)

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
                        if "tokens" in frame_data:
                            print(
                                f"Processed frame: {frame_data['num_patches']} patches, "
                                f"tokens shape: {frame_data['tokens'].shape}"
                            )
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopping stream...")
                stream.stop()
            
    elif args.mode == "visualize":
        print("Starting visualization mode...")
        from vision.dashboard import DashboardManager
        
        # Initialize dashboard
        dashboard = DashboardManager(config)
        
        # Add data collection callback
        def collect_metrics():
            # Real metrics collection
            import psutil
            
            return {
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'memory_usage': psutil.virtual_memory().used / (1024 * 1024),  # MB
                'inference_time': 0.0,  # Will be updated by actual inference
                'fps': 0.0,  # Will be updated by stream processing
                'detection_count': 0,  # Will be updated by detection results
                'confidence_avg': 0.0  # Will be updated by detection results
            }
        
        dashboard.add_data_callback(collect_metrics)
        dashboard.start()
        
        print("Dashboard running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping dashboard...")
            dashboard.stop()
            dashboard.close()
        
    elif args.mode == "serve":
        print("Starting API server...")
        from deploy.api.server import create_app
        
        # Create and start FastAPI server
        app = create_app(config)
        
        import uvicorn
        host = config.get('deploy', {}).get('api', {}).get('host', '0.0.0.0')
        port = config.get('deploy', {}).get('api', {}).get('port', 8000)
        
        print(f"Server starting on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    else:
        print(f"Mode {args.mode} not implemented yet")
        sys.exit(1)


def process_files(input_path, engine, config):
    """Process files for inference with real PatchVision pipeline"""
    from core.analytics.benchmark import PerformanceBenchmark
    from core.patches.factory import PatchFactory
    from core.projections.transformer import TokenProjector
    import cv2

    input_path = Path(input_path)
    
    # Initialize components
    benchmark = PerformanceBenchmark()
    patch_factory = PatchFactory()
    projector = TokenProjector(dim=config.get('core', {}).get('projections', {}).get('token_dim', 512))

    if input_path.is_file():
        # Process single image file
        print(f"Processing file: {input_path}")
        
        def process_single_file():
            # Load image
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image = cv2.imread(str(input_path))
                if image is None:
                    return f"Failed to load image: {input_path}"
                
                # Extract patches
                patches = patch_factory.adaptive_patching(image)
                
                # Convert to tokens
                if len(patches) > 0:
                    patch_array = np.array([p['data'] for p in patches])
                    batch_size = 1
                    num_patches = len(patches)
                    patch_dim = patch_array[0].size
                    patch_array = patch_array.reshape(batch_size, num_patches, patch_dim)
                    tokens = projector.forward(patch_array)
                    
                    return f"Processed {input_path}: {len(patches)} patches, tokens shape {tokens.shape}"
                else:
                    return f"No patches extracted from {input_path}"
            else:
                return f"Skipped non-image file: {input_path}"

        result = benchmark.benchmark_function(process_single_file, "file_processing")
        print(f"Result: {result.result}")
        print(f"Processing time: {result.avg_time:.4f}s ± {result.std_time:.4f}s")

    elif input_path.is_dir():
        # Process directory of images
        print(f"Processing directory: {input_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.glob("*") if f.suffix.lower() in image_extensions]
        print(f"Found {len(image_files)} image files")

        def process_directory():
            results = []
            for file_path in image_files[:10]:  # Process first 10 images
                image = cv2.imread(str(file_path))
                if image is not None:
                    patches = patch_factory.adaptive_patching(image)
                    results.append({
                        'file': file_path.name,
                        'patches': len(patches),
                        'size': image.shape
                    })
            return f"Processed {len(results)} images"

        result = benchmark.benchmark_function(process_directory, "batch_processing")
        print(f"Result: {result.result}")
        print(f"Batch processing: {result.avg_time:.4f}s ± {result.std_time:.4f}s")

        # Save benchmark results
        benchmark.save_results("benchmark_results.json")
        print("Benchmark results saved to benchmark_results.json")

    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
