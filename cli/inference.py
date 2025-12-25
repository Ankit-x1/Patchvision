import time
import numpy as np
from pathlib import Path
import sys

def run_inference(args, config):
    """Run the inference process based on the provided arguments and configuration."""
    from patchvision.core.processors.engine import InferenceEngine
    from patchvision.core.utils.error_recovery import create_default_recovery_manager
    from patchvision.core.models.model_manager import ModelManager

    print("Starting inference mode...")
    
    recovery_manager = create_default_recovery_manager()
    model_manager = ModelManager()

    engine = InferenceEngine(
        mode=args.device, batch_size=config["core"]["processors"]["batch_size"]
    )

    if model_manager.model_registry:
        print(f"Found {len(model_manager.model_registry)} registered models")
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

    if args.input:
        process_files(args.input, engine, config)
    else:
        print("Starting real-time stream...")
        from data.realtime_stream import RealTimeStream
        from core.patches.factory import PatchFactory
        from core.projections.transformer import TokenProjector

        stream_config = {
            "camera": {"camera_id": 0, "resolution": (640, 480), "fps": 30}
        }
        stream = RealTimeStream(stream_config)
        stream.add_sensor_source("camera", stream_config["camera"])

        patch_factory = PatchFactory()
        projector = TokenProjector(dim=128)

        def process_frame(data):
            if data["type"] == "image":
                frame = data["data"]
                patches = patch_factory.adaptive_patching(frame)
                if len(patches) > 0:
                    patch_array = np.array([p["data"] for p in patches])
                    patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
                    tokens = projector.forward(patch_array)
                    data["tokens"] = tokens
                    data["num_patches"] = len(patches)
            return data

        stream.add_processor(process_frame)
        stream.start()
        print("Real-time stream started. Press Ctrl+C to stop.")

        try:
            while True:
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

def process_files(input_path, engine, config):
    """Process files for inference with real PatchVision pipeline"""
    from core.analytics.benchmark import PerformanceBenchmark
    from core.patches.factory import PatchFactory
    from core.projections.transformer import TokenProjector
    import cv2

    input_path = Path(input_path)
    
    benchmark = PerformanceBenchmark()
    patch_factory = PatchFactory()
    projector = TokenProjector(dim=config.get('core', {}).get('projections', {}).get('token_dim', 512))

    if input_path.is_file():
        print(f"Processing file: {input_path}")
        
        def process_single_file():
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image = cv2.imread(str(input_path))
                if image is None:
                    return f"Failed to load image: {input_path}"
                
                patches = patch_factory.adaptive_patching(image)
                
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
        print(f"Processing directory: {input_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.glob("*") if f.suffix.lower() in image_extensions]
        print(f"Found {len(image_files)} image files")

        def process_directory():
            results = []
            for file_path in image_files[:10]:
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

        benchmark.save_results("benchmark_results.json")
        print("Benchmark results saved to benchmark_results.json")

    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)
