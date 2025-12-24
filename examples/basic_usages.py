#!/usr/bin/env python3
"""
Basic usage examples for PatchVision
"""

import numpy as np
import cv2
import yaml
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def example_patch_processing():
    """Example of patch processing"""
    from core.patches.factory import PatchFactory
    from core.projections.transformer import TokenProjector
    from core.processors.engine import InferenceEngine
    
    print("=== Example 1: Patch Processing ===")
    
    # Create synthetic image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize components
    patch_factory = PatchFactory()
    projector = TokenProjector(dim=512)
    engine = InferenceEngine(batch_size=8)
    
    # Generate patches
    patches = patch_factory.adaptive_patching(image)
    print(f"Generated {len(patches)} patches")
    
    # Convert patches to tokens
    patch_data = np.array([p['data'].flatten() for p in patches])
    patch_data = patch_data.reshape(len(patches), 16, 16, 3)
    patch_data = patch_data.reshape(len(patches), -1)
    
    # Project tokens
    tokens = patch_data.reshape(1, len(patches), -1)
    projected = projector.forward(tokens)
    print(f"Projected tokens shape: {projected.shape}")
    
    # Simulate inference
    def dummy_model(x):
        return x * 0.5 + 0.5
    
    result = engine.process(projected.numpy() if hasattr(projected, 'numpy') else projected, 
                           dummy_model)
    print(f"Inference result shape: {result.shape}")
    
    return patches, result

def example_synthetic_data():
    """Example of synthetic data generation"""
    from data.synthetic_factory import SyntheticDataFactory
    
    print("\n=== Example 2: Synthetic Data Generation ===")
    
    factory = SyntheticDataFactory()
    
    # Generate defect dataset
    dataset = factory.generate_defect_dataset(num_samples=5)
    
    print(f"Generated dataset with {len(dataset['images'])} samples")
    print(f"Sample labels: {dataset['labels'][0]}")
    
    # Display first sample
    if len(dataset['images']) > 0:
        cv2.imwrite('example_defect.png', dataset['images'][0])
        print("Saved example image to 'example_defect.png'")
    
    return dataset

def example_visualization():
    """Example of 3D visualization"""
    from vision.holographic import HolographicVisualizer
    
    print("\n=== Example 3: 3D Visualization ===")
    
    # Create point cloud
    points = np.random.randn(1000, 3)
    values = np.random.rand(1000)
    
    # Create visualizer
    visualizer = HolographicVisualizer(theme='industrial')
    
    # Create 3D plot
    fig = visualizer.create_3d_point_cloud(points, values)
    
    # Save to HTML
    fig.write_html('example_point_cloud.html')
    print("Saved 3D visualization to 'example_point_cloud.html'")
    
    return fig

def example_realtime_stream():
    """Example of real-time streaming"""
    from data.realtime_stream import RealTimeStream, CameraSource
    
    print("\n=== Example 4: Real-time Streaming ===")
    
    # Create stream
    stream = RealTimeStream(
        source_config={'camera_id': 0, 'resolution': (640, 480)}
    )
    
    # Add camera source
    camera = CameraSource({'camera_id': 0})
    
    # Capture one frame
    try:
        data = camera.read()
        if data:
            print(f"Captured image with shape: {data['shape']}")
            cv2.imwrite('example_capture.png', data['data'])
            print("Saved captured image to 'example_capture.png'")
    except Exception as e:
        print(f"Camera not available: {e}")
        print("Using synthetic data instead...")
        # Create synthetic image
        synthetic = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite('example_capture.png', synthetic)
    
    return stream

def example_dashboard():
    """Example of real-time dashboard"""
    from vision.dashboard import RealTimeDashboard
    
    print("\n=== Example 5: Real-time Dashboard ===")
    
    # Create dashboard
    dashboard = RealTimeDashboard(
        title="PatchVision Demo",
        update_interval=2000
    )
    
    # Start dashboard in background
    dashboard.start_server(host="127.0.0.1", port=8050)
    
    print("Dashboard started at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop")
    
    return dashboard

def main():
    """Run all examples"""
    print("PatchVision Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_patch_processing()
        example_synthetic_data()
        example_visualization()
        example_realtime_stream()
        
        # Uncomment to run dashboard (blocks)
        # example_dashboard()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("1. example_defect.png - Synthetic defect image")
        print("2. example_point_cloud.html - 3D visualization")
        print("3. example_capture.png - Camera capture")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()