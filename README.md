# PatchVision 
**Industrial-Grade Vision Processing Framework**

*Production-ready computer vision for industrial applications*

## Features

### Core Technology
- **Adaptive Patching**: Content-aware patch extraction with entropy-based importance scoring
- **Transformer Projections**: Multi-head attention with positional embeddings
- **Hardware Optimization**: Automatic backend selection (CPU/CUDA/MPS)
- **Real-time Processing**: Camera-based streaming with low latency

### Industrial Applications
- **Defect Detection**: Automated surface inspection using patch analysis
- **Quality Control**: Statistical quality metrics from image patches
- **Assembly Verification**: Spatial coverage analysis for component detection

## Installation

### Quick Install
```bash
pip install -e .
```

### With GPU Support
```bash
pip install -e .[gpu]
```

### Full Installation (with visualization)
```bash
pip install -e .[full]
```

## Quick Start

### 1. Process a Single Image
```python
from core.patches.factory import PatchFactory
from core.projections.transformer import TokenProjector
import cv2

# Load image
image = cv2.imread("test.jpg")

# Extract patches
factory = PatchFactory()
patches = factory.adaptive_patching(image)

# Project to tokens
projector = TokenProjector(dim=512)
tokens = projector.forward(patches)

print(f"Extracted {len(patches)} patches")
```

### 2. Run Inference via CLI
```bash
# Process single image
python main.py --mode inference --input test.jpg

# Process directory of images
python main.py --mode inference --input ./images/

# Start visualization dashboard
python main.py --mode visualize

# Start API server
python main.py --mode serve
```

### 3. Use REST API
```bash
# Start server
python main.py --mode serve

# Process image (from another terminal)
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "task": "defect_detection"
  }'
```

## Configuration

### Industrial Configuration
Create `configs/industrial.yaml`:
```yaml
core:
  patches:
    default_size: 16
    default_stride: 8
    strategies: ["adaptive", "hierarchical", "temporal"]
    
  projections:
    token_dim: 512
    num_heads: 8
    attention_type: "sparse"
    
  processors:
    default_backend: "auto"
    use_quantization: true
    batch_size: 32
```

### Medical Configuration (High Precision)
Create `configs/medical.yaml`:
```yaml
core:
  patches:
    default_size: 8  # Smaller for precision
    default_stride: 4
    
  projections:
    token_dim: 1024  # Higher dimension
    num_heads: 16
    attention_type: "full"
    
  processors:
    use_quantization: false  # Disable for precision
```

## API Reference

### Python API

#### Patch Extraction
```python
from core.patches.factory import PatchFactory

factory = PatchFactory()
patches = factory.adaptive_patching(image)
# Returns: List[Dict] with patch data, coordinates, importance, metadata
```

#### Token Projection
```python
from core.projections.transformer import TokenProjector

projector = TokenProjector(dim=512, num_heads=8)
tokens = projector.forward(patch_array)
# Returns: np.ndarray of shape (batch, num_patches, dim)
```

#### Inference Engine
```python
from core.processors.engine import InferenceEngine

engine = InferenceEngine(mode='auto', batch_size=32)
results = engine.process(inputs, model)
# Automatically selects best hardware backend
```

### REST API

#### Endpoints

**GET /health**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-24T23:00:00"
}
```

**POST /process**
```json
{
  "image": "base64_encoded_image",
  "task": "defect_detection",
  "parameters": {}
}
```

Response:
```json
{
  "success": true,
  "task": "defect_detection",
  "result": {
    "defects_found": 2,
    "defects": [...],
    "patches_analyzed": 156
  }
}
```

**GET /models**
```json
{
  "models": [
    "defect_detection",
    "quality_inspection",
    "assembly_verification"
  ]
}
```

## Real-time Streaming

### Camera-based Processing
```python
from data.realtime_stream import RealTimeStream

# Configure camera
config = {
    "camera": {
        "camera_id": 0,
        "resolution": (640, 480),
        "fps": 30
    }
}

# Start stream
stream = RealTimeStream(config)
stream.add_sensor_source("camera", config["camera"])
stream.start()

# Get frames
frames = stream.get_latest(num_samples=1)
```

### Custom Sensor Plugins
For LiDAR, Thermal, or IMU sensors, implement the sensor interface:
```python
class CustomSensor:
    def __init__(self, config: Dict):
        self.sensor_type = 'custom'
        # Initialize your sensor SDK here
        
    def read(self) -> Optional[Dict]:
        # Return sensor data
        return {
            'type': 'custom_data',
            'data': your_sensor_data
        }
```

## Architecture

### Core Components
- `core/patches/` - Adaptive patch extraction
- `core/projections/` - Transformer-based token projections
- `core/processors/` - Hardware-optimized inference engine
- `core/models/` - Model management and versioning
- `data/` - Real-time streaming and data augmentation
- `deploy/` - API server and edge deployment
- `vision/` - Visualization and monitoring

### Processing Pipeline
```
Image → Adaptive Patching → Token Projection → Inference → Results
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
black .
isort .
```

## Production Deployment

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["python", "main.py", "--mode", "serve"]
```

### Build and Run
```bash
docker build -t patchvision .
docker run -p 8000:8000 patchvision
```

## Hardware Requirements

### Minimum
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 2GB

### Recommended
- CPU: Intel i7 or AMD Ryzen 7
- GPU: NVIDIA RTX 2060+ (for CUDA acceleration)
- RAM: 16GB
- Storage: 5GB

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/Ankit-x1/Patchvision/issues)
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

---

**PatchVision** - Production-Ready Industrial Vision Processing  
*Built for real-world deployment*