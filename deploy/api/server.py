from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import asyncio
import json
import numpy as np
import cv2
import base64
from datetime import datetime
import threading


class APIServer:
    """
    Production-ready API server for PatchVision
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 8000, api_key: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.api_key = api_key

        # Initialize FastAPI app
        self.app = FastAPI(
            title="PatchVision API",
            description="Industrial Vision Processing API",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request models
        class ProcessRequest(BaseModel):
            image: str  # base64 encoded
            task: str
            parameters: Optional[Dict] = None

        class BatchRequest(BaseModel):
            images: List[str]
            task: str
            parameters: Optional[Dict] = None

        # API endpoints
        @self.app.get("/")
        async def root():
            return {"message": "PatchVision API", "status": "running"}

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.post("/process")
        async def process_image(request: ProcessRequest):
            """Process single image"""
            try:
                # Decode image
                image_data = base64.b64decode(request.image)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Process based on task
                result = await self._process_task(
                    image, request.task, request.parameters
                )

                return {
                    "success": True,
                    "task": request.task,
                    "result": result,
                    "processing_time": 0.1,  # Placeholder
                }

            except Exception as e:
                return {"success": False, "error": str(e)}

        @self.app.post("/batch_process")
        async def batch_process(request: BatchRequest):
            """Process batch of images"""
            try:
                images = []
                for img_str in request.images:
                    image_data = base64.b64decode(img_str)
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(image)

                # Process batch
                results = []
                for image in images:
                    result = await self._process_task(
                        image, request.task, request.parameters
                    )
                    results.append(result)

                return {
                    "success": True,
                    "task": request.task,
                    "results": results,
                    "count": len(results),
                }

            except Exception as e:
                return {"success": False, "error": str(e)}

        @self.app.get("/models")
        async def list_models():
            """List available models"""
            return {
                "models": [
                    "defect_detection",
                    "quality_inspection",
                    "assembly_verification",
                ],
                "status": "available",
            }

        # WebSocket endpoint
        self.websocket_server = WebSocketServer()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_server.handle_connection(websocket)

    async def _process_task(
        self, image: np.ndarray, task: str, parameters: Optional[Dict]
    ) -> Dict:
        """Process image based on task using real inference pipeline"""
        try:
            # Import core components
            from core.patches.factory import PatchFactory
            from core.projections.transformer import TokenProjector
            from core.processors.engine import InferenceEngine
            
            # Initialize components
            patch_factory = PatchFactory()
            projector = TokenProjector(dim=512)
            engine = InferenceEngine(mode='auto', batch_size=1)
            
            # Extract patches
            patches = patch_factory.adaptive_patching(image)
            
            # Convert patches to tokens
            if len(patches) > 0:
                patch_array = np.array([p['data'] for p in patches])
                # Flatten patches for projection
                batch_size = 1
                num_patches = len(patches)
                patch_dim = patch_array[0].size
                patch_array = patch_array.reshape(batch_size, num_patches, patch_dim)
                
                # Project to tokens
                tokens = projector.forward(patch_array)
                
                # Process based on task
                if task == "defect_detection":
                    return self._process_defect_detection(image, patches, tokens)
                elif task == "quality_inspection":
                    return self._process_quality_inspection(image, patches, tokens)
                elif task == "assembly_verification":
                    return self._process_assembly_verification(image, patches, tokens)
                else:
                    raise ValueError(f"Unknown task: {task}")
            else:
                return {"error": "No patches extracted from image"}
                
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

    @staticmethod
    def _process_defect_detection(image: np.ndarray, patches: List[Dict], tokens: np.ndarray) -> Dict:
        """Real defect detection using patch analysis"""
        h, w = image.shape[:2]
        defects = []
        
        # Analyze patches for defects based on entropy and contrast
        for patch_info in patches:
            metadata = patch_info.get('metadata', {})
            contrast = metadata.get('contrast', 0)
            entropy = metadata.get('entropy', 0)
            importance = patch_info.get('importance', 0)
            
            # High importance + low contrast often indicates defects
            if importance > 0.7 and contrast < 0.3:
                x, y, w_patch, h_patch = patch_info['coordinates']
                defects.append({
                    'bbox': [int(x), int(y), int(x + w_patch), int(y + h_patch)],
                    'confidence': float(importance),
                    'type': 'anomaly',
                    'metrics': {
                        'contrast': float(contrast),
                        'entropy': float(entropy)
                    }
                })
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'image_size': [w, h],
            'patches_analyzed': len(patches),
            'tokens_shape': list(tokens.shape)
        }

    @staticmethod
    def _process_quality_inspection(image: np.ndarray, patches: List[Dict], tokens: np.ndarray) -> Dict:
        """Real quality inspection using patch statistics"""
        # Calculate quality metrics from patches
        contrasts = [p.get('metadata', {}).get('contrast', 0) for p in patches]
        entropies = [p.get('metadata', {}).get('entropy', 0) for p in patches]
        importances = [p.get('importance', 0) for p in patches]
        
        avg_contrast = np.mean(contrasts) if contrasts else 0
        avg_entropy = np.mean(entropies) if entropies else 0
        avg_importance = np.mean(importances) if importances else 0
        
        # Quality score based on uniformity and detail
        quality_score = (avg_contrast * 0.4 + avg_entropy * 0.3 + (1 - avg_importance) * 0.3)
        
        # Pass if quality score is above threshold
        passed = quality_score > 0.5
        
        return {
            'quality_score': float(quality_score),
            'pass': bool(passed),
            'metrics': {
                'avg_contrast': float(avg_contrast),
                'avg_entropy': float(avg_entropy),
                'uniformity': float(1 - avg_importance)
            },
            'patches_analyzed': len(patches)
        }

    @staticmethod
    def _process_assembly_verification(image: np.ndarray, patches: List[Dict], tokens: np.ndarray) -> Dict:
        """Real assembly verification using spatial analysis"""
        # Analyze spatial distribution of high-importance patches
        high_importance_patches = [p for p in patches if p.get('importance', 0) > 0.6]
        
        # Calculate coverage
        total_area = image.shape[0] * image.shape[1]
        covered_area = sum([p['coordinates'][2] * p['coordinates'][3] for p in high_importance_patches])
        coverage = covered_area / total_area if total_area > 0 else 0
        
        # Assembly is complete if coverage is high and uniform
        assembly_complete = coverage > 0.3 and len(high_importance_patches) > 5
        
        return {
            'assembly_complete': bool(assembly_complete),
            'coverage': float(coverage),
            'components_detected': len(high_importance_patches),
            'verification_score': float(min(coverage * 2, 1.0)),
            'patches_analyzed': len(patches)
        }

    def start(self):
        """Start API server"""
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)

        # Run in separate thread
        thread = threading.Thread(target=server.run)
        thread.daemon = True
        thread.start()

        print(f"API server running at http://{self.host}:{self.port}")
        return thread


class WebSocketServer:
    """
    WebSocket server for real-time communication
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                # Process message
                response = await self._process_message(message)

                # Send response
                await websocket.send_text(json.dumps(response))

        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.active_connections.remove(websocket)

    async def _process_message(self, message: Dict) -> Dict:
        """Process WebSocket message"""
        message_type = message.get("type")

        if message_type == "ping":
            return {"type": "pong", "timestamp": datetime.now().isoformat()}
        elif message_type == "process":
            # Process image data in real-time
            try:
                image_data = message.get("image")
                task = message.get("task", "defect_detection")

                # Decode image
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Import processing components
                from core.patches.factory import PatchFactory
                from core.projections.transformer import TokenProjector
                
                # Process image
                patch_factory = PatchFactory()
                patches = patch_factory.adaptive_patching(image)
                
                result = {
                    "task": task,
                    "patches_extracted": len(patches),
                    "timestamp": datetime.now().isoformat(),
                    "status": "processed"
                }

                return {"type": "result", "data": result}
            except Exception as e:
                return {"type": "error", "message": f"Processing failed: {str(e)}"}
        else:
            return {"type": "error", "message": "Unknown message type"}

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove disconnected client
                self.active_connections.remove(connection)


def create_app(config: Dict):
    """Create FastAPI application from config"""
    api_config = config.get("deploy", {}).get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)

    server = APIServer(host=host, port=port)
    return server.app
