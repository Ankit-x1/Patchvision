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
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="PatchVision API",
            description="Industrial Vision Processing API",
            version="1.0.0"
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
                result = await self._process_task(image, request.task, request.parameters)
                
                return {
                    "success": True,
                    "task": request.task,
                    "result": result,
                    "processing_time": 0.1  # Placeholder
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
                    result = await self._process_task(image, request.task, request.parameters)
                    results.append(result)
                
                return {
                    "success": True,
                    "task": request.task,
                    "results": results,
                    "count": len(results)
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/models")
        async def list_models():
            """List available models"""
            return {
                "models": ["defect_detection", "quality_inspection", "assembly_verification"],
                "status": "available"
            }
        
        # WebSocket endpoint
        self.websocket_server = WebSocketServer()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_server.handle_connection(websocket)
    
    async def _process_task(self, 
                          image: np.ndarray, 
                          task: str, 
                          parameters: Optional[Dict]) -> Dict:
        """Process image based on task"""
        # This is a placeholder - implement actual processing
        if task == "defect_detection":
            return self._simulate_defect_detection(image)
        elif task == "quality_inspection":
            return self._simulate_quality_inspection(image)
        elif task == "assembly_verification":
            return self._simulate_assembly_verification(image)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    @staticmethod
    def _simulate_defect_detection(image: np.ndarray) -> Dict:
        """Simulate defect detection"""
        h, w = image.shape[:2]
        
        # Simulate defect detection
        defects = []
        for _ in range(np.random.randint(0, 5)):
            x = np.random.randint(0, w-50)
            y = np.random.randint(0, h-50)
            width = np.random.randint(10, 50)
            height = np.random.randint(10, 50)
            
            defects.append({
                "bbox": [x, y, x+width, y+height],
                "confidence": np.random.random(),
                "class": np.random.choice(["crack", "scratch", "dent"])
            })
        
        return {
            "defects_found": len(defects),
            "defects": defects,
            "image_size": [w, h]
        }
    
    @staticmethod
    def _simulate_quality_inspection(image: np.ndarray) -> Dict:
        """Simulate quality inspection"""
        return {
            "quality_score": np.random.random(),
            "pass": np.random.random() > 0.3,
            "defects": np.random.randint(0, 10),
            "areas_to_check": ["edge_quality", "surface_finish", "dimensions"]
        }
    
    @staticmethod
    def _simulate_assembly_verification(image: np.ndarray) -> Dict:
        """Simulate assembly verification"""
        return {
            "assembly_complete": np.random.random() > 0.2,
            "missing_parts": np.random.randint(0, 3),
            "misaligned_parts": np.random.randint(0, 2),
            "verification_score": np.random.random()
        }
    
    def start(self):
        """Start API server"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
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
            # Process image data
            image_data = message.get("image")
            task = message.get("task", "inspect")
            
            # Decode and process
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Simulate processing
            result = {
                "task": task,
                "result": "processed",
                "timestamp": datetime.now().isoformat()
            }
            
            return {"type": "result", "data": result}
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