#!/usr/bin/env python3
"""
API Demo for PatchVision
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path

def test_api():
    """Test the PatchVision API"""
    
    # URL of the running API server
    BASE_URL = "http://localhost:8000"
    
    print("Testing PatchVision API")
    print("=" * 50)
    
    # 1. Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 2. List models
    print("\n2. Listing available models...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Models: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Create test image
    print("\n3. Creating test image...")
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # 4. Process image
    print("\n4. Processing image...")
    payload = {
        "image": image_b64,
        "task": "defect_detection",
        "parameters": {
            "confidence_threshold": 0.8
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Task: {result['task']}")
            print(f"Defects found: {result['result'].get('defects_found', 0)}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Batch processing
    print("\n5. Testing batch processing...")
    images_b64 = [image_b64, image_b64, image_b64]  # Same image 3 times
    
    batch_payload = {
        "images": images_b64,
        "task": "quality_inspection"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_process",
            json=batch_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Processed {result['count']} images")
            print(f"Success: {result['success']}")
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")

def websocket_example():
    """Example of WebSocket usage"""
    import asyncio
    import websockets
    
    async def test_websocket():
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket")
                
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                print(f"Ping response: {response}")
                
                # Send test image
                test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', test_image)
                image_b64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "process",
                    "image": image_b64,
                    "task": "inspect"
                }))
                
                response = await websocket.recv()
                print(f"Process response: {response}")
                
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    print("\nTesting WebSocket (requires server running)...")
    asyncio.run(test_websocket())

if __name__ == "__main__":
    print("PatchVision API Demo")
    print("Make sure the API server is running first!")
    print("Run: python main.py --mode serve")
    print("=" * 50)
    
    test_api()
    
    # Uncomment to test WebSocket
    # websocket_example()