import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass
import random

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation"""
    resolution: Tuple[int, int] = (1024, 1024)
    num_objects: int = 10
    noise_level: float = 0.1
    occlusion_prob: float = 0.3
    motion_blur: bool = True
    illumination_variation: bool = True

class SyntheticDataFactory:
    """
    Generate any industrial scenario synthetically
    """
    
    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        
    def generate_defect_dataset(self,
                               num_samples: int = 1000,
                               defect_types: List[str] = None) -> Dict:
        """
        Generate synthetic defect dataset for industrial inspection
        """
        if defect_types is None:
            defect_types = ['crack', 'scratch', 'dent', 'corrosion', 'pitting']
            
        dataset = {'images': [], 'masks': [], 'labels': []}
        
        for i in range(num_samples):
            # Generate base material
            base = self._generate_base_material()
            
            # Add random defects
            num_defects = random.randint(1, 3)
            defects = random.sample(defect_types, num_defects)
            
            image = base.copy()
            mask = np.zeros(base.shape[:2], dtype=np.uint8)
            
            for defect in defects:
                image, defect_mask = self._add_defect(image, defect)
                mask = np.maximum(mask, defect_mask)
                
            # Add noise and variations
            image = self._add_realistic_variations(image)
            
            dataset['images'].append(image)
            dataset['masks'].append(mask)
            dataset['labels'].append(defects)
            
        return dataset
    
    def generate_assembly_scene(self,
                               num_parts: int = 5,
                               assembly_stages: int = 3) -> List[Dict]:
        """
        Generate synthetic assembly line scenes
        """
        scenes = []
        
        for stage in range(assembly_stages):
            scene = {
                'stage': stage,
                'parts': [],
                'image': None,
                'annotations': []
            }
            
            # Generate base scene
            base_image = self._generate_factory_floor()
            
            # Add parts at different assembly stages
            for part_id in range(num_parts):
                part = self._generate_mechanical_part()
                position = self._get_assembly_position(part_id, stage)
                
                # Add part to scene
                base_image = self._overlay_part(base_image, part, position)
                
                scene['parts'].append({
                    'id': part_id,
                    'type': f'part_{part_id}',
                    'position': position,
                    'assembly_complete': stage >= part_id  # Simulate assembly progression
                })
                
                scene['annotations'].append({
                    'bbox': self._get_part_bbox(position, part.shape),
                    'label': f'part_{part_id}'
                })
                
            scene['image'] = base_image
            scenes.append(scene)
            
        return scenes
    
    def generate_thermal_data(self,
                            resolution: Tuple[int, int] = (640, 512),
                            temperature_range: Tuple[float, float] = (20.0, 500.0)) -> np.ndarray:
        """
        Generate synthetic thermal imaging data
        """
        h, w = resolution
        
        # Create base temperature distribution
        thermal = np.random.randn(h, w) * 50 + 273.15  # Kelvin
        
        # Add hot spots (simulating machinery)
        num_hotspots = random.randint(3, 8)
        for _ in range(num_hotspots):
            center_x = random.randint(100, w-100)
            center_y = random.randint(100, h-100)
            intensity = random.uniform(temperature_range[0], temperature_range[1])
            radius = random.randint(20, 80)
            
            # Create Gaussian hotspot
            y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            mask = x*x + y*y <= radius*radius
            thermal[mask] += intensity
            
        # Add cold spots
        num_coldspots = random.randint(2, 5)
        for _ in range(num_coldspots):
            center_x = random.randint(100, w-100)
            center_y = random.randint(100, h-100)
            intensity = random.uniform(-30, -10)
            radius = random.randint(30, 60)
            
            y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            mask = x*x + y*y <= radius*radius
            thermal[mask] += intensity
            
        # Convert to 8-bit for visualization
        thermal_normalized = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
        
        return thermal_normalized.astype(np.uint8)
    
    def generate_lidar_point_cloud(self,
                                 num_points: int = 100000,
                                 scene_type: str = 'industrial') -> np.ndarray:
        """
        Generate synthetic LiDAR point cloud
        """
        points = []
        
        if scene_type == 'industrial':
            # Add floor
            floor_points = self._generate_plane([0, 0, 0], [10, 0, 0], [0, 10, 0], 10000)
            points.extend(floor_points)
            
            # Add machinery
            points.extend(self._generate_machinery())
            
            # Add pipes
            points.extend(self._generate_pipes())
            
            # Add structural elements
            points.extend(self._generate_structure())
            
        elif scene_type == 'urban':
            # Add ground
            ground_points = self._generate_plane([0, 0, 0], [50, 0, 0], [0, 50, 0], 50000)
            points.extend(ground_points)
            
            # Add buildings
            for _ in range(random.randint(5, 15)):
                points.extend(self._generate_building())
                
            # Add vehicles
            for _ in range(random.randint(3, 8)):
                points.extend(self._generate_vehicle())
                
            # Add vegetation
            points.extend(self._generate_vegetation(1000))
            
        # Add noise
        points = np.array(points)
        points += np.random.randn(*points.shape) * 0.01  # Add Gaussian noise
        
        return points[:num_points]  # Limit to requested number
    
    # Helper methods
    def _generate_base_material(self) -> np.ndarray:
        """Generate base material texture"""
        h, w = self.config.resolution
        
        # Create metallic texture
        base = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add grain
        for channel in range(3):
            noise = np.random.randn(h, w) * 30
            base[:, :, channel] = 128 + noise
            
        # Add machining marks
        for _ in range(20):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)
            cv2.line(base, (x1, y1), (x2, y2), (150, 150, 150), 1)
            
        return base
    
    def _add_defect(self, 
                   image: np.ndarray, 
                   defect_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Add specific defect to image"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if defect_type == 'crack':
            # Generate crack
            points = []
            start_x = random.randint(100, w-100)
            start_y = random.randint(100, h-100)
            
            for _ in range(random.randint(5, 20)):
                angle = random.uniform(0, 2*np.pi)
                length = random.randint(20, 100)
                end_x = int(start_x + length * np.cos(angle))
                end_y = int(start_y + length * np.sin(angle))
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (50, 50, 50), 2)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, 2)
                
                start_x, start_y = end_x, end_y
                
        elif defect_type == 'scratch':
            # Generate scratch
            x1 = random.randint(50, w-50)
            y1 = random.randint(50, h-50)
            x2 = random.randint(50, w-50)
            y2 = random.randint(50, h-50)
            
            cv2.line(image, (x1, y1), (x2, y2), (70, 70, 70), 3)
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
            
        elif defect_type == 'dent':
            # Generate dent
            center_x = random.randint(100, w-100)
            center_y = random.randint(100, h-100)
            radius = random.randint(20, 60)
            
            cv2.circle(image, (center_x, center_y), radius, (90, 90, 90), -1)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
        return image, mask
    
    def _add_realistic_variations(self, image: np.ndarray) -> np.ndarray:
        """Add realistic variations to image"""
        # Add Gaussian noise
        if self.config.noise_level > 0:
            noise = np.random.randn(*image.shape) * self.config.noise_level * 255
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        # Add motion blur
        if self.config.motion_blur and random.random() > 0.5:
            kernel_size = random.randint(3, 7)
            angle = random.uniform(0, 180)
            kernel = self._motion_kernel(kernel_size, angle)
            image = cv2.filter2D(image, -1, kernel)
            
        # Add illumination variation
        if self.config.illumination_variation:
            # Simulate non-uniform lighting
            h, w = image.shape[:2]
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            
            illumination = 0.7 + 0.3 * np.exp(-(xx**2 + yy**2) / 0.5)
            image = np.clip(image.astype(np.float32) * illumination[..., None], 0, 255).astype(np.uint8)
            
        return image
    
    @staticmethod
    def _motion_kernel(size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel"""
        kernel = np.zeros((size, size))
        kernel[size//2, :] = np.ones(size)
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        return kernel / kernel.sum()
    
    def _generate_factory_floor(self) -> np.ndarray:
        """Generate factory floor background"""
        h, w = self.config.resolution
        
        # Create concrete floor
        floor = np.zeros((h, w, 3), dtype=np.uint8)
        floor[:, :] = [100, 100, 100]  # Concrete color
        
        # Add floor patterns
        grid_size = 100
        for i in range(0, w, grid_size):
            cv2.line(floor, (i, 0), (i, h), (80, 80, 80), 2)
        for i in range(0, h, grid_size):
            cv2.line(floor, (0, i), (w, i), (80, 80, 80), 2)
            
        return floor
    
    def _generate_mechanical_part(self) -> np.ndarray:
        """Generate synthetic mechanical part"""
        size = random.randint(100, 300)
        part = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Random geometric shape
        shape_type = random.choice(['gear', 'bearing', 'shaft', 'housing'])
        
        if shape_type == 'gear':
            # Draw gear
            center = (size//2, size//2)
            radius = size//2 - 10
            
            cv2.circle(part, center, radius, (200, 200, 200), -1)
            
            # Add teeth
            for angle in np.linspace(0, 2*np.pi, 20, endpoint=False):
                x1 = int(center[0] + radius * np.cos(angle))
                y1 = int(center[1] + radius * np.sin(angle))
                x2 = int(center[0] + (radius + 20) * np.cos(angle + np.pi/20))
                y2 = int(center[1] + (radius + 20) * np.sin(angle + np.pi/20))
                cv2.line(part, (x1, y1), (x2, y2), (150, 150, 150), 5)
                
        elif shape_type == 'bearing':
            # Draw bearing
            center = (size//2, size//2)
            
            cv2.circle(part, center, size//2 - 10, (180, 180, 180), -1)
            cv2.circle(part, center, size//4, (100, 100, 100), -1)
            
            # Add balls
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                x = int(center[0] + (size//3) * np.cos(angle))
                y = int(center[1] + (size//3) * np.sin(angle))
                cv2.circle(part, (x, y), size//20, (220, 220, 220), -1)
                
        return part
    
    @staticmethod
    def _get_assembly_position(part_id: int, stage: int) -> Tuple[int, int]:
        """Get position for part in assembly stage"""
        base_x = 200 + part_id * 150
        base_y = 200 + stage * 100
        return (base_x + random.randint(-20, 20), 
                base_y + random.randint(-20, 20))
    
    @staticmethod
    def _overlay_part(background: np.ndarray, 
                     part: np.ndarray, 
                     position: Tuple[int, int]) -> np.ndarray:
        """Overlay part on background"""
        h, w = part.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        x, y = position
        x = max(0, min(x, bg_w - w))
        y = max(0, min(y, bg_h - h))
        
        # Simple alpha blending (assuming part has alpha or we use binary mask)
        roi = background[y:y+h, x:x+w]
        mask = part[:, :, 0] > 0  # Simple mask based on non-zero pixels
        
        for c in range(3):
            roi_channel = roi[:, :, c]
            part_channel = part[:, :, c]
            roi_channel[mask] = part_channel[mask]
            
        background[y:y+h, x:x+w] = roi
        return background
    
    @staticmethod
    def _get_part_bbox(position: Tuple[int, int], 
                      part_shape: Tuple[int, int]) -> List[int]:
        """Get bounding box for part"""
        x, y = position
        h, w = part_shape[:2]
        return [x, y, x + w, y + h]
    
    # Point cloud generation helpers
    @staticmethod
    def _generate_plane(p1: List[float], 
                       p2: List[float], 
                       p3: List[float], 
                       num_points: int) -> List[List[float]]:
        """Generate points on a plane"""
        points = []
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        
        for _ in range(num_points):
            a, b = np.random.rand(2)
            point = np.array(p1) + a * v1 + b * v2
            points.append(point.tolist())
            
        return points
    
    @staticmethod
    def _generate_machinery() -> List[List[float]]:
        """Generate machinery point cloud"""
        points = []
        
        # Generate a simple machine block
        for x in np.linspace(0, 2, 20):
            for y in np.linspace(0, 1, 10):
                for z in np.linspace(0, 1.5, 15):
                    points.append([x + 1, y + 3, z])
                    
        return points
    
    @staticmethod
    def _generate_pipes() -> List[List[float]]:
        """Generate pipe point cloud"""
        points = []
        
        # Generate cylindrical pipes
        for angle in np.linspace(0, 2*np.pi, 50):
            for z in np.linspace(0, 3, 30):
                x = 2 + 0.2 * np.cos(angle)
                y = 2 + 0.2 * np.sin(angle)
                points.append([x, y, z])
                
        return points