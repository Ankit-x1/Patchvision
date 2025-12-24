import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import random
import torch
class DataWarfare:
    """
    Advanced data augmentation for robust industrial vision
    """
    
    def __init__(self, 
                 augmentation_config: Optional[Dict] = None):
        self.config = augmentation_config or self._default_config()
        
    def apply_industrial_augmentations(self,
                                      image: np.ndarray,
                                      mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply industrial-specific augmentations
        """
        aug_image = image.copy()
        aug_mask = mask.copy() if mask is not None else None
        
        # Apply augmentations based on config
        for augmentation, params in self.config.items():
            if random.random() < params.get('probability', 0.5):
                if augmentation == 'illumination_variation':
                    aug_image = self._illumination_variation(aug_image, **params)
                elif augmentation == 'motion_blur':
                    aug_image = self._motion_blur(aug_image, **params)
                elif augmentation == 'defocus_blur':
                    aug_image = self._defocus_blur(aug_image, **params)
                elif augmentation == 'sensor_noise':
                    aug_image = self._sensor_noise(aug_image, **params)
                elif augmentation == 'weather_effects':
                    aug_image = self._weather_effects(aug_image, **params)
                elif augmentation == 'occlusion':
                    aug_image, aug_mask = self._occlusion(aug_image, aug_mask, **params)
                elif augmentation == 'perspective_distortion':
                    aug_image = self._perspective_distortion(aug_image, **params)
                    
        return aug_image, aug_mask
    
    def generate_adversarial_variants(self,
                                     image: np.ndarray,
                                     model: callable,
                                     epsilon: float = 0.1) -> np.ndarray:
        """
        Generate adversarial examples for robustness testing
        """
        # FGSM attack
        image_tensor = torch.from_numpy(image).float().requires_grad_(True)
        
        # Forward pass
        output = model(image_tensor.unsqueeze(0))
        
        # Create adversarial perturbation
        loss = output.sum()  # Simple loss for demonstration
        loss.backward()
        
        # Get gradient sign
        perturbation = epsilon * image_tensor.grad.sign()
        
        # Create adversarial example
        adversarial = image_tensor + perturbation
        adversarial = torch.clamp(adversarial, 0, 1)
        
        return adversarial.detach().numpy()
    
    def simulate_sensor_failure(self,
                               image: np.ndarray,
                               failure_type: str) -> np.ndarray:
        """
        Simulate various sensor failures
        """
        if failure_type == 'dead_pixels':
            return self._add_dead_pixels(image)
        elif failure_type == 'column_noise':
            return self._add_column_noise(image)
        elif failure_type == 'row_noise':
            return self._add_row_noise(image)
        elif failure_type == 'hot_pixels':
            return self._add_hot_pixels(image)
        elif failure_type == 'streaking':
            return self._add_streaking(image)
        else:
            return image
    
    # Augmentation implementations
    @staticmethod
    def _illumination_variation(image: np.ndarray,
                               intensity_range: Tuple[float, float] = (0.5, 1.5),
                               color_temp_range: Tuple[int, int] = (3000, 7000)) -> np.ndarray:
        """Simulate illumination variations"""
        # Adjust brightness
        intensity = random.uniform(*intensity_range)
        image = np.clip(image.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
        
        # Adjust color temperature (simplified)
        temp = random.randint(*color_temp_range)
        if temp < 4000:  # Warm
            image[:, :, 0] = np.clip(image[:, :, 0] * 0.9, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)
        elif temp > 6000:  # Cool
            image[:, :, 0] = np.clip(image[:, :, 0] * 1.1, 0, 255)
            image[:, :, 2] = np.clip(image[:, :, 2] * 0.9, 0, 255)
            
        return image
    
    @staticmethod
    def _motion_blur(image: np.ndarray,
                    kernel_size_range: Tuple[int, int] = (3, 15),
                    angle_range: Tuple[float, float] = (0, 180)) -> np.ndarray:
        """Apply motion blur"""
        kernel_size = random.randint(*kernel_size_range)
        angle = random.uniform(*angle_range)
        
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = np.ones(kernel_size)
        
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def _defocus_blur(image: np.ndarray,
                     kernel_size_range: Tuple[int, int] = (3, 11)) -> np.ndarray:
        """Apply defocus blur"""
        kernel_size = random.choice([k for k in range(*kernel_size_range) if k % 2 == 1])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def _sensor_noise(image: np.ndarray,
                     noise_type: str = 'gaussian',
                     intensity: float = 0.1) -> np.ndarray:
        """Add sensor noise"""
        if noise_type == 'gaussian':
            noise = np.random.randn(*image.shape) * intensity * 255
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            result = image.copy()
            num_salt = np.ceil(intensity * image.size * 0.5)
            coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
            result[coords[0], coords[1], :] = 255
            
            num_pepper = np.ceil(intensity * image.size * 0.5)
            coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
            result[coords[0], coords[1], :] = 0
            
            return result
        else:
            return image
    
    @staticmethod
    def _weather_effects(image: np.ndarray,
                        effect_type: str = 'rain') -> np.ndarray:
        """Add weather effects"""
        if effect_type == 'rain':
            # Add rain streaks
            h, w = image.shape[:2]
            rain_layer = np.zeros((h, w, 3), dtype=np.uint8)
            
            for _ in range(random.randint(50, 200)):
                x = random.randint(0, w-1)
                length = random.randint(20, 100)
                width = random.randint(1, 3)
                angle = random.uniform(70, 110)  # Mostly vertical
                
                y1 = random.randint(0, h-length)
                y2 = y1 + length
                
                cv2.line(rain_layer, (x, y1), (x, y2), 
                        (200, 200, 200), width)
                
            # Blend with image
            alpha = 0.3
            return cv2.addWeighted(image, 1-alpha, rain_layer, alpha, 0)
            
        elif effect_type == 'fog':
            # Add fog effect
            h, w = image.shape[:2]
            fog = np.ones((h, w, 3), dtype=np.uint8) * 200
            
            alpha = random.uniform(0.1, 0.4)
            return cv2.addWeighted(image, 1-alpha, fog, alpha, 0)
            
        else:
            return image
    
    @staticmethod
    def _occlusion(image: np.ndarray,
                  mask: Optional[np.ndarray],
                  occlusion_type: str = 'random') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Add occlusion"""
        h, w = image.shape[:2]
        
        if occlusion_type == 'random':
            # Random rectangular occlusion
            occ_h = random.randint(h//10, h//4)
            occ_w = random.randint(w//10, w//4)
            occ_x = random.randint(0, w - occ_w)
            occ_y = random.randint(0, h - occ_h)
            
            image[occ_y:occ_y+occ_h, occ_x:occ_x+occ_w] = 0
            
            if mask is not None:
                mask[occ_y:occ_y+occ_h, occ_x:occ_x+occ_w] = 0
                
        elif occlusion_type == 'stripes':
            # Striped occlusion (simulating machinery)
            stripe_width = random.randint(5, 20)
            for x in range(0, w, stripe_width*2):
                image[:, x:x+stripe_width] = 0
                if mask is not None:
                    mask[:, x:x+stripe_width] = 0
                    
        return image, mask
    
    @staticmethod
    def _perspective_distortion(image: np.ndarray,
                               max_shift: float = 0.1) -> np.ndarray:
        """Apply perspective distortion"""
        h, w = image.shape[:2]
        
        # Random perspective points
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        dx = random.uniform(-max_shift, max_shift) * w
        dy = random.uniform(-max_shift, max_shift) * h
        
        dst_points = np.float32([
            [dx, dy],
            [w - dx, dy],
            [w - dx, h - dy],
            [dx, h - dy]
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, M, (w, h))
    
    # Sensor failure simulations
    @staticmethod
    def _add_dead_pixels(image: np.ndarray,
                        density: float = 0.001) -> np.ndarray:
        """Add dead pixels"""
        result = image.copy()
        h, w = image.shape[:2]
        num_dead = int(density * h * w)
        
        for _ in range(num_dead):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            result[y, x] = 0
            
        return result
    
    @staticmethod
    def _add_column_noise(image: np.ndarray,
                         num_columns: int = 3) -> np.ndarray:
        """Add column noise"""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_columns):
            col = random.randint(0, w-1)
            noise = np.random.randint(0, 256, h)
            result[:, col] = noise
            
        return result
    
    @staticmethod
    def _add_hot_pixels(image: np.ndarray,
                       density: float = 0.0005) -> np.ndarray:
        """Add hot pixels"""
        result = image.copy()
        h, w = image.shape[:2]
        num_hot = int(density * h * w)
        
        for _ in range(num_hot):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            result[y, x] = 255
            
        return result
    
    @staticmethod
    def _add_streaking(image: np.ndarray) -> np.ndarray:
        """Add streaking artifacts"""
        result = image.copy()
        h, w = image.shape[:2]
        
        streak_length = random.randint(h//4, h//2)
        streak_width = random.randint(1, 3)
        x = random.randint(0, w-1)
        y_start = random.randint(0, h - streak_length)
        
        for y in range(y_start, y_start + streak_length):
            result[y, x:x+streak_width] = 255
            
        return result
    
    @staticmethod
    def _default_config() -> Dict:
        """Default augmentation configuration"""
        return {
            'illumination_variation': {
                'probability': 0.7,
                'intensity_range': (0.5, 1.5),
                'color_temp_range': (3000, 7000)
            },
            'motion_blur': {
                'probability': 0.4,
                'kernel_size_range': (3, 15)
            },
            'sensor_noise': {
                'probability': 0.6,
                'noise_type': 'gaussian',
                'intensity': 0.1
            },
            'occlusion': {
                'probability': 0.3,
                'occlusion_type': 'random'
            }
        }

class AdversarialAugmentation:
    """
    Advanced adversarial augmentation for robustness
    """
    
    def __init__(self):
        pass
        
    def apply_physical_adversarial(self,
                                  image: np.ndarray,
                                  patch: np.ndarray,
                                  position: Tuple[int, int]) -> np.ndarray:
        """
        Apply physical adversarial patch
        """
        result = image.copy()
        h, w = patch.shape[:2]
        x, y = position
        
        # Ensure within bounds
        x = max(0, min(x, image.shape[1] - w))
        y = max(0, min(y, image.shape[0] - h))
        
        # Blend patch
        alpha = 0.7
        roi = result[y:y+h, x:x+w]
        blended = cv2.addWeighted(roi, 1-alpha, patch, alpha, 0)
        result[y:y+h, x:x+w] = blended
        
        return result
    
    def generate_robustness_dataset(self,
                                   base_images: List[np.ndarray],
                                   num_variants: int = 10) -> List[np.ndarray]:
        """
        Generate robustness testing dataset
        """
        augmented = []
        warfare = DataWarfare()
        
        for image in base_images:
            for _ in range(num_variants):
                aug_image, _ = warfare.apply_industrial_augmentations(image)
                augmented.append(aug_image)
                
        return augmented