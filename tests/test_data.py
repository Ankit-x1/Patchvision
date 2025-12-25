import unittest
import numpy as np
from patchvision.data.synthetic_factory import SyntheticDataFactory
from patchvision.data.realtime_stream import CameraSource
from patchvision.data.augmentation.warfare import DataWarfare

class TestSyntheticDataFactory(unittest.TestCase):
    
    def setUp(self):
        self.factory = SyntheticDataFactory()
    
    def test_generate_defect_dataset(self):
        dataset = self.factory.generate_defect_dataset(num_samples=5)
        self.assertEqual(len(dataset['images']), 5)
        self.assertEqual(len(dataset['masks']), 5)
        self.assertEqual(len(dataset['labels']), 5)
    
    def test_generate_thermal_data(self):
        thermal = self.factory.generate_thermal_data()
        self.assertEqual(thermal.shape, (640, 512))
        self.assertEqual(thermal.dtype, np.uint8)

class TestDataWarfare(unittest.TestCase):
    
    def setUp(self):
        self.warfare = DataWarfare()
        self.test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_illumination_variation(self):
        augmented = self.warfare._illumination_variation(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        self.assertEqual(augmented.dtype, self.test_image.dtype)
    
    def test_motion_blur(self):
        augmented = self.warfare._motion_blur(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
    
    def test_sensor_noise(self):
        augmented = self.warfare._sensor_noise(self.test_image, noise_type='gaussian')
        self.assertEqual(augmented.shape, self.test_image.shape)

if __name__ == '__main__':
    unittest.main()