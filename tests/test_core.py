import unittest
import numpy as np
from patchvision.core.patches import PatchFactory
from patchvision.core.projections import TokenProjector
from patchvision.core.processors import InferenceEngine

class TestPatchFactory(unittest.TestCase):
    
    def setUp(self):
        self.factory = PatchFactory()
        self.test_image = np.random.rand(256, 256, 3).astype(np.uint8)
    
    def test_adaptive_patching(self):
        patches = self.factory.adaptive_patching(self.test_image)
        self.assertGreater(len(patches), 0)
        self.assertIn('id', patches[0])
        self.assertIn('coordinates', patches[0])
        self.assertIn('importance', patches[0])
    
    def test_hierarchical_patching(self):
        pyramid = self.factory.hierarchical_patching(self.test_image, levels=3)
        self.assertEqual(len(pyramid), 3)
        for level in pyramid:
            self.assertGreater(len(pyramid[level]), 0)

class TestTokenProjector(unittest.TestCase):
    
    def setUp(self):
        self.projector = TokenProjector(dim=512, num_heads=8)
        self.test_tokens = np.random.randn(2, 10, 512).astype(np.float32)
    
    def test_forward(self):
        output = self.projector.forward(self.test_tokens)
        self.assertEqual(output.shape, self.test_tokens.shape)
    
    def test_sparse_projection(self):
        output = self.projector.sparse_projection(self.test_tokens, sparsity=0.3)
        self.assertEqual(output.shape, self.test_tokens.shape)

class TestInferenceEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = InferenceEngine(batch_size=4)
    
    def test_process(self):
        test_data = [np.random.randn(3, 224, 224) for _ in range(10)]
        
        # Mock model
        def mock_model(x):
            return x * 2
        
        result = self.engine.process(test_data, mock_model)
        self.assertEqual(len(result), len(test_data))

if __name__ == '__main__':
    unittest.main()