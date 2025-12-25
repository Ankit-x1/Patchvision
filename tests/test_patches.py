import unittest
import numpy as np
from unittest.mock import patch
from core.patches.factory import PatchFactory, HAS_CV2


class TestPatchFactory(unittest.TestCase):
    def setUp(self):
        """Set up a dummy image and a PatchFactory instance."""
        self.image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        self.factory = PatchFactory(
            config={"patch_size": 16, "stride": 8, "mode": "adaptive"}
        )

    def test_adaptive_patching_grid(self):
        """Test the grid generation of adaptive_patching."""
        patch_size = self.factory.config["patch_size"]
        stride = self.factory.config["stride"]
        patches = self.factory.adaptive_patching(self.image)

        # Calculate expected number of patches
        h, w = self.image.shape[:2]
        expected_x = len(range(0, w - patch_size + 1, stride))
        expected_y = len(range(0, h - patch_size + 1, stride))
        self.assertEqual(len(patches), expected_x * expected_y)

    def test_adaptive_patching_content(self):
        """Test the content and metadata of a single patch."""
        patches = self.factory.adaptive_patching(self.image)
        self.assertTrue(len(patches) > 0)

        patch_info = patches[0]
        self.assertIn("id", patch_info)
        self.assertIn("coordinates", patch_info)
        self.assertIn("data", patch_info)
        self.assertIn("importance", patch_info)
        self.assertIn("resolution", patch_info)
        self.assertIn("metadata", patch_info)

        self.assertEqual(patch_info["id"], 0)
        x, y, ps_w, ps_h = patch_info["coordinates"]
        self.assertEqual(ps_w, self.factory.config["patch_size"])
        self.assertEqual(ps_h, self.factory.config["patch_size"])
        self.assertEqual(patch_info["data"].shape, (self.factory.config["patch_size"], self.factory.config["patch_size"], 3))

    def test_saliency_map_importance(self):
        """Test that saliency map influences patch importance."""
        saliency_map = np.zeros_like(self.gray_image, dtype=np.float32)
        saliency_map[10:20, 10:20] = 1.0  # A salient region

        patches = self.factory.adaptive_patching(self.image, saliency_map=saliency_map)
        
        # Find a patch that overlaps with the salient region
        salient_patch_found = False
        for p in patches:
            if p['importance'] > 0:
                salient_patch_found = True
                break
        self.assertTrue(salient_patch_found, "No patch was flagged as important.")

    @patch("core.patches.factory.HAS_CV2", False)
    def test_entropy_no_cv2_grayscale(self):
        """Test entropy calculation without OpenCV for grayscale images."""
        entropy = self.factory._compute_entropy(self.gray_image[0:16, 0:16])
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0)

    @patch("core.patches.factory.HAS_CV2", False)
    def test_entropy_no_cv2_color(self):
        """Test entropy calculation without OpenCV for color images."""
        entropy = self.factory._compute_entropy(self.image[0:16, 0:16])
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0)
        

if __name__ == "__main__":
    unittest.main()
