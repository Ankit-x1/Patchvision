"""
Comprehensive testing and integration tests for PatchVision
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.patches.factory import PatchFactory
from core.projections.transformer import TokenProjector
from core.processors.engine import OptimizedProcessor
from core.validation.input_validator import InputValidator, create_validation_pipeline
from core.validation.integrity import DataIntegrityValidator, create_integrity_validator
from core.utils.unified_error_handler import UnifiedErrorHandler, create_unified_error_handler
from core.analytics.monitoring import create_monitoring_system
from core.models.model_manager import ModelManager


class TestPatchVisionCore(unittest.TestCase):
    """Test core PatchVision components"""
    
    def setUp(self):
        """Setup test environment"""
        self.patch_factory = PatchFactory()
        self.projector = TokenProjector(dim=128)
        self.processor = OptimizedProcessor()
        self.validator = create_validation_pipeline(strict_mode=False)
        self.integrity_validator = create_integrity_validator(strict_mode=False)
        self.error_handler = create_unified_error_handler()
        
        # Create test data
        self.test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.test_batch = [np.random.randn(32, 32) for _ in range(4)]
    
    def test_patch_factory(self):
        """Test PatchFactory functionality"""
        print("Testing PatchFactory...")
        
        # Test adaptive patching
        patches = self.patch_factory.adaptive_patching(self.test_image)
        self.assertGreater(len(patches), 0, "Should generate patches")
        
        # Test hierarchical patching
        pyramid = self.patch_factory.hierarchical_patching(self.test_image, levels=2)
        self.assertEqual(len(pyramid), 2, "Should generate 2 pyramid levels")
        
        print("✓ PatchFactory tests passed")
    
    def test_token_projector(self):
        """Test TokenProjector functionality"""
        print("Testing TokenProjector...")
        
        # Test forward pass
        patches = self.patch_factory.adaptive_patching(self.test_image)
        if len(patches) > 0:
            patch_array = np.array([p['data'] for p in patches])
            patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
            
            tokens = self.projector.forward(patch_array)
            self.assertEqual(tokens.shape[0], 1, "Batch size should be 1")
            self.assertEqual(tokens.shape[2], 128, "Token dimension should be 128")
        
        # Test batch processing
        batch_tokens = self.projector.batch_forward([patch_array[:10] for _ in range(2)])
        self.assertEqual(len(batch_tokens), 2, "Should process 2 batches")
        
        print("✓ TokenProjector tests passed")
    
    def test_optimized_processor(self):
        """Test OptimizedProcessor functionality"""
        print("Testing OptimizedProcessor...")
        
        # Test matrix multiplication
        A = np.random.randn(32, 64)
        B = np.random.randn(64, 32)
        result = self.processor.process('matmul', A, B)
        self.assertEqual(result.shape, (32, 32), "MatMul result shape mismatch")
        
        # Test attention
        Q = np.random.randn(2, 8, 16)
        K = np.random.randn(2, 8, 16)
        V = np.random.randn(2, 8, 16)
        attention_result = self.processor.process('attention', Q, K, V)
        self.assertEqual(attention_result.shape, (2, 8, 16), "Attention result shape mismatch")
        
        print("✓ OptimizedProcessor tests passed")
    
    def test_input_validation(self):
        """Test input validation"""
        print("Testing InputValidator...")
        
        # Test valid image
        result = self.validator.validate_image_input(self.test_image)
        self.assertTrue(result.is_valid, "Valid image should pass validation")
        
        # Test invalid image (NaN)
        invalid_image = self.test_image.astype(np.float32)
        invalid_image[0, 0, 0] = np.nan
        result = self.validator.validate_image_input(invalid_image)
        self.assertFalse(result.is_valid, "Image with NaN should fail validation")
        self.assertGreater(len(result.errors), 0, "Should have validation errors")
        
        # Test batch validation
        result = self.validator.validate_batch_input(self.test_batch)
        self.assertTrue(result.is_valid, "Valid batch should pass validation")
        
        print("✓ InputValidator tests passed")
    
    def test_data_integrity(self):
        """Test data integrity validation"""
        print("Testing DataIntegrityValidator...")
        
        # Test array integrity
        checks = self.integrity_validator.validate_array_integrity(self.test_image)
        self.assertGreater(len(checks), 0, "Should generate integrity checks")
        
        # Test batch integrity
        batch_result = self.integrity_validator.validate_batch_integrity(self.test_batch)
        self.assertTrue(batch_result["batch_valid"], "Valid batch should pass integrity check")
        
        print("✓ DataIntegrityValidator tests passed")
    
    def test_error_handling(self):
        """Test error handling"""
        print("Testing Error Handling...")
        
        # Test error registration and handling
        test_error = ValueError("Test error")
        context = {"component": "test", "operation": "unit_test"}
        
        recovered = self.error_handler.handle_error(test_error, context)
        # Should return False for unregistered error types
        self.assertFalse(recovered, "Unregistered error should not be recovered")
        
        # Test error statistics
        stats = self.error_handler.get_error_statistics()
        self.assertIn("total_errors", stats, "Should track total errors")
        
        print("✓ Error handling tests passed")
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        print("Testing Integration Workflow...")
        
        try:
            # Step 1: Validate input
            validation_result = self.validator.validate_image_input(self.test_image)
            self.assertTrue(validation_result.is_valid, "Input validation should pass")
            
            # Step 2: Generate patches
            patches = self.patch_factory.adaptive_patching(self.test_image)
            self.assertGreater(len(patches), 0, "Should generate patches")
            
            # Step 3: Project to tokens
            if len(patches) > 0:
                patch_array = np.array([p['data'] for p in patches])
                patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
                tokens = self.projector.forward(patch_array)
                self.assertEqual(tokens.shape[2], 128, "Token dimension should be 128")
                
                # Step 4: Process with optimized processor
                result = self.processor.process('matmul', tokens, tokens.transpose(0, 2, 1))
                self.assertIsNotNone(result, "Processing should produce result")
            
            print("✓ Integration workflow test passed")
            
        except Exception as e:
            self.fail(f"Integration workflow failed: {e}")


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring system"""
    
    def setUp(self):
        """Setup test environment"""
        self.perf_monitor, self.health_checker = create_monitoring_system()
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        print("Testing Performance Monitoring...")
        
        # Test metric recording
        self.perf_monitor.start_monitoring("test_component")
        self.perf_monitor.record_metric("test_metric", 1.5, component="test_component", unit="ms")
        
        # Test metrics retrieval
        metrics = self.perf_monitor.get_metrics("test_component")
        self.assertIn("test_metric", metrics["test_component"], "Should record test metric")
        
        # Test report generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_generated = self.perf_monitor.generate_report(f.name)
            self.assertTrue(report_generated, "Should generate performance report")
        
        print("✓ Performance monitoring tests passed")
    
    def test_health_checks(self):
        """Test health checks"""
        print("Testing Health Checks...")
        
        # Test health check
        health_status = self.health_checker.check_health("test_component")
        self.assertIn("status", health_status, "Should return health status")
        
        # Test system health
        system_health = self.health_checker.get_system_health()
        self.assertIn("overall_status", system_health, "Should return overall system health")
        
        print("✓ Health check tests passed")


class TestModelManagement(unittest.TestCase):
    """Test model management system"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(model_dir=self.temp_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_management(self):
        """Test model management"""
        print("Testing Model Management...")
        
        # Test model saving
        test_model = {"weights": np.random.randn(10, 10), "config": {"layers": 2}}
        model_path = self.model_manager.save_model(test_model, "test_model", "numpy")
        self.assertIsNotNone(model_path, "Should save model successfully")
        
        # Test model loading
        loaded_model = self.model_manager.load_model("test_model")
        self.assertIsNotNone(loaded_model, "Should load model successfully")
        
        # Test model registry
        self.assertIn("test_model", self.model_manager.model_registry, "Should register model")
        
        print("✓ Model management tests passed")


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline(self):
        """Test complete processing pipeline"""
        print("Testing Complete Pipeline...")
        
        try:
            # Initialize all components
            patch_factory = PatchFactory()
            projector = TokenProjector(dim=64)
            processor = OptimizedProcessor()
            validator = create_validation_pipeline(strict_mode=False)
            
            # Create test data
            test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            
            # Validate input
            validation_result = validator.validate_image_input(test_image)
            self.assertTrue(validation_result.is_valid, "Input should be valid")
            
            # Process through pipeline
            patches = patch_factory.adaptive_patching(test_image)
            self.assertGreater(len(patches), 0, "Should generate patches")
            
            # Convert to tokens
            patch_array = np.array([p['data'] for p in patches])
            patch_array = patch_array.reshape(1, -1, patch_array.shape[-1])
            tokens = projector.forward(patch_array)
            self.assertEqual(tokens.shape[2], 64, "Token dimension should be 64")
            
            # Process tokens
            processed = processor.process('matmul', tokens, tokens.transpose(0, 2, 1))
            self.assertIsNotNone(processed, "Should process tokens successfully")
            
            # Validate output integrity
            integrity_validator = create_integrity_validator(strict_mode=False)
            integrity_checks = integrity_validator.validate_array_integrity(processed)
            self.assertGreater(len(integrity_checks), 0, "Should generate integrity checks")
            
            print("✓ Complete pipeline test passed")
            
        except Exception as e:
            self.fail(f"Complete pipeline test failed: {e}")
    
    def test_error_recovery_integration(self):
        """Test error recovery in integration"""
        print("Testing Error Recovery Integration...")
        
        # Initialize error handler with recovery actions
        error_handler = create_unified_error_handler()
        
        # Test recovery action registration
        def test_recovery(error, context, retry_count):
            return retry_count < 2
        
        error_handler.register_recovery_action("ValueError", test_recovery)
        
        # Test error with recovery
        test_error = ValueError("Recoverable error")
        context = {"component": "integration_test", "operation": "recovery_test"}
        
        recovered = error_handler.handle_error(test_error, context)
        self.assertTrue(recovered, "Should recover from error")
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        self.assertIn("total_errors", stats, "Should track error statistics")
        
        print("✓ Error recovery integration test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("PATCHVISION COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestPatchVisionCore))
    test_suite.addTest(unittest.makeSuite(TestPerformanceMonitoring))
    test_suite.addTest(unittest.makeSuite(TestModelManagement))
    test_suite.addTest(unittest.makeSuite(TestEndToEndIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 60)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
