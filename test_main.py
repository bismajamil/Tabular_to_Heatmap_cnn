#!/usr/bin/env python3
"""
Unit Tests for Heatmap + CNN Model
Run with: pytest test_main.py -v
Or: python -m unittest test_main.py
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Import from main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    Config, create_sample_data, table_to_heatmap_image,
    convert_data_to_heatmaps, split_data, build_cnn_model
)


class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test that config has expected values"""
        self.assertEqual(Config.IMAGE_SIZE, 28)
        self.assertEqual(Config.EPOCHS, 15)
        self.assertEqual(Config.TEST_SIZE, 0.2)
        
    def test_config_setup(self):
        """Test that output directory is created"""
        Config.setup()
        self.assertTrue(Path(Config.OUTPUT_DIR).exists())


class TestDataCreation(unittest.TestCase):
    """Test data creation functions"""
    
    def setUp(self):
        """Setup test data"""
        self.n_samples = 50
        self.n_features = 10
        
    def test_create_sample_data_shape(self):
        """Test that created data has correct shape"""
        X, y, names = create_sample_data(self.n_samples, self.n_features)
        self.assertEqual(X.shape, (self.n_samples, self.n_features))
        self.assertEqual(y.shape, (self.n_samples,))
        self.assertEqual(len(names), self.n_features)
        
    def test_create_sample_data_values(self):
        """Test that data values are numeric"""
        X, y, _ = create_sample_data(self.n_samples, self.n_features)
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isfinite(y).all())
        
    def test_create_sample_data_labels(self):
        """Test that labels are binary (0 or 1)"""
        X, y, _ = create_sample_data(self.n_samples, self.n_features)
        self.assertTrue(np.all((y == 0) | (y == 1)))


class TestHeatmapConversion(unittest.TestCase):
    """Test heatmap conversion functions"""
    
    def setUp(self):
        """Setup test data"""
        self.data_row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
    def test_table_to_heatmap_image_shape(self):
        """Test that heatmap image has correct shape"""
        image = table_to_heatmap_image(self.data_row, size=28)
        self.assertEqual(image.shape, (28, 28))
        
    def test_table_to_heatmap_image_range(self):
        """Test that pixel values are in correct range"""
        image = table_to_heatmap_image(self.data_row, size=28)
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(image <= 255))
        
    def test_table_to_heatmap_image_type(self):
        """Test that image is correct data type"""
        image = table_to_heatmap_image(self.data_row, size=28)
        self.assertEqual(image.dtype, np.uint8)
        
    def test_convert_data_to_heatmaps_shape(self):
        """Test that converted heatmaps have correct shape"""
        X, _, _ = create_sample_data(50, 10)
        X_images = convert_data_to_heatmaps(X)
        self.assertEqual(X_images.shape[0], 50)
        self.assertEqual(X_images.shape[1], 28)
        self.assertEqual(X_images.shape[2], 28)
        
    def test_convert_data_to_heatmaps_range(self):
        """Test that converted heatmaps are normalized"""
        X, _, _ = create_sample_data(50, 10)
        X_images = convert_data_to_heatmaps(X)
        self.assertTrue(np.all(X_images >= 0))
        self.assertTrue(np.all(X_images <= 1.0))


class TestDataSplitting(unittest.TestCase):
    """Test data splitting functions"""
    
    def setUp(self):
        """Setup test data"""
        X, y, _ = create_sample_data(100, 10)
        self.X_images = convert_data_to_heatmaps(X)
        self.y = y
        
    def test_split_data_shape(self):
        """Test that split data has correct shapes"""
        X_train, X_test, y_train, y_test = split_data(self.X_images, self.y)
        
        # Check total count
        total = len(X_train) + len(X_test)
        self.assertEqual(total, 100)
        
        # Check train/test ratio (~80/20)
        train_ratio = len(X_train) / total
        self.assertGreater(train_ratio, 0.75)
        self.assertLess(train_ratio, 0.85)
        
    def test_split_data_channels(self):
        """Test that split data has correct channel dimension"""
        X_train, X_test, _, _ = split_data(self.X_images, self.y)
        self.assertEqual(X_train.shape[3], 1)  # 1 channel (grayscale)
        self.assertEqual(X_test.shape[3], 1)
        
    def test_split_data_label_distribution(self):
        """Test that both train and test have both classes"""
        X_train, X_test, y_train, y_test = split_data(self.X_images, self.y)
        self.assertTrue(0 in y_train)
        self.assertTrue(1 in y_train)
        self.assertTrue(0 in y_test)
        self.assertTrue(1 in y_test)


class TestModelBuilding(unittest.TestCase):
    """Test model building functions"""
    
    def test_build_cnn_model_type(self):
        """Test that model is built correctly"""
        model = build_cnn_model()
        self.assertIsNotNone(model)
        
    def test_build_cnn_model_layers(self):
        """Test that model has expected layers"""
        model = build_cnn_model()
        # Should have: Conv2D, MaxPool, Conv2D, MaxPool, Flatten, Dense, Dropout, Dense, Dropout, Dense
        self.assertGreaterEqual(len(model.layers), 8)
        
    def test_build_cnn_model_output_shape(self):
        """Test that model output shape is correct"""
        model = build_cnn_model()
        # Output layer should be 1 (binary classification)
        output_shape = model.output_shape
        self.assertEqual(output_shape[-1], 1)
        
    def test_build_cnn_model_compiled(self):
        """Test that model is compiled"""
        model = build_cnn_model()
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to model"""
        # Create data
        X, y, _ = create_sample_data(50, 10)
        self.assertEqual(X.shape[0], 50)
        
        # Convert to heatmaps
        X_images = convert_data_to_heatmaps(X)
        self.assertEqual(X_images.shape[0], 50)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X_images, y)
        self.assertEqual(len(X_train) + len(X_test), 50)
        
        # Build model
        model = build_cnn_model()
        self.assertIsNotNone(model)
        
        # Test prediction
        predictions = model.predict(X_test[:5], verbose=0)
        self.assertEqual(predictions.shape[0], 5)
        self.assertEqual(predictions.shape[1], 1)


# ============================================
# TEST RUNNER
# ============================================
def run_tests():
    """Run all tests with detailed output"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestHeatmapConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestDataSplitting))
    suite.addTests(loader.loadTestsFromTestCase(TestModelBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)