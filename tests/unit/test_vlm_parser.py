#!/usr/bin/env python3
"""
Unit Tests for VisionParse Core Functionality

Tests the main VisionParse class methods and configuration.
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import VisionParse, VisionParseError


class TestVisionParseInitialization(unittest.TestCase):
    """Test VisionParse initialization and configuration"""
    
    def test_default_initialization(self):
        """Test basic initialization with defaults"""
        with patch('os.path.exists', return_value=True):
            parser = VisionParse(vlm_type='gpt4o')
            
            self.assertEqual(parser.vlm_type, 'gpt4o')
            self.assertEqual(parser.confidence_threshold, 0.05)
            self.assertEqual(parser.iou_threshold, 0.1)
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        with patch('os.path.exists', return_value=True):
            parser = VisionParse(
                vlm_type='claude',
                model='claude-4-opus',
                confidence_threshold=0.3,
                iou_threshold=0.5,
                api_key='test-key'
            )
            
            self.assertEqual(parser.vlm_type, 'claude')
            self.assertEqual(parser.model, 'claude-4-opus')
            self.assertEqual(parser.confidence_threshold, 0.3)
            self.assertEqual(parser.iou_threshold, 0.5)
            self.assertEqual(parser.api_key, 'test-key')
    
    def test_invalid_thresholds(self):
        """Test that invalid threshold values raise errors"""
        with patch('os.path.exists', return_value=True):
            # Test confidence threshold out of range
            with self.assertRaises(VisionParseError):
                VisionParse(vlm_type='gpt4o', confidence_threshold=1.5)
            
            with self.assertRaises(VisionParseError):
                VisionParse(vlm_type='gpt4o', confidence_threshold=-0.1)
            
            # Test IoU threshold out of range
            with self.assertRaises(VisionParseError):
                VisionParse(vlm_type='gpt4o', iou_threshold=1.5)
            
            with self.assertRaises(VisionParseError):
                VisionParse(vlm_type='gpt4o', iou_threshold=-0.1)
    
    def test_missing_yolo_model(self):
        """Test error when YOLO model file is missing"""
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(VisionParseError) as context:
                VisionParse(vlm_type='gpt4o')
            
            self.assertIn("YOLO model not found", str(context.exception))
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'vlm_type': 'claude',
                'confidence_threshold': 0.2,
                'yolo_config': {'iou_threshold': 0.4}
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            with patch('os.path.exists', return_value=True):
                parser = VisionParse(config_path=config_path)
                
                # Config values should be loaded
                self.assertEqual(parser.confidence_threshold, 0.2)
        finally:
            os.unlink(config_path)


class TestVisionParseConfiguration(unittest.TestCase):
    """Test configuration methods and property changes"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('os.path.exists', return_value=True):
            self.parser = VisionParse(vlm_type='gpt4o')
    
    def test_threshold_modification(self):
        """Test modifying thresholds after initialization"""
        # Test valid modifications
        self.parser.confidence_threshold = 0.3
        self.parser.iou_threshold = 0.6
        
        self.assertEqual(self.parser.confidence_threshold, 0.3)
        self.assertEqual(self.parser.iou_threshold, 0.6)
    
    def test_api_key_detection(self):
        """Test API key detection from various sources"""
        # Test environment variable detection
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            with patch('os.path.exists', return_value=True):
                parser = VisionParse(vlm_type='openai')
                self.assertEqual(parser.api_key, 'env-key')
    
    def test_default_model_selection(self):
        """Test default model selection for different providers"""
        with patch('os.path.exists', return_value=True):
            # Test different VLM types get appropriate default models
            parser_openai = VisionParse(vlm_type='openai')
            parser_claude = VisionParse(vlm_type='claude')
            
            # Should have different default models
            self.assertNotEqual(parser_openai.model, parser_claude.model)


class TestVisionParseAnalysis(unittest.TestCase):
    """Test analysis methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('os.path.exists', return_value=True):
            self.parser = VisionParse(vlm_type='gpt4o', api_key='test-key')
    
    @patch('src.vlm_parser.detect_ui_elements')
    @patch('src.vlm_parser.crop_image_regions')
    @patch('src.vlm_parser.batch_analyze_regions')
    @patch('src.vlm_parser.draw_boxes_on_image')
    def test_successful_analysis(self, mock_draw, mock_batch_analyze, mock_crop, mock_detect):
        """Test successful image analysis"""
        # Mock YOLO detection
        mock_detect.return_value = [
            {
                'id': 1,
                'bbox': [100, 50, 200, 100],
                'confidence': 0.85,
                'type': 'icon'
            }
        ]
        
        # Mock region cropping
        mock_crop.return_value = [{'id': 1, 'image': 'mock_image'}]
        
        # Mock VLM analysis
        mock_batch_analyze.return_value = [
            {
                'id': 1,
                'bbox': [100, 50, 200, 100],
                'confidence': 0.85,
                'name': 'Test Button',
                'type': 'Button',
                'use': 'Click to test',
                'description': 'A test button for unit testing'
            }
        ]
        
        # Test analysis
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            results = self.parser.analyze(temp_file.name)
            
            self.assertTrue(results['success'])
            self.assertEqual(len(results['elements']), 1)
            self.assertEqual(results['elements'][0]['name'], 'Test Button')
            self.assertEqual(results['yolo_detections'], 1)
    
    def test_missing_image_file(self):
        """Test analysis with missing image file"""
        results = self.parser.analyze('nonexistent_image.png')
        
        self.assertFalse(results['success'])
        self.assertIn('error', results)
    
    @patch('src.vlm_parser.detect_ui_elements')
    def test_no_elements_detected(self, mock_detect):
        """Test analysis when no UI elements are detected"""
        # Mock no detections
        mock_detect.return_value = []
        
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            results = self.parser.analyze(temp_file.name)
            
            self.assertTrue(results['success'])
            self.assertEqual(len(results['elements']), 0)
            self.assertEqual(results['yolo_detections'], 0)


class TestVisionParseBatchProcessing(unittest.TestCase):
    """Test batch processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('os.path.exists', return_value=True):
            self.parser = VisionParse(vlm_type='gpt4o', api_key='test-key')
    
    @patch.object(VisionParse, 'analyze')
    def test_batch_analysis_success(self, mock_analyze):
        """Test successful batch analysis"""
        # Mock individual analysis results
        mock_analyze.side_effect = [
            {
                'success': True,
                'elements': [{'id': 1, 'name': 'Button1'}],
                'image_path': 'image1.png'
            },
            {
                'success': True,
                'elements': [{'id': 1, 'name': 'Button2'}],
                'image_path': 'image2.png'
            }
        ]
        
        # Test batch processing
        image_paths = ['image1.png', 'image2.png']
        results = self.parser.analyze_batch(image_paths)
        
        self.assertTrue(results['success'])
        self.assertEqual(results['summary']['total'], 2)
        self.assertEqual(results['summary']['successful'], 2)
        self.assertEqual(results['summary']['failed'], 0)
    
    @patch.object(VisionParse, 'analyze')
    def test_batch_analysis_mixed_results(self, mock_analyze):
        """Test batch analysis with mixed success/failure"""
        # Mock mixed results
        mock_analyze.side_effect = [
            {
                'success': True,
                'elements': [{'id': 1, 'name': 'Button1'}],
                'image_path': 'image1.png'
            },
            {
                'success': False,
                'error': 'Analysis failed',
                'image_path': 'image2.png'
            }
        ]
        
        # Test batch processing
        image_paths = ['image1.png', 'image2.png']
        results = self.parser.analyze_batch(image_paths)
        
        self.assertTrue(results['success'])
        self.assertEqual(results['summary']['total'], 2)
        self.assertEqual(results['summary']['successful'], 1)
        self.assertEqual(results['summary']['failed'], 1)


class TestVisionParseExport(unittest.TestCase):
    """Test result export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('os.path.exists', return_value=True):
            self.parser = VisionParse(vlm_type='gpt4o')
        
        self.sample_results = {
            'success': True,
            'elements': [
                {
                    'id': 1,
                    'bbox': [100, 50, 200, 100],
                    'confidence': 0.85,
                    'name': 'Test Button',
                    'type': 'Button',
                    'use': 'Testing',
                    'description': 'A test button'
                }
            ]
        }
    
    def test_json_export(self):
        """Test JSON export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            exported_path = self.parser.export_results(
                self.sample_results, 
                output_path, 
                format='json'
            )
            
            self.assertEqual(exported_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify exported content
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertEqual(exported_data['success'], True)
            self.assertEqual(len(exported_data['elements']), 1)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_csv_export(self):
        """Test CSV export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            exported_path = self.parser.export_results(
                self.sample_results,
                output_path,
                format='csv'
            )
            
            self.assertEqual(exported_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify CSV has content
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn('Test Button', content)
                self.assertIn('Button', content)
                
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_unsupported_export_format(self):
        """Test error handling for unsupported export formats"""
        with self.assertRaises(VisionParseError) as context:
            self.parser.export_results(
                self.sample_results,
                'test.xml',
                format='xml'
            )
        
        self.assertIn("Unsupported export format", str(context.exception))


if __name__ == '__main__':
    # Set up test environment
    print("Running VisionParse Unit Tests")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVisionParseInitialization,
        TestVisionParseConfiguration,
        TestVisionParseAnalysis,
        TestVisionParseBatchProcessing,
        TestVisionParseExport
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)