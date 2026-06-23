#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for src/model_loader.py.
Tests the functionality of the ModelLoader class including lazy loading and error handling.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try to import ModelLoader. 
# If this fails, it might be due to missing dependencies in the environment.
try:
    from src.model_loader import ModelLoader
except ImportError as e:
    print(f"Error importing ModelLoader: {e}")
    print("Please ensure you are running this in the 'rexomni' conda environment.")
    sys.exit(1)


class TestModelLoader(unittest.TestCase):
    """Test cases for the ModelLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda"  # Use CPU for testing to avoid CUDA requirement
        self.loader = ModelLoader(device=self.device, verbose=False)

    def test_initialization(self):
        """Test that ModelLoader initializes with correct default and custom values."""
        loader = ModelLoader(device="cuda", verbose=True)
        self.assertEqual(loader.device, "cuda")
        self.assertTrue(loader.verbose)
        
        # Check that internal model references are None initially (lazy loading)
        self.assertIsNone(loader._detector)
        self.assertIsNone(loader._segmenter)
        self.assertIsNone(loader._extractor)
        self.assertIsNone(loader._matcher)

    @patch('src.utils.rexomni_detector.RexOmniDetector')
    def test_load_detector(self, MockDetector):
        """Test lazy loading of the RexOmni detector."""
        # Setup mock
        mock_instance = MagicMock()
        MockDetector.return_value = mock_instance

        # First access - should trigger load
        detector = self.loader.load_detector()
        
        # Verify instantiation
        MockDetector.assert_called_once()
        self.assertEqual(detector, mock_instance)
        
        # Verify arguments passed to constructor
        call_args = MockDetector.call_args
        self.assertEqual(call_args.kwargs['model_path'], self.loader.rexomni_model_path)
        self.assertEqual(call_args.kwargs['verbose'], self.loader.verbose)

        # Second access - should return cached instance
        detector_again = self.loader.detector  # Access via property
        MockDetector.assert_called_once()  # Should not be called again
        self.assertEqual(detector_again, detector)

    @patch('src.utils.sam2_segmenter.SAM2Segmenter')
    def test_load_segmenter(self, MockSegmenter):
        """Test lazy loading of the SAM2 segmenter."""
        mock_instance = MagicMock()
        MockSegmenter.return_value = mock_instance

        segmenter = self.loader.load_segmenter()
        
        MockSegmenter.assert_called_once()
        self.assertEqual(segmenter, mock_instance)
        
        # Check args
        call_args = MockSegmenter.call_args
        self.assertEqual(call_args.kwargs['checkpoint_path'], self.loader.sam2_checkpoint)
        self.assertEqual(call_args.kwargs['device'], self.loader.device)

        # Property access
        self.assertEqual(self.loader.segmenter, mock_instance)

    @patch('src.utils.mgfa_extractor.MGFAFeatureExtractor')
    def test_load_extractor(self, MockExtractor):
        """Test lazy loading of the DINOv3-MGFA extractor."""
        mock_instance = MagicMock()
        # Mock feat_dim attribute
        mock_instance.feat_dim = 768
        MockExtractor.return_value = mock_instance

        extractor = self.loader.load_extractor()
        
        MockExtractor.assert_called_once()
        self.assertEqual(extractor, mock_instance)
        
        call_args = MockExtractor.call_args
        self.assertEqual(call_args.kwargs['backbone'], self.loader.dinov3_backbone)
        self.assertEqual(call_args.kwargs['device'], self.loader.device)

        # Property access
        self.assertEqual(self.loader.extractor, mock_instance)

    @patch('src.utils.matcher.TemplateMatcher')
    def test_load_matcher(self, MockMatcher):
        """Test lazy loading of the Template Matcher."""
        mock_instance = MagicMock()
        MockMatcher.return_value = mock_instance

        matcher = self.loader.load_matcher()
        
        MockMatcher.assert_called_once()
        self.assertEqual(matcher, mock_instance)
        self.assertEqual(call_args := MockMatcher.call_args, MockMatcher.call_args)
        self.assertEqual(call_args.kwargs['device'], self.loader.device)

    @patch('src.utils.rexomni_detector.RexOmniDetector')
    @patch('src.utils.sam2_segmenter.SAM2Segmenter')
    @patch('src.utils.mgfa_extractor.MGFAFeatureExtractor')
    @patch('src.utils.matcher.TemplateMatcher')
    def test_load_all(self, MockMatcher, MockExtractor, MockSegmenter, MockDetector):
        """Test load_all method."""
        # Setup mocks to avoid actual model loading
        MockDetector.return_value = MagicMock()
        MockSegmenter.return_value = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.feat_dim = 768
        MockExtractor.return_value = mock_extractor
        MockMatcher.return_value = MagicMock()

        models = self.loader.load_all()
        
        self.assertIn('detector', models)
        self.assertIn('segmenter', models)
        self.assertIn('extractor', models)
        self.assertIn('matcher', models)
        
        # Verify all were called
        MockDetector.assert_called_once()
        MockSegmenter.assert_called_once()
        MockExtractor.assert_called_once()
        MockMatcher.assert_called_once()

        # Print model info to confirm loading
        import json
        info = self.loader.get_model_info()
        print("\n\n[Test Info] Model Loading Status:")
        print(json.dumps(info, indent=2, default=str))

    @patch('src.utils.mgfa_extractor.MGFAFeatureExtractor')
    def test_get_model_info(self, MockExtractor):
        """Test get_model_info method."""
        # Initial info (nothing loaded)
        info = self.loader.get_model_info()
        self.assertFalse(info['models_loaded']['detector'])
        
        # Load extractor
        mock_instance = MagicMock()
        mock_instance.feat_dim = 1024
        MockExtractor.return_value = mock_instance
        self.loader.load_extractor()
        
        # Check info again
        info = self.loader.get_model_info()
        self.assertTrue(info['models_loaded']['extractor'])
        self.assertEqual(info['visual_feature_dim'], 1024)


if __name__ == '__main__':
    print(f"Running tests in environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    unittest.main()
