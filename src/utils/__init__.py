"""
Shared utilities for object registration and recognition.

This module provides core tools:
- RexOmniDetector: Object detection
- SAM2Segmenter: Instance segmentation
- MGFAFeatureExtractor: DINOv3 + MGFA feature extraction
- TemplateMatcher: Similarity-based matching
"""

from .rexomni_detector import RexOmniDetector
from .sam2_segmenter import SAM2Segmenter
from .mgfa_extractor import MGFAFeatureExtractor
from .template_manager import TemplateManager
from .matcher import TemplateMatcher, compute_matching_metrics
from .bbox_utils import bbox_iou, seg_map_to_bbox, seg_mask_iou

__all__ = [
    "RexOmniDetector",
    "SAM2Segmenter",
    "MGFAFeatureExtractor",
    "TemplateManager",
    "TemplateMatcher",
    "compute_matching_metrics",
    "bbox_iou",
    "seg_map_to_bbox",
    "seg_mask_iou",
]
