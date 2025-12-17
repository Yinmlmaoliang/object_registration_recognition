"""
Object Recognizer for Few-Shot Object Recognition System.

This module implements the core logic for recognizing registered objects
in query scenes by detecting, segmenting, extracting features, and matching
against template libraries.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class RecognitionResult:
    """Result of a single object recognition attempt."""
    success: bool
    num_detections: int
    recognitions: Optional[List[Dict]] = None  # List of recognized objects
    error_message: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"RecognitionResult(success=True, "
                f"num_detections={self.num_detections}, "
                f"recognitions={len(self.recognitions) if self.recognitions else 0})"
            )
        else:
            return (
                f"RecognitionResult(success=False, "
                f"error='{self.error_message}')"
            )


class ObjectRecognizer:
    """
    Core class for recognizing registered objects in query scenes.

    This class encapsulates the "query-time recognition" logic:
    1. Detect all objects in the scene using RexOmni
    2. Segment each detected object using SAM2
    3. Extract features using DINOv3-FFA
    4. Match features against template library using TemplateMatcher
    5. Return instance IDs and confidence scores

    Example:
        >>> from src.model_loader import ModelLoader
        >>> from src.recognition import ObjectRecognizer
        >>>
        >>> loader = ModelLoader(device="cuda")
        >>> models = loader.load_all()
        >>>
        >>> recognizer = ObjectRecognizer(
        ...     detector=models['detector'],
        ...     segmenter=models['segmenter'],
        ...     extractor=models['extractor'],
        ...     matcher=models['matcher']
        ... )
        >>>
        >>> # Load templates
        >>> recognizer.load_templates("templates/cellphone1")
        >>>
        >>> # Recognize objects in query image
        >>> query_image = Image.open("scene.jpg")
        >>> result = recognizer.recognize(query_image)
        >>> print(result)
    """

    DEFAULT_PROMPT = "objects"

    def __init__(
        self,
        detector: Any,
        segmenter: Any,
        extractor: Any,
        matcher: Any,
        default_prompt: str = DEFAULT_PROMPT,
        confidence_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize the ObjectRecognizer.

        Args:
            detector: RexOmniDetector instance
            segmenter: SAM2Segmenter instance
            extractor: FFAFeatureExtractor instance
            matcher: TemplateMatcher instance
            default_prompt: Default text prompt for detection
            confidence_threshold: Minimum confidence score for recognition
            verbose: Whether to print progress information
        """
        self.detector = detector
        self.segmenter = segmenter
        self.extractor = extractor
        self.matcher = matcher
        self.default_prompt = default_prompt
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        # Template storage
        self._templates: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Any] = {}

        if self.verbose:
            print("ObjectRecognizer initialized")
            print(f"  Default prompt: '{self.default_prompt}'")
            print(f"  Confidence threshold: {self.confidence_threshold}")
            print(f"  Feature dimension: {self.extractor.feat_dim}")

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def load_templates(
        self,
        load_path: Union[str, Path],
        clear_existing: bool = True
    ) -> int:
        """
        Load template library from disk.

        Args:
            load_path: Base path for loading (without extension)
            clear_existing: Whether to clear existing templates

        Returns:
            Number of instances loaded
        """
        import pickle
        import json

        if clear_existing:
            self._templates.clear()
            self._metadata.clear()

        # Load templates
        templates_path = f"{load_path}_templates.pkl"
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"Templates file not found: {templates_path}")

        with open(templates_path, 'rb') as f:
            loaded_templates = pickle.load(f)

        self._templates.update(loaded_templates)

        # Load metadata
        metadata_path = f"{load_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            self._metadata.update(loaded_metadata)

        self._log(f"\nTemplates loaded from {load_path}")
        self._log(f"  Instances: {len(loaded_templates)}")
        for inst_id, templates in loaded_templates.items():
            self._log(f"    - {inst_id}: {len(templates)} templates")

        return len(loaded_templates)

    def set_templates(
        self,
        templates: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set templates directly (without loading from disk).

        Args:
            templates: Template embeddings {instance_id: [N_templates, feat_dim]}
            metadata: Optional metadata dictionary
        """
        self._templates = templates
        if metadata is not None:
            self._metadata = metadata

        self._log(f"\nTemplates set: {len(templates)} instances")
        for inst_id, template_array in templates.items():
            self._log(f"  - {inst_id}: {len(template_array)} templates")

    def recognize(
        self,
        image: Union[Image.Image, str, Path],
        prompt: Optional[str] = None,
        return_features: bool = False,
        return_masks: bool = True
    ) -> RecognitionResult:
        """
        Recognize all objects in a query scene image.

        Args:
            image: PIL Image, or path to image file
            prompt: Text prompt for detection (uses default if None)
            return_features: Whether to include feature vectors in results
            return_masks: Whether to include segmentation masks in results

        Returns:
            RecognitionResult with detected and recognized objects
        """
        # Check if templates are loaded
        if not self._templates:
            return RecognitionResult(
                success=False,
                num_detections=0,
                error_message="No templates loaded. Call load_templates() first."
            )

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type: {type(image)}")

        prompt = prompt or self.default_prompt

        self._log(f"\nRecognizing objects in query image")
        self._log(f"  Image size: {image.size}")
        self._log(f"  Prompt: '{prompt}'")

        # Stage 1: Detection
        self._log("  [Stage 1/4] Detecting objects...")
        detection_result = self.detector.detect(image, prompt)

        if not detection_result['success'] or detection_result['num_detections'] == 0:
            error_msg = detection_result.get('error', 'No objects detected')
            self._log(f"  [FAILED] Detection failed: {error_msg}")
            return RecognitionResult(
                success=False,
                num_detections=0,
                error_message=f"Detection failed: {error_msg}"
            )

        bboxes = detection_result['predictions']
        num_detections = len(bboxes)
        self._log(f"  [Stage 1/4] Detected {num_detections} object(s)")

        # Stage 2: Segmentation (batch)
        self._log("  [Stage 2/4] Segmenting objects...")
        seg_result = self.segmenter.segment_batch(image, bboxes)

        if not seg_result['success']:
            error_msg = seg_result.get('error', 'Segmentation failed')
            self._log(f"  [FAILED] Segmentation failed: {error_msg}")
            return RecognitionResult(
                success=False,
                num_detections=num_detections,
                error_message=f"Segmentation failed: {error_msg}"
            )

        masks = seg_result['masks']
        seg_scores = seg_result['scores']
        self._log(f"  [Stage 2/4] Segmented {len(masks)} object(s)")

        # Stage 3: Feature Extraction
        self._log("  [Stage 3/4] Extracting features...")
        proposal_embeddings = []

        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            embedding = self.extractor.extract_embedding(
                image, bbox=bbox, mask=mask, use_ffa=True
            )
            proposal_embeddings.append(embedding)

        proposal_embeddings = np.stack(proposal_embeddings, axis=0)
        self._log(f"  [Stage 3/4] Extracted {len(proposal_embeddings)} features")

        # Stage 4: Matching
        self._log("  [Stage 4/4] Matching against templates...")
        predictions = self.matcher.match_proposals(
            proposal_embeddings=proposal_embeddings,
            template_embeddings=self._templates,
            return_scores=True,
            top_k=5
        )

        self._log(f"  [Stage 4/4] Matching complete")

        # Build recognition results
        recognitions = []
        for i, pred in enumerate(predictions):
            recognition = {
                'detection_id': i,
                'bbox': bboxes[i],
                'instance_id': pred['instance_id'],
                'confidence': pred['score'],
                'seg_score': float(seg_scores[i]),
                'top_k_matches': pred.get('top_k_matches', [])
            }

            if return_masks:
                recognition['mask'] = masks[i]

            if return_features:
                recognition['feature'] = proposal_embeddings[i]

            recognitions.append(recognition)

        # Filter by confidence threshold
        high_conf_recognitions = [
            r for r in recognitions
            if r['confidence'] >= self.confidence_threshold
        ]

        self._log(f"\n  Recognition summary:")
        self._log(f"    Total detections: {num_detections}")
        self._log(f"    High confidence (>={self.confidence_threshold}): {len(high_conf_recognitions)}")

        for i, rec in enumerate(recognitions):
            conf_str = f"{rec['confidence']:.3f}"
            status = "✓" if rec['confidence'] >= self.confidence_threshold else "✗"
            self._log(f"    {status} Detection {i+1}: {rec['instance_id']} (conf={conf_str})")

        return RecognitionResult(
            success=True,
            num_detections=num_detections,
            recognitions=recognitions
        )

    def recognize_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        prompt: Optional[str] = None,
        return_features: bool = False,
        return_masks: bool = True
    ) -> List[RecognitionResult]:
        """
        Recognize objects in multiple query images.

        Args:
            images: List of PIL Images or paths
            prompt: Text prompt for detection
            return_features: Whether to include feature vectors
            return_masks: Whether to include segmentation masks

        Returns:
            List of RecognitionResult for each image
        """
        results = []
        self._log(f"\nBatch recognition: {len(images)} images")

        for i, image in enumerate(images):
            self._log(f"\n--- Image {i + 1}/{len(images)} ---")
            result = self.recognize(
                image=image,
                prompt=prompt,
                return_features=return_features,
                return_masks=return_masks
            )
            results.append(result)

        # Summary
        successes = sum(1 for r in results if r.success)
        total_detections = sum(r.num_detections for r in results)

        self._log(f"\nBatch complete:")
        self._log(f"  Successful: {successes}/{len(images)}")
        self._log(f"  Total detections: {total_detections}")

        return results

    def get_template_info(self) -> Dict[str, Any]:
        """Get information about loaded templates."""
        if not self._templates:
            return {
                'num_instances': 0,
                'templates_loaded': False
            }

        info = {
            'num_instances': len(self._templates),
            'templates_loaded': True,
            'instance_ids': list(self._templates.keys()),
            'templates_per_instance': {},
            'total_templates': 0,
            'feature_dim': None
        }

        for inst_id, templates in self._templates.items():
            n_templates = len(templates)
            info['templates_per_instance'][inst_id] = n_templates
            info['total_templates'] += n_templates

            if info['feature_dim'] is None and n_templates > 0:
                info['feature_dim'] = templates.shape[1]

        return info

    def clear_templates(self):
        """Clear all loaded templates."""
        self._templates.clear()
        self._metadata.clear()
        self._log("All templates cleared")

    def list_instances(self) -> List[str]:
        """List all registered instance IDs."""
        return list(self._templates.keys())
