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
    3. Extract features using DINOv3-MGFA
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
            extractor: MGFAFeatureExtractor instance
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
        self._text_embeddings: Dict[str, np.ndarray] = {}
        self._attributes: Dict[str, List[str]] = {}

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

        Supports both new unified format and old format for backward compatibility.

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
            self._text_embeddings.clear()
            self._attributes.clear()

        # Load templates
        templates_path = f"{load_path}_templates.pkl"
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"Templates file not found: {templates_path}")

        with open(templates_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Detect format: new unified format vs old format
        if isinstance(loaded_data, dict) and 'visual_embeddings' in loaded_data:
            # New unified format
            loaded_templates = loaded_data['visual_embeddings']
            loaded_text_embeddings = loaded_data.get('text_embeddings', {})
        else:
            # Old format: direct visual embeddings
            loaded_templates = loaded_data
            loaded_text_embeddings = {}

        self._templates.update(loaded_templates)
        self._text_embeddings.update(loaded_text_embeddings)

        # Load metadata
        metadata_path = f"{load_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)

            # Handle both new and old metadata format
            for inst_id, meta_data in loaded_metadata.items():
                if isinstance(meta_data, dict) and 'visual_metadata' in meta_data:
                    # New format
                    self._metadata[inst_id] = meta_data['visual_metadata']
                    attributes = meta_data.get('attributes', [])
                    if attributes:
                        self._attributes[inst_id] = attributes
                else:
                    # Old format: meta_data is directly a list
                    self._metadata[inst_id] = meta_data

        self._log(f"\nTemplates loaded from {load_path}")
        self._log(f"  Instances: {len(loaded_templates)}")
        for inst_id, templates in loaded_templates.items():
            has_text = inst_id in self._text_embeddings
            text_marker = " [+text]" if has_text else ""
            self._log(f"    - {inst_id}: {len(templates)} templates{text_marker}")

        if loaded_text_embeddings:
            self._log(f"  Text embeddings: {len(loaded_text_embeddings)} instances")

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
                image, bbox=bbox, mask=mask, use_mgfa=True
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
        self._text_embeddings.clear()
        self._attributes.clear()
        self._log("All templates cleared")

    def list_instances(self) -> List[str]:
        """List all registered instance IDs."""
        return list(self._templates.keys())

    def has_text_embeddings(self) -> bool:
        """Check if text embeddings are available."""
        return len(self._text_embeddings) > 0

    def get_text_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all text embeddings."""
        return self._text_embeddings.copy()

    def retrieve_by_text(
        self,
        query: str,
        text_encoder: Any,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve matching objects by text query.

        Uses cosine similarity between query embedding and template text embeddings.

        Args:
            query: Natural language query (e.g., "Find my Mickey Mouse mug")
            text_encoder: TextEncoder instance for encoding the query
            top_k: Number of top results to return

        Returns:
            List of dicts with 'instance_id' and 'score', sorted by score descending
        """
        if not self._text_embeddings:
            self._log("Warning: No text embeddings available for retrieval")
            return []

        self._log(f"\nText retrieval for query: '{query}'")

        # Encode query
        query_embedding = text_encoder.encode_query(query)

        # Compute similarities
        results = text_encoder.compute_similarity(
            query_embedding=query_embedding,
            template_embeddings=self._text_embeddings,
            top_k=top_k
        )

        # Log results
        self._log(f"  Top-{min(top_k, len(results))} matches:")
        for i, result in enumerate(results, 1):
            self._log(f"    {i}. {result['instance_id']}: {result['score']:.4f}")

        return results

    def recognize_with_text_query(
        self,
        image: Union[Image.Image, str, Path],
        query: str,
        text_encoder: Any,
        prompt: Optional[str] = None,
        text_threshold: float = 0.3,
        return_features: bool = False,
        return_masks: bool = True
    ) -> 'RecognitionResult':
        """
        Recognize objects in a query scene, filtered by text query.

        First retrieves candidate instances by text similarity, then performs
        visual recognition and filters results to matching instances.

        Args:
            image: PIL Image, or path to image file
            query: Natural language query for text-based retrieval
            text_encoder: TextEncoder instance
            prompt: Text prompt for detection (uses default if None)
            text_threshold: Minimum text similarity score for candidate instances
            return_features: Whether to include feature vectors in results
            return_masks: Whether to include segmentation masks in results

        Returns:
            RecognitionResult with detected and recognized objects matching the query
        """
        # Step 1: Text-based retrieval
        text_results = self.retrieve_by_text(query, text_encoder, top_k=len(self._templates))

        # Filter by threshold
        candidate_instances = {
            r['instance_id'] for r in text_results
            if r['score'] >= text_threshold
        }

        if not candidate_instances:
            self._log(f"  No instances matched text query with threshold >= {text_threshold}")
            return RecognitionResult(
                success=True,
                num_detections=0,
                recognitions=[],
                error_message=None
            )

        self._log(f"  Candidate instances from text retrieval: {candidate_instances}")

        # Step 2: Visual recognition
        result = self.recognize(
            image=image,
            prompt=prompt,
            return_features=return_features,
            return_masks=return_masks
        )

        if not result.success:
            return result

        # Step 3: Filter recognitions to text-matched instances
        filtered_recognitions = []
        for rec in result.recognitions:
            if rec['instance_id'] in candidate_instances:
                # Add text retrieval score
                text_score = next(
                    (r['score'] for r in text_results if r['instance_id'] == rec['instance_id']),
                    0.0
                )
                rec['text_score'] = text_score
                filtered_recognitions.append(rec)

        self._log(f"  Filtered recognitions: {len(filtered_recognitions)} (from {len(result.recognitions)})")

        return RecognitionResult(
            success=True,
            num_detections=result.num_detections,
            recognitions=filtered_recognitions,
            error_message=None
        )
