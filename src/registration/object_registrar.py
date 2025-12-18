"""
Object Registrar for Few-Shot Object Registration System.

This module implements the core business logic for registering objects
from handheld scenes and building template libraries for later recognition.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class RegistrationResult:
    """Result of a single object registration attempt."""
    success: bool
    instance_id: str
    feature_dim: Optional[int] = None
    detection_bbox: Optional[List[float]] = None
    detection_confidence: Optional[float] = None
    segmentation_score: Optional[float] = None
    error_message: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"RegistrationResult(success=True, "
                f"instance_id='{self.instance_id}', "
                f"feature_dim={self.feature_dim})"
            )
        else:
            return (
                f"RegistrationResult(success=False, "
                f"instance_id='{self.instance_id}', "
                f"error='{self.error_message}')"
            )


class ObjectRegistrar:
    """
    Core class for registering objects from handheld scenes.

    This class encapsulates the "handheld interaction learning" logic:
    1. Detect the handheld object using RexOmni
    2. Segment the object using SAM2
    3. Extract features using DINOv3-FFA
    4. Store the feature vector in the template library

    Example:
        >>> from src.model_loader import ModelLoader
        >>> from src.registration import ObjectRegistrar
        >>>
        >>> loader = ModelLoader(device="cuda")
        >>> models = loader.load_all()
        >>>
        >>> registrar = ObjectRegistrar(
        ...     detector=models['detector'],
        ...     segmenter=models['segmenter'],
        ...     extractor=models['extractor']
        ... )
        >>>
        >>> image = Image.open("handheld_mug.jpg")
        >>> result = registrar.register(image, "mug_01")
        >>> print(result)
    """

    DEFAULT_PROMPT = "item held by hand"

    def __init__(
        self,
        detector: Any,
        segmenter: Any,
        extractor: Any,
        default_prompt: str = DEFAULT_PROMPT,
        verbose: bool = True
    ):
        """
        Initialize the ObjectRegistrar.

        Args:
            detector: RexOmniDetector instance
            segmenter: SAM2Segmenter instance
            extractor: FFAFeatureExtractor instance
            default_prompt: Default text prompt for detection
            verbose: Whether to print progress information
        """
        self.detector = detector
        self.segmenter = segmenter
        self.extractor = extractor
        self.default_prompt = default_prompt
        self.verbose = verbose

        # Template storage
        self._templates: Dict[str, List[np.ndarray]] = {}
        self._metadata: Dict[str, List[Dict]] = {}
        self._text_embeddings: Dict[str, np.ndarray] = {}
        self._attributes: Dict[str, List[str]] = {}

        if self.verbose:
            print("ObjectRegistrar initialized")
            print(f"  Default prompt: '{self.default_prompt}'")
            print(f"  Feature dimension: {self.extractor.feat_dim}")

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def register(
        self,
        image: Union[Image.Image, str, Path],
        instance_id: str,
        prompt: Optional[str] = None,
        gt_bbox: Optional[List[float]] = None
    ) -> RegistrationResult:
        """
        Register a single object from a handheld scene image.

        Args:
            image: PIL Image, or path to image file
            instance_id: Unique identifier for this object instance
            prompt: Text prompt for detection (uses default if None)
            gt_bbox: Optional ground truth bounding box [x0, y0, x1, y1]

        Returns:
            RegistrationResult with success status and details
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type: {type(image)}")

        prompt = prompt or self.default_prompt

        self._log(f"\nRegistering object: {instance_id}")
        self._log(f"  Prompt: '{prompt}'")

        # Stage 1: Detection
        self._log("  [Stage 1] Detecting object...")
        detection_result = self.detector.detect(image, prompt)

        if not detection_result['success'] or detection_result['num_detections'] == 0:
            error_msg = detection_result.get('error', 'No objects detected')
            self._log(f"  [FAILED] Detection failed: {error_msg}")
            return RegistrationResult(
                success=False,
                instance_id=instance_id,
                error_message=f"Detection failed: {error_msg}"
            )

        # Select best prediction
        predictions = detection_result['predictions']
        if gt_bbox is not None and len(predictions) > 1:
            from ..utils.bbox_utils import bbox_iou
            best_bbox, best_iou = self.detector.get_best_prediction(
                predictions, gt_bbox, bbox_iou
            )
            if best_bbox is None:
                best_bbox = predictions[0]
        else:
            best_bbox = predictions[0]

        self._log(f"  [Stage 1] Detected {len(predictions)} object(s)")
        self._log(f"            Selected bbox: {[f'{x:.1f}' for x in best_bbox]}")

        # Stage 2: Segmentation
        self._log("  [Stage 2] Segmenting object...")
        seg_result = self.segmenter.segment_single(image, best_bbox)

        if not seg_result['success']:
            error_msg = seg_result.get('error', 'Segmentation failed')
            self._log(f"  [FAILED] Segmentation failed: {error_msg}")
            return RegistrationResult(
                success=False,
                instance_id=instance_id,
                detection_bbox=best_bbox,
                error_message=f"Segmentation failed: {error_msg}"
            )

        mask = seg_result['mask']
        seg_score = seg_result['score']
        self._log(f"  [Stage 2] Segmentation complete (score: {seg_score:.3f})")

        # Stage 3: Feature Extraction
        self._log("  [Stage 3] Extracting features...")
        embedding = self.extractor.extract_embedding(
            image, bbox=best_bbox, mask=mask, use_ffa=True
        )

        self._log(f"  [Stage 3] Feature extracted (dim: {embedding.shape[0]})")
        self._log(f"            Norm: {np.linalg.norm(embedding):.4f}")

        # Stage 4: Store template
        self._log("  [Stage 4] Storing template...")
        if instance_id not in self._templates:
            self._templates[instance_id] = []
            self._metadata[instance_id] = []

        self._templates[instance_id].append(embedding)
        self._metadata[instance_id].append({
            'bbox': best_bbox,
            'seg_score': seg_score,
            'prompt': prompt
        })

        num_templates = len(self._templates[instance_id])
        self._log(f"  [Stage 4] Template stored ({num_templates} total)")
        self._log(f"  [SUCCESS] Registration complete!")

        return RegistrationResult(
            success=True,
            instance_id=instance_id,
            feature_dim=embedding.shape[0],
            detection_bbox=best_bbox,
            segmentation_score=seg_score
        )

    def register_with_attributes(
        self,
        image: Union[Image.Image, str, Path],
        instance_id: str,
        attributes: List[str],
        text_encoder: Any,
        prompt: Optional[str] = None,
        gt_bbox: Optional[List[float]] = None
    ) -> RegistrationResult:
        """
        Register an object with text attributes.

        This method performs visual registration and also encodes text attributes
        for text-based retrieval.

        Args:
            image: PIL Image, or path to image file
            instance_id: Unique identifier for this object instance
            attributes: List of text attribute descriptions
            text_encoder: TextEncoder instance for encoding attributes
            prompt: Text prompt for detection (uses default if None)
            gt_bbox: Optional ground truth bounding box [x0, y0, x1, y1]

        Returns:
            RegistrationResult with success status and details
        """
        # Perform visual registration
        result = self.register(image, instance_id, prompt, gt_bbox)

        # Encode text attributes (only on first registration for this instance)
        if result.success and instance_id not in self._text_embeddings:
            if attributes and text_encoder is not None:
                self._log(f"  [Stage 5] Encoding text attributes...")
                text_embedding = text_encoder.encode_attributes(attributes)
                self._text_embeddings[instance_id] = text_embedding
                self._attributes[instance_id] = attributes
                self._log(f"  [Stage 5] Text embedding stored (dim: {text_embedding.shape[0]})")

        return result

    def register_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        instance_id: str,
        prompt: Optional[str] = None
    ) -> List[RegistrationResult]:
        """
        Register multiple images of the same object instance.

        Args:
            images: List of PIL Images or paths
            instance_id: Unique identifier for this object instance
            prompt: Text prompt for detection

        Returns:
            List of RegistrationResult for each image
        """
        results = []
        self._log(f"\nBatch registration: {instance_id} ({len(images)} images)")

        for i, image in enumerate(images):
            self._log(f"\n--- Image {i + 1}/{len(images)} ---")
            result = self.register(image, instance_id, prompt)
            results.append(result)

        successes = sum(1 for r in results if r.success)
        self._log(f"\nBatch complete: {successes}/{len(images)} successful")

        return results

    def get_templates(self) -> Dict[str, np.ndarray]:
        """Get all registered templates as stacked arrays."""
        return {
            inst_id: np.stack(embeddings, axis=0)
            for inst_id, embeddings in self._templates.items()
            if len(embeddings) > 0
        }

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered templates."""
        stats = {
            'num_instances': len(self._templates),
            'total_templates': sum(len(v) for v in self._templates.values()),
            'templates_per_instance': {
                inst_id: len(embeddings)
                for inst_id, embeddings in self._templates.items()
            },
            'visual_feature_dim': self.extractor.feat_dim,
            'has_text_embeddings': len(self._text_embeddings) > 0,
            'num_text_embeddings': len(self._text_embeddings)
        }

        if stats['num_instances'] > 0:
            stats['avg_templates_per_instance'] = (
                stats['total_templates'] / stats['num_instances']
            )
        else:
            stats['avg_templates_per_instance'] = 0.0

        # Add text embedding dimension if available
        if self._text_embeddings:
            first_emb = next(iter(self._text_embeddings.values()))
            stats['text_embedding_dim'] = first_emb.shape[0]

        return stats

    def save_templates(self, save_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Save templates to disk.

        Saves both visual and text embeddings in a unified format.

        Args:
            save_path: Base path for saving (without extension)

        Returns:
            Tuple of (templates_path, metadata_path)
        """
        import pickle
        import json

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save templates (unified format with visual and text embeddings)
        templates_path = f"{save_path}_templates.pkl"
        templates_to_save = {
            'visual_embeddings': self.get_templates(),
            'text_embeddings': self._text_embeddings.copy()
        }

        with open(templates_path, 'wb') as f:
            pickle.dump(templates_to_save, f)

        # Save metadata (including attributes)
        metadata_path = f"{save_path}_metadata.json"
        metadata_json = {}

        for inst_id, meta_list in self._metadata.items():
            metadata_json[inst_id] = {
                'visual_metadata': [],
                'attributes': self._attributes.get(inst_id, [])
            }
            for meta in meta_list:
                meta_copy = meta.copy()
                if 'bbox' in meta_copy and meta_copy['bbox'] is not None:
                    meta_copy['bbox'] = [float(x) for x in meta_copy['bbox']]
                if 'seg_score' in meta_copy and meta_copy['seg_score'] is not None:
                    meta_copy['seg_score'] = float(meta_copy['seg_score'])
                metadata_json[inst_id]['visual_metadata'].append(meta_copy)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)

        self._log(f"\nTemplates saved:")
        self._log(f"  Templates: {templates_path}")
        self._log(f"  Metadata: {metadata_path}")
        if self._text_embeddings:
            self._log(f"  Text embeddings: {len(self._text_embeddings)} instances")

        return templates_path, metadata_path

    def load_templates(self, load_path: Union[str, Path]) -> int:
        """
        Load templates from disk.

        Supports both new unified format and old format for backward compatibility.

        Args:
            load_path: Base path for loading (without extension)

        Returns:
            Number of instances loaded
        """
        import pickle
        import json

        templates_path = f"{load_path}_templates.pkl"
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

        # Load visual templates
        for inst_id, templates_array in loaded_templates.items():
            if inst_id not in self._templates:
                self._templates[inst_id] = []
            for i in range(templates_array.shape[0]):
                self._templates[inst_id].append(templates_array[i])

        # Load text embeddings
        for inst_id, text_emb in loaded_text_embeddings.items():
            if inst_id not in self._text_embeddings:
                self._text_embeddings[inst_id] = text_emb

        # Load metadata
        metadata_path = f"{load_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)

            for inst_id, meta_data in loaded_metadata.items():
                # Handle both new and old metadata format
                if isinstance(meta_data, dict) and 'visual_metadata' in meta_data:
                    # New format
                    meta_list = meta_data['visual_metadata']
                    attributes = meta_data.get('attributes', [])
                    if attributes and inst_id not in self._attributes:
                        self._attributes[inst_id] = attributes
                else:
                    # Old format: meta_data is directly a list
                    meta_list = meta_data

                if inst_id not in self._metadata:
                    self._metadata[inst_id] = []
                self._metadata[inst_id].extend(meta_list)

        self._log(f"\nTemplates loaded from {load_path}")
        self._log(f"  Instances: {len(loaded_templates)}")
        if loaded_text_embeddings:
            self._log(f"  Text embeddings: {len(loaded_text_embeddings)} instances")

        return len(loaded_templates)

    def clear_templates(self):
        """Clear all registered templates."""
        self._templates.clear()
        self._metadata.clear()
        self._text_embeddings.clear()
        self._attributes.clear()
        self._log("All templates cleared")

    def remove_instance(self, instance_id: str) -> bool:
        """Remove templates for a specific instance."""
        if instance_id in self._templates:
            del self._templates[instance_id]
            if instance_id in self._metadata:
                del self._metadata[instance_id]
            if instance_id in self._text_embeddings:
                del self._text_embeddings[instance_id]
            if instance_id in self._attributes:
                del self._attributes[instance_id]
            self._log(f"Removed instance: {instance_id}")
            return True
        return False

    def list_instances(self) -> List[str]:
        """List all registered instance IDs."""
        return list(self._templates.keys())

    def get_text_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all text embeddings."""
        return self._text_embeddings.copy()

    def get_attributes(self) -> Dict[str, List[str]]:
        """Get all attributes."""
        return self._attributes.copy()

    def has_text_embeddings(self) -> bool:
        """Check if text embeddings are available."""
        return len(self._text_embeddings) > 0
