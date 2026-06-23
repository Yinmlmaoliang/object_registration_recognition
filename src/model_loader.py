"""
Model Loader for Object Registration and Recognition System.

This module provides a unified interface for loading all required models:
- RexOmni: Object detection
- SAM2: Instance segmentation
- DINOv3-MGFA: Feature extraction
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class ModelLoader:
    """
    Unified model loading interface.

    Loads and manages all required models:
    - RexOmni (Detection)
    - SAM2 (Segmentation)
    - DINOv3-MGFA (Feature Extraction)

    Example:
        >>> loader = ModelLoader(device="cuda")
        >>> models = loader.load_all()
        >>> detector = models["detector"]
        >>> segmenter = models["segmenter"]
        >>> extractor = models["extractor"]
    """

    # Default model paths (relative to project root)
    DEFAULT_REXOMNI_PATH = "models/Rex-Omni/checkpoints"
    DEFAULT_SAM2_CHECKPOINT = "models/sam2/checkpoints/sam2.1_hiera_large.pt"
    DEFAULT_DINOV3_BACKBONE = "vitb16"

    def __init__(
        self,
        device: str = "cuda",
        rexomni_model_path: Optional[str] = None,
        sam2_checkpoint: Optional[str] = None,
        dinov3_backbone: str = "vitb16",
        verbose: bool = True
    ):
        """
        Initialize the model loader.

        Args:
            device: Device to load models on ('cuda' or 'cpu')
            rexomni_model_path: Path to RexOmni model weights
            sam2_checkpoint: Path to SAM2 checkpoint
            dinov3_backbone: DINOv3 backbone type ('vits16', 'vitb16', 'vitl16')
            verbose: Whether to print loading progress
        """
        self.device = device
        self.verbose = verbose

        # Set paths
        self.rexomni_model_path = rexomni_model_path or str(
            PROJECT_ROOT / self.DEFAULT_REXOMNI_PATH
        )
        self.sam2_checkpoint = sam2_checkpoint or str(
            PROJECT_ROOT / self.DEFAULT_SAM2_CHECKPOINT
        )
        self.dinov3_backbone = dinov3_backbone

        # Model instances (lazy loading)
        self._detector = None
        self._segmenter = None
        self._extractor = None
        self._matcher = None
        self._text_encoder = None

        if self.verbose:
            print("=" * 60)
            print("ModelLoader initialized")
            print("=" * 60)
            print(f"  Device: {self.device}")
            print(f"  RexOmni path: {self.rexomni_model_path}")
            print(f"  SAM2 checkpoint: {self.sam2_checkpoint}")
            print(f"  DINOv3 backbone: {self.dinov3_backbone}")
            print("=" * 60)

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def load_detector(self) -> Any:
        """Load the RexOmni detector model."""
        if self._detector is None:
            self._log("\n[1/4] Loading RexOmni detector...")

            from .utils.rexomni_detector import RexOmniDetector

            self._detector = RexOmniDetector(
                model_path=self.rexomni_model_path,
                backend="transformers",
                verbose=self.verbose
            )

            self._log("  RexOmni detector loaded successfully!")

        return self._detector

    def load_segmenter(self) -> Any:
        """Load the SAM2 segmenter model."""
        if self._segmenter is None:
            self._log("\n[2/4] Loading SAM2 segmenter...")

            from .utils.sam2_segmenter import SAM2Segmenter

            self._segmenter = SAM2Segmenter(
                checkpoint_path=self.sam2_checkpoint,
                device=self.device,
                verbose=self.verbose
            )

            self._log("  SAM2 segmenter loaded successfully!")

        return self._segmenter

    def load_extractor(self) -> Any:
        """Load the DINOv3-MGFA feature extractor."""
        if self._extractor is None:
            self._log("\n[3/4] Loading DINOv3-MGFA extractor...")

            from .utils.mgfa_extractor import MGFAFeatureExtractor

            self._extractor = MGFAFeatureExtractor(
                backbone=self.dinov3_backbone,
                device=self.device,
                image_size=448,
                normalize_features=True
            )

            self._log(f"  DINOv3 {self.dinov3_backbone} extractor loaded!")
            self._log(f"  Feature dimension: {self._extractor.feat_dim}")

        return self._extractor

    def load_matcher(self) -> Any:
        """Load the template matcher."""
        if self._matcher is None:
            self._log("\n[4/4] Loading template matcher...")

            from .utils.matcher import TemplateMatcher

            self._matcher = TemplateMatcher(
                similarity_metric='cosine',
                aggregation='mean',
                device=self.device
            )

            self._log("  Template matcher loaded successfully!")

        return self._matcher

    def load_text_encoder(self) -> Any:
        """Load the text encoder for attribute/query embedding."""
        if self._text_encoder is None:
            self._log("\n[5/5] Loading text encoder...")

            from .utils.text_encoder import TextEncoder

            self._text_encoder = TextEncoder(
                device=self.device,
                verbose=self.verbose
            )

            self._log(f"  Text encoder loaded successfully!")
            self._log(f"  Embedding dimension: {self._text_encoder.embedding_dim}")

        return self._text_encoder

    def load_all(self, include_text_encoder: bool = False) -> Dict[str, Any]:
        """
        Load all models and return them as a dictionary.

        Args:
            include_text_encoder: Whether to include the text encoder

        Returns:
            Dictionary containing all model instances
        """
        self._log("\nLoading all models...")
        self._log("=" * 60)

        models = {
            'detector': self.load_detector(),
            'segmenter': self.load_segmenter(),
            'extractor': self.load_extractor(),
            'matcher': self.load_matcher()
        }

        if include_text_encoder:
            models['text_encoder'] = self.load_text_encoder()

        self._log("\n" + "=" * 60)
        self._log("All models loaded successfully!")
        self._log("=" * 60)

        return models

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'device': self.device,
            'rexomni_path': self.rexomni_model_path,
            'sam2_checkpoint': self.sam2_checkpoint,
            'dinov3_backbone': self.dinov3_backbone,
            'models_loaded': {
                'detector': self._detector is not None,
                'segmenter': self._segmenter is not None,
                'extractor': self._extractor is not None,
                'matcher': self._matcher is not None,
                'text_encoder': self._text_encoder is not None
            }
        }

        if self._extractor is not None:
            info['visual_feature_dim'] = self._extractor.feat_dim

        if self._text_encoder is not None:
            info['text_embedding_dim'] = self._text_encoder.embedding_dim

        return info

    @property
    def detector(self) -> Any:
        """Get the detector model (lazy loading)."""
        return self.load_detector()

    @property
    def segmenter(self) -> Any:
        """Get the segmenter model (lazy loading)."""
        return self.load_segmenter()

    @property
    def extractor(self) -> Any:
        """Get the feature extractor (lazy loading)."""
        return self.load_extractor()

    @property
    def matcher(self) -> Any:
        """Get the template matcher (lazy loading)."""
        return self.load_matcher()

    @property
    def text_encoder(self) -> Any:
        """Get the text encoder (lazy loading)."""
        return self.load_text_encoder()
