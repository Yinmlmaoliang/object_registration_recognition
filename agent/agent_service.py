"""
Service layer for the conversation agent.

Encapsulates all interactions with ObjectRegistrar and ObjectRecognizer,
providing a clean interface for the conversation agent to use.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import ModelLoader
from src.registration.object_registrar import ObjectRegistrar
from src.recognition.object_recognizer import ObjectRecognizer
from src.utils.result_analyzer import draw_recognition_result


@dataclass
class RegistrationResponse:
    """Response from object registration."""
    success: bool
    instance_id: str
    num_images_processed: int
    num_successful: int
    templates_path: str
    error_message: Optional[str] = None
    has_text_embeddings: bool = False


@dataclass
class RecognitionResponse:
    """Response from object recognition."""
    success: bool
    query: str
    num_images_searched: int
    matches: List[Dict[str, Any]]
    visualization_path: Optional[str] = None
    error_message: Optional[str] = None


class AgentService:
    """
    Service layer that encapsulates object registration and recognition logic.

    This class provides a clean interface for the conversation agent to interact
    with the underlying ObjectRegistrar and ObjectRecognizer classes.

    Example:
        >>> service = AgentService(device="cuda")
        >>> response = service.register_object(
        ...     image_path="examples/handheld/mug1/",
        ...     instance_id="mug1",
        ...     attributes=["my favorite blue mug", "Mickey Mouse pattern"]
        ... )
        >>> print(response.success)
        True
    """

    def __init__(
        self,
        device: str = "cuda",
        default_templates_path: str = "templates/objects",
        default_output_path: str = "recognition_results",
        verbose: bool = False
    ):
        """
        Initialize the AgentService.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
            default_templates_path: Default path for template storage
            default_output_path: Default path for recognition results
            verbose: Whether to print detailed progress
        """
        self.device = device
        self.default_templates_path = Path(PROJECT_ROOT) / default_templates_path
        self.default_output_path = Path(PROJECT_ROOT) / default_output_path
        self.verbose = verbose

        # Lazy-loaded components
        self._model_loader: Optional[ModelLoader] = None
        self._registrar: Optional[ObjectRegistrar] = None
        self._recognizer: Optional[ObjectRecognizer] = None
        self._text_encoder = None
        self._models_loaded = False

    def preload_models(self):
        """
        Preload all models at initialization time.

        Call this method during agent startup to avoid loading delays
        during the first conversation turn.
        """
        print("Loading models (this may take a moment)...")
        self._ensure_models_loaded(include_text_encoder=True)
        print("Models loaded successfully!")

    def is_models_loaded(self) -> bool:
        """Check if models are already loaded."""
        return self._models_loaded

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AgentService] {message}")

    def _ensure_models_loaded(self, include_text_encoder: bool = True):
        """
        Lazy load models when first needed.

        Args:
            include_text_encoder: Whether to load the text encoder
        """
        if self._models_loaded:
            return

        self._log("Loading models...")
        self._model_loader = ModelLoader(device=self.device, verbose=self.verbose)
        models = self._model_loader.load_all(include_text_encoder=include_text_encoder)

        self._registrar = ObjectRegistrar(
            detector=models['detector'],
            segmenter=models['segmenter'],
            extractor=models['extractor'],
            verbose=self.verbose
        )

        self._recognizer = ObjectRecognizer(
            detector=models['detector'],
            segmenter=models['segmenter'],
            extractor=models['extractor'],
            matcher=models['matcher'],
            verbose=self.verbose
        )

        if include_text_encoder:
            self._text_encoder = models.get('text_encoder')

        self._models_loaded = True
        self._log("Models loaded successfully")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to project root if not absolute."""
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def _get_image_paths(self, path: Path) -> List[Path]:
        """
        Get list of image paths from file or directory.

        Args:
            path: Path to file or directory

        Returns:
            List of image file paths
        """
        if path.is_file():
            return [path]
        elif path.is_dir():
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_paths.extend(sorted(path.glob(ext)))
            return image_paths
        return []

    def _load_existing_templates(self, templates_path: Path) -> bool:
        """
        Load existing templates if available.

        Args:
            templates_path: Path to templates

        Returns:
            True if templates were loaded, False otherwise
        """
        templates_file = f"{templates_path}_templates.pkl"
        if Path(templates_file).exists():
            try:
                self._registrar.load_templates(str(templates_path))
                self._log(f"Loaded existing templates from {templates_path}")
                return True
            except Exception as e:
                self._log(f"Warning: Failed to load existing templates: {e}")
        return False

    def register_object(
        self,
        image_path: str,
        instance_id: str,
        attributes: List[str],
        templates_path: Optional[str] = None
    ) -> RegistrationResponse:
        """
        Register an object from handheld images.

        This method:
        1. Loads images from the specified path
        2. Detects and segments the object in each image
        3. Extracts visual features using DINOv3-MGFA
        4. Encodes text attributes using SentenceTransformer
        5. Saves all features to the template library

        Args:
            image_path: Path to image file or directory
            instance_id: Unique identifier for the object
            attributes: List of text descriptions for the object
            templates_path: Optional custom templates path

        Returns:
            RegistrationResponse with registration results
        """
        self._ensure_models_loaded(include_text_encoder=True)

        # Resolve paths
        resolved_templates_path = (
            self._resolve_path(templates_path)
            if templates_path
            else self.default_templates_path
        )
        resolved_image_path = self._resolve_path(image_path)

        self._log(f"Registering object: {instance_id}")
        self._log(f"  Image path: {resolved_image_path}")
        self._log(f"  Attributes: {attributes}")

        try:
            # Get image files
            image_paths = self._get_image_paths(resolved_image_path)
            if not image_paths:
                return RegistrationResponse(
                    success=False,
                    instance_id=instance_id,
                    num_images_processed=0,
                    num_successful=0,
                    templates_path=str(resolved_templates_path),
                    error_message=f"No images found at {resolved_image_path}"
                )

            self._log(f"  Found {len(image_paths)} image(s)")

            # Load existing templates if available (for incremental addition)
            self._load_existing_templates(resolved_templates_path)

            # Register each image
            num_successful = 0
            for i, img_path in enumerate(image_paths):
                self._log(f"  Processing image {i+1}/{len(image_paths)}: {img_path.name}")

                image = Image.open(img_path).convert('RGB')

                if attributes and self._text_encoder:
                    result = self._registrar.register_with_attributes(
                        image=image,
                        instance_id=instance_id,
                        attributes=attributes,
                        text_encoder=self._text_encoder
                    )
                else:
                    result = self._registrar.register(
                        image=image,
                        instance_id=instance_id
                    )

                if result.success:
                    num_successful += 1
                    self._log(f"    Success: bbox={result.detection_bbox}")
                else:
                    self._log(f"    Failed: {result.error_message}")

            # Save templates
            resolved_templates_path.parent.mkdir(parents=True, exist_ok=True)
            self._registrar.save_templates(str(resolved_templates_path))
            self._log(f"  Templates saved to {resolved_templates_path}")

            return RegistrationResponse(
                success=num_successful > 0,
                instance_id=instance_id,
                num_images_processed=len(image_paths),
                num_successful=num_successful,
                templates_path=str(resolved_templates_path),
                has_text_embeddings=bool(attributes and self._text_encoder)
            )

        except Exception as e:
            self._log(f"  Error: {e}")
            return RegistrationResponse(
                success=False,
                instance_id=instance_id,
                num_images_processed=0,
                num_successful=0,
                templates_path=str(resolved_templates_path),
                error_message=str(e)
            )

    def recognize_object(
        self,
        query_image_path: str,
        text_query: str,
        templates_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        text_threshold: float = 0.3
    ) -> RecognitionResponse:
        """
        Recognize objects in query scene images.

        This method:
        1. Loads templates with visual and text embeddings
        2. Uses text query to retrieve candidate objects
        3. Detects and segments objects in the query image
        4. Matches against templates filtered by text similarity
        5. Returns recognized objects with confidence scores

        Args:
            query_image_path: Path to query image(s)
            text_query: Natural language query for retrieval
            templates_path: Path to templates
            confidence_threshold: Visual matching threshold
            text_threshold: Text similarity threshold

        Returns:
            RecognitionResponse with recognition results
        """
        self._ensure_models_loaded(include_text_encoder=True)

        # Resolve paths
        resolved_templates_path = (
            self._resolve_path(templates_path)
            if templates_path
            else self.default_templates_path
        )
        resolved_query_path = self._resolve_path(query_image_path)

        self._log(f"Recognizing objects with query: '{text_query}'")
        self._log(f"  Query images: {resolved_query_path}")
        self._log(f"  Templates: {resolved_templates_path}")

        try:
            # Load templates
            templates_file = f"{resolved_templates_path}_templates.pkl"
            if not Path(templates_file).exists():
                return RecognitionResponse(
                    success=False,
                    query=text_query,
                    num_images_searched=0,
                    matches=[],
                    error_message=f"Templates not found at {resolved_templates_path}"
                )

            self._recognizer.load_templates(str(resolved_templates_path))
            self._log(f"  Loaded templates: {self._recognizer.list_instances()}")

            # Get image files
            image_paths = self._get_image_paths(resolved_query_path)
            if not image_paths:
                return RecognitionResponse(
                    success=False,
                    query=text_query,
                    num_images_searched=0,
                    matches=[],
                    error_message=f"No images found at {resolved_query_path}"
                )

            self._log(f"  Found {len(image_paths)} query image(s)")

            all_matches = []
            all_recognitions_for_viz = []

            for i, img_path in enumerate(image_paths):
                self._log(f"  Processing image {i+1}/{len(image_paths)}: {img_path.name}")

                image = Image.open(img_path).convert('RGB')

                # Use text-based recognition if text encoder and embeddings available
                if self._text_encoder and self._recognizer.has_text_embeddings():
                    result = self._recognizer.recognize_with_text_query(
                        image=image,
                        query=text_query,
                        text_encoder=self._text_encoder,
                        text_threshold=text_threshold,
                        return_masks=True
                    )
                else:
                    result = self._recognizer.recognize(
                        image=image,
                        return_masks=True
                    )

                if result.success and result.recognitions:
                    # Filter by threshold and keep only the best match per image
                    valid_recognitions = [
                        rec for rec in result.recognitions
                        if rec['confidence'] >= confidence_threshold
                    ]

                    if valid_recognitions:
                        # Sort by confidence and keep only the best match
                        best_match = max(valid_recognitions, key=lambda x: x['confidence'])

                        match_info = {
                            'image': str(img_path.name),
                            'instance_id': best_match['instance_id'],
                            'visual_confidence': round(best_match['confidence'], 4),
                            'text_score': round(best_match.get('text_score', 0), 4) if best_match.get('text_score') else None,
                            'bbox': [round(x, 1) for x in best_match['bbox']]
                        }
                        all_matches.append(match_info)

                        self._log(
                            f"    Best match: {best_match['instance_id']} "
                            f"(visual={best_match['confidence']:.3f}, "
                            f"text={best_match.get('text_score', 'N/A')})"
                        )

                        # Save visualization with only the best match
                        viz_path = self._save_visualization(
                            image, [best_match], img_path, confidence_threshold
                        )

            # Determine visualization path (directory)
            viz_dir = self.default_output_path / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            return RecognitionResponse(
                success=True,
                query=text_query,
                num_images_searched=len(image_paths),
                matches=all_matches,
                visualization_path=str(viz_dir)
            )

        except Exception as e:
            self._log(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return RecognitionResponse(
                success=False,
                query=text_query,
                num_images_searched=0,
                matches=[],
                error_message=str(e)
            )

    def _save_visualization(
        self,
        image: Image.Image,
        recognitions: List[Dict],
        image_path: Path,
        threshold: float
    ) -> Optional[str]:
        """
        Save visualization of recognition results.

        Args:
            image: Query image
            recognitions: List of recognition results
            image_path: Original image path
            threshold: Confidence threshold

        Returns:
            Path to saved visualization, or None if failed
        """
        try:
            viz_dir = self.default_output_path / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Filter by threshold
            filtered = [r for r in recognitions if r['confidence'] >= threshold]

            if filtered:
                img_with_results = draw_recognition_result(
                    image=image,
                    recognitions=filtered,
                    confidence_threshold=threshold,
                    show_labels=True,
                    show_masks=True,
                    mask_alpha=0.3
                )

                save_path = viz_dir / f"{image_path.stem}_recognition.jpg"
                img_with_results.save(save_path, quality=95)
                self._log(f"    Visualization saved: {save_path}")
                return str(save_path)

        except Exception as e:
            self._log(f"    Failed to save visualization: {e}")

        return None

    def list_registered_objects(
        self,
        templates_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all registered objects in template library.

        Args:
            templates_path: Path to templates

        Returns:
            Dictionary with object information
        """
        resolved_templates_path = (
            self._resolve_path(templates_path)
            if templates_path
            else self.default_templates_path
        )

        templates_file = f"{resolved_templates_path}_templates.pkl"
        if not Path(templates_file).exists():
            return {
                'success': False,
                'error': f"No templates found at {resolved_templates_path}",
                'num_instances': 0,
                'objects': []
            }

        try:
            # Ensure models loaded for recognizer
            self._ensure_models_loaded()

            # Load templates
            self._recognizer.load_templates(str(resolved_templates_path))

            info = self._recognizer.get_template_info()
            text_embeddings = self._recognizer.get_text_embeddings()

            objects = []
            for inst_id in info.get('instance_ids', []):
                objects.append({
                    'instance_id': inst_id,
                    'num_templates': info['templates_per_instance'].get(inst_id, 0),
                    'has_text_embedding': inst_id in text_embeddings
                })

            return {
                'success': True,
                'num_instances': info.get('num_instances', 0),
                'templates_path': str(resolved_templates_path),
                'objects': objects
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'num_instances': 0,
                'objects': []
            }
