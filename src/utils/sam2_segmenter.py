"""
SAM2 segmenter wrapper for object segmentation using bounding box prompts.

This module provides a high-level interface to the SAM2 model for segmenting
objects based on detected bounding boxes.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Default checkpoint and config paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class SAM2Segmenter:
    """
    Wrapper for SAM2 model to segment objects using bounding box prompts.

    This class provides a simplified interface for using SAM2 with bounding boxes
    as prompts to generate segmentation masks.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_cfg: str = DEFAULT_CONFIG,
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize SAM2 segmenter.

        Args:
            checkpoint_path: Path to SAM2 checkpoint file. If None, uses default.
            model_cfg: Path to model config file (relative to sam2 directory)
            device: Device to run model on ('cuda' or 'cpu')
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.device = device

        if checkpoint_path is None:
            checkpoint_path = str(DEFAULT_CHECKPOINT)

        if self.verbose:
            print(f"Initializing SAM2 model from {checkpoint_path}...")
            print(f"Config: {model_cfg}")
            print(f"Device: {device}")

        # Build model
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        if self.verbose:
            print("SAM2 model initialized successfully!")

    def segment_single(
        self,
        image: Union[Image.Image, np.ndarray],
        bbox: List[float],
        multimask_output: bool = False,
        return_timing: bool = True
    ) -> Dict:
        """
        Segment a single object using a bounding box prompt.

        Args:
            image: PIL Image or numpy array (RGB)
            bbox: Bounding box [x0, y0, x1, y1]
            multimask_output: If True, returns 3 candidate masks with scores
            return_timing: Whether to include timing information

        Returns:
            Dictionary containing:
                - success: Whether segmentation succeeded
                - mask: Binary segmentation mask (H, W) or masks (N, H, W)
                - score: Confidence score(s)
                - bbox: Input bounding box
                - inference_time: Time taken for inference
        """
        start_time = time.time() if return_timing else None

        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

            # Ensure RGB format
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            # Set image in predictor
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.predictor.set_image(image_array)

                # Prepare bbox input
                input_box = np.array(bbox)

                # Predict mask
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=multimask_output
                )

            inference_time = time.time() - start_time if return_timing else None

            # Return best mask if not multimask
            if not multimask_output:
                mask = masks[0]
                score = float(scores[0])
            else:
                mask = masks
                score = [float(s) for s in scores]

            return {
                "success": True,
                "mask": mask,
                "score": score,
                "bbox": bbox,
                "inference_time": inference_time
            }

        except Exception as e:
            inference_time = time.time() - start_time if return_timing and start_time else None
            return {
                "success": False,
                "mask": None,
                "score": 0.0,
                "bbox": bbox,
                "inference_time": inference_time,
                "error": str(e)
            }

    def segment_multiple(
        self,
        image: Union[Image.Image, np.ndarray],
        bboxes: List[List[float]],
        return_timing: bool = True
    ) -> Dict:
        """
        Segment multiple objects using bounding box prompts.

        Args:
            image: PIL Image or numpy array (RGB)
            bboxes: List of bounding boxes [[x0, y0, x1, y1], ...]
            return_timing: Whether to include timing information

        Returns:
            Dictionary containing:
                - success: Whether segmentation succeeded
                - masks: List of binary segmentation masks
                - scores: List of confidence scores
                - bboxes: Input bounding boxes
                - inference_time: Time taken for inference
        """
        start_time = time.time() if return_timing else None

        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

            # Ensure RGB format
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            masks = []
            scores = []

            # Set image once for all bboxes
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.predictor.set_image(image_array)

                # Process each bbox
                for bbox in bboxes:
                    input_box = np.array(bbox)

                    mask_preds, score_preds, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,
                        multimask_output=False
                    )

                    masks.append(mask_preds[0])
                    scores.append(float(score_preds[0]))

            inference_time = time.time() - start_time if return_timing else None

            return {
                "success": True,
                "masks": masks,
                "scores": scores,
                "bboxes": bboxes,
                "inference_time": inference_time,
                "num_segments": len(masks)
            }

        except Exception as e:
            inference_time = time.time() - start_time if return_timing and start_time else None
            return {
                "success": False,
                "masks": [],
                "scores": [],
                "bboxes": bboxes,
                "inference_time": inference_time,
                "num_segments": 0,
                "error": str(e)
            }

    def segment_batch(
        self,
        image: Union[Image.Image, np.ndarray],
        bboxes: List[List[float]],
        return_timing: bool = True
    ) -> Dict:
        """
        Segment multiple objects in batch mode (more efficient for many boxes).

        Args:
            image: PIL Image or numpy array (RGB)
            bboxes: List of bounding boxes [[x0, y0, x1, y1], ...]
            return_timing: Whether to include timing information

        Returns:
            Same format as segment_multiple
        """
        start_time = time.time() if return_timing else None

        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

            # Ensure RGB format
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.predictor.set_image(image_array)

                # Batch input
                input_boxes = np.array(bboxes)

                # Predict all masks at once
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False
                )

            inference_time = time.time() - start_time if return_timing else None

            # Convert to list format
            masks_list = [masks[i] for i in range(len(bboxes))]
            scores_list = [float(scores[i]) for i in range(len(bboxes))]

            return {
                "success": True,
                "masks": masks_list,
                "scores": scores_list,
                "bboxes": bboxes,
                "inference_time": inference_time,
                "num_segments": len(masks_list)
            }

        except Exception as e:
            # Fallback to sequential processing if batch fails
            if self.verbose:
                print(f"Batch segmentation failed, falling back to sequential: {e}")
            return self.segment_multiple(image, bboxes, return_timing)


def get_default_checkpoint_path() -> str:
    """Get the default SAM2 checkpoint path."""
    return str(DEFAULT_CHECKPOINT)
