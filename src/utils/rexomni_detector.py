"""
Rex-Omni detector wrapper for object detection.

This module provides a high-level interface to the Rex-Omni model for detecting
objects using various text prompting strategies.
"""

import time
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Import from pip-installed rex_omni package
from rex_omni import RexOmniWrapper


# Prompt strategies for handheld object detection
PROMPT_STRATEGIES = {
    "direct_object": "object in hand",
    "held_item": "item held by hand",
    "handheld": "handheld object",
    "being_held": "object being held",
    "grasped": "object grasped by hand",
    "hand_holding": "hand holding object",
    "person_holding": "person holding something",
}


class RexOmniDetector:
    """
    Wrapper for Rex-Omni model to detect objects.

    This class provides a simplified interface for using Rex-Omni with various
    text prompting strategies.
    """

    def __init__(
        self,
        model_path: str = "models/Rex-Omni",
        backend: str = "transformers",
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 2560 * 28 * 28,
        max_tokens: int = 4096,
        repetition_penalty: float = 1.05,
        verbose: bool = False
    ):
        """
        Initialize Rex-Omni detector.

        Args:
            model_path: Path to Rex-Omni model weights
            backend: Backend to use ('transformers' or 'vllm')
            min_pixels: Minimum image size in pixels
            max_pixels: Maximum image size in pixels
            max_tokens: Maximum number of tokens to generate
            repetition_penalty: Repetition penalty
            verbose: Whether to print verbose output

        Note:
            To avoid warnings about conflicting generation parameters, we need to
            modify the generation config after initialization.
        """
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing Rex-Omni model from {model_path}...")
            print(f"Backend: {backend}")

        self.rex = RexOmniWrapper(
            model_path=model_path,
            backend=backend,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

        # Fix generation parameters to avoid warnings about conflicting parameters
        # RexOmniWrapper uses do_sample=False (greedy decoding) but sets temperature/top_p/top_k
        # These parameters are only used in sample-based generation modes
        # Set them to default/neutral values to avoid conflicting with do_sample=False
        self.rex.temperature = 1.0  # Default value, won't affect greedy decoding
        self.rex.top_p = 1.0  # Default value (no nucleus sampling)
        self.rex.top_k = 50  # Default transformers value

        if self.verbose:
            print("Rex-Omni model initialized successfully!")

    def detect(
        self,
        image: Image.Image,
        prompt_text: str,
        return_timing: bool = True
    ) -> Dict:
        """
        Detect objects in an image using a text prompt.

        Args:
            image: PIL Image to process
            prompt_text: Text prompt describing what to detect
            return_timing: Whether to include timing information

        Returns:
            Dictionary containing:
                - success: Whether detection succeeded
                - predictions: List of detected bounding boxes [x0, y0, x1, y1]
                - prompt: The prompt used
                - inference_time: Time taken for inference
                - num_detections: Number of objects detected
                - raw_result: Raw result from Rex-Omni
        """
        start_time = time.time() if return_timing else None

        try:
            # Run detection
            results = self.rex.inference(
                images=image,
                task="detection",
                categories=[prompt_text]
            )

            inference_time = time.time() - start_time if return_timing else None

            # Parse results
            if not results or len(results) == 0:
                return {
                    "success": False,
                    "predictions": [],
                    "prompt": prompt_text,
                    "inference_time": inference_time,
                    "num_detections": 0,
                    "raw_result": None,
                    "error": "No results returned from model"
                }

            result = results[0]

            # Extract predictions
            predictions = []
            if result.get("success", False):
                extracted = result.get("extracted_predictions", {})
                if prompt_text in extracted:
                    for pred in extracted[prompt_text]:
                        if pred["type"] == "box":
                            predictions.append(pred["coords"])

            return {
                "success": result.get("success", False),
                "predictions": predictions,
                "prompt": prompt_text,
                "inference_time": inference_time,
                "num_detections": len(predictions),
                "raw_result": result
            }

        except Exception as e:
            inference_time = time.time() - start_time if return_timing and start_time else None
            return {
                "success": False,
                "predictions": [],
                "prompt": prompt_text,
                "inference_time": inference_time,
                "num_detections": 0,
                "raw_result": None,
                "error": str(e)
            }

    def detect_with_all_prompts(
        self,
        image: Image.Image,
        prompt_strategies: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict]:
        """
        Detect objects using all prompt strategies.

        Args:
            image: PIL Image to process
            prompt_strategies: Dictionary of {strategy_name: prompt_text}.
                             If None, uses default PROMPT_STRATEGIES.

        Returns:
            Dictionary mapping strategy names to detection results
        """
        if prompt_strategies is None:
            prompt_strategies = PROMPT_STRATEGIES

        results = {}
        for strategy_name, prompt_text in prompt_strategies.items():
            if self.verbose:
                print(f"  Testing prompt strategy: {strategy_name} ('{prompt_text}')")

            result = self.detect(image, prompt_text, return_timing=True)
            results[strategy_name] = result

        return results

    def get_best_prediction(
        self,
        predictions: List[List[float]],
        gt_bbox: List[float],
        iou_func
    ) -> Tuple[Optional[List[float]], float]:
        """
        Select the best prediction based on IoU with ground truth.

        Args:
            predictions: List of predicted bounding boxes [[x0,y0,x1,y1], ...]
            gt_bbox: Ground truth bounding box [x0,y0,x1,y1]
            iou_func: Function to compute IoU between two boxes

        Returns:
            Tuple of (best_bbox, best_iou)
        """
        if not predictions:
            return None, 0.0

        best_bbox = None
        best_iou = 0.0

        for pred_bbox in predictions:
            iou = iou_func(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_bbox = pred_bbox

        return best_bbox, best_iou


def get_prompt_strategies() -> Dict[str, str]:
    """
    Get the default prompt strategies for object detection.

    Returns:
        Dictionary mapping strategy names to prompt texts
    """
    return PROMPT_STRATEGIES.copy()
