"""
Result analyzer for object detection evaluation.

This module provides functions to visualize detection results, save detailed logs,
and analyze failure cases.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Color palette for multiple object segmentation masks
MASK_COLORS = [
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (255, 0, 128),    # Pink
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
]


def draw_segmentation_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    draw_contour: bool = True,
    contour_width: int = 2
) -> Image.Image:
    """
    Draw a segmentation mask overlay on an image.

    Args:
        image: PIL Image to draw on
        mask: Binary segmentation mask (H, W) as numpy array
        color: RGB color tuple for the mask
        alpha: Transparency of the mask (0.0 to 1.0)
        draw_contour: Whether to draw mask contour
        contour_width: Width of contour line

    Returns:
        Image with mask overlay
    """
    # Create a copy to draw on
    img_array = np.array(image).copy()

    # Ensure mask is 2D
    if mask.ndim == 3:
        if mask.shape[0] == 1:  # (1, H, W)
            mask = mask[0]
        elif mask.shape[-1] == 1:  # (H, W, 1)
            mask = mask.squeeze(-1)

    # Ensure mask is binary
    if mask.dtype != bool:
        mask = mask > 0.5

    # Create colored mask overlay
    mask_overlay = np.zeros_like(img_array)
    mask_overlay[mask] = color

    # Blend mask with image
    img_array = np.where(
        mask[:, :, np.newaxis],
        (img_array * (1 - alpha) + mask_overlay * alpha).astype(np.uint8),
        img_array
    )

    result = Image.fromarray(img_array)

    # Draw contour if requested
    if draw_contour:
        result = _draw_mask_contour(result, mask, color, contour_width)

    return result


def _draw_mask_contour(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    width: int = 2
) -> Image.Image:
    """
    Draw the contour of a mask on the image.

    Args:
        image: PIL Image to draw on
        mask: Binary mask
        color: RGB color for contour
        width: Line width

    Returns:
        Image with contour drawn
    """
    from scipy import ndimage

    # Find contour by detecting edges
    # Dilate and erode to get boundary
    dilated = ndimage.binary_dilation(mask, iterations=width)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    contour = dilated & ~eroded

    # Draw contour
    img_array = np.array(image)
    img_array[contour] = color

    return Image.fromarray(img_array)


def draw_detection_with_segmentation(
    image: Image.Image,
    pred_bbox: Optional[List[float]],
    gt_bbox: List[float],
    mask: Optional[np.ndarray],
    iou: float,
    prompt_text: str = "",
    mask_color: Tuple[int, int, int] = (0, 255, 0),
    mask_alpha: float = 0.4,
    show_labels: bool = True
) -> Image.Image:
    """
    Draw detection result with bounding boxes and segmentation mask.

    Args:
        image: PIL Image to draw on
        pred_bbox: Predicted bounding box [x0, y0, x1, y1] or None
        gt_bbox: Ground truth bounding box [x0, y0, x1, y1]
        mask: Segmentation mask (H, W) or None
        iou: IoU value between prediction and ground truth
        prompt_text: The prompt used for detection
        mask_color: RGB color for segmentation mask
        mask_alpha: Transparency of mask
        show_labels: Whether to show labels

    Returns:
        Image with drawn boxes and mask
    """
    # First draw segmentation mask if available
    if mask is not None:
        image = draw_segmentation_mask(
            image, mask, mask_color, mask_alpha, draw_contour=True
        )

    # Then draw bounding boxes on top
    return draw_detection_result(
        image, pred_bbox, gt_bbox, iou, prompt_text, show_labels
    )


def draw_multi_object_with_segmentation(
    image: Image.Image,
    pred_bboxes: List[List[float]],
    gt_bboxes: List[List[float]],
    masks: Optional[List[np.ndarray]],
    matched_pairs: List[Tuple[int, int, float]],
    unmatched_pred_indices: List[int],
    unmatched_gt_indices: List[int],
    gt_labels: Optional[List[str]] = None,
    prompt_text: str = "",
    mask_alpha: float = 0.4,
    show_labels: bool = True
) -> Image.Image:
    """
    Draw multi-object detection with segmentation masks.

    Args:
        image: PIL Image to draw on
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground truth bounding boxes
        masks: List of segmentation masks for predictions, or None
        matched_pairs: List of (pred_idx, gt_idx, iou) tuples
        unmatched_pred_indices: Indices of false positive predictions
        unmatched_gt_indices: Indices of false negative ground truths
        gt_labels: Optional labels for ground truth boxes
        prompt_text: The prompt used for detection
        mask_alpha: Transparency of masks
        show_labels: Whether to show labels

    Returns:
        Image with drawn boxes and masks
    """
    # Draw masks first (if available)
    if masks:
        for i, mask in enumerate(masks):
            if mask is not None:
                # Use different colors for different objects
                color = MASK_COLORS[i % len(MASK_COLORS)]
                image = draw_segmentation_mask(
                    image, mask, color, mask_alpha, draw_contour=True
                )

    # Then draw bounding boxes on top
    return draw_multi_object_detection_result(
        image, pred_bboxes, gt_bboxes,
        matched_pairs, unmatched_pred_indices, unmatched_gt_indices,
        gt_labels, prompt_text, show_labels
    )


def draw_detection_result(
    image: Image.Image,
    pred_bbox: Optional[List[float]],
    gt_bbox: List[float],
    iou: float,
    prompt_text: str = "",
    show_labels: bool = True
) -> Image.Image:
    """
    Draw detection result with predicted and ground truth bounding boxes.

    Args:
        image: PIL Image to draw on
        pred_bbox: Predicted bounding box [x0, y0, x1, y1] or None if no detection
        gt_bbox: Ground truth bounding box [x0, y0, x1, y1]
        iou: IoU value between prediction and ground truth
        prompt_text: The prompt used for detection
        show_labels: Whether to show labels on boxes

    Returns:
        Image with drawn bounding boxes
    """
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw ground truth box (green)
    x0, y0, x1, y1 = gt_bbox
    draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
    if show_labels:
        draw.text((x0, y0 - 20), "GT", fill="green", font=font)

    # Draw predicted box (red if exists)
    if pred_bbox is not None:
        x0, y0, x1, y1 = pred_bbox
        color = "red" if iou < 0.5 else "orange" if iou < 0.75 else "lime"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        if show_labels:
            draw.text((x0, y0 - 40), f"Pred (IoU={iou:.2f})", fill=color, font=font)
    else:
        # No detection
        if show_labels:
            draw.text((10, 10), "No detection", fill="red", font=font)

    # Draw prompt text at bottom
    if prompt_text and show_labels:
        img_width, img_height = img_draw.size
        text = f"Prompt: {prompt_text}"
        draw.text((10, img_height - 25), text, fill="white", font=font_small)

    return img_draw


def draw_multi_object_detection_result(
    image: Image.Image,
    pred_bboxes: List[List[float]],
    gt_bboxes: List[List[float]],
    matched_pairs: List[Tuple[int, int, float]],
    unmatched_pred_indices: List[int],
    unmatched_gt_indices: List[int],
    gt_labels: Optional[List[str]] = None,
    prompt_text: str = "",
    show_labels: bool = True
) -> Image.Image:
    """
    Draw multi-object detection result with matched and unmatched boxes.

    Args:
        image: PIL Image to draw on
        pred_bboxes: List of predicted bounding boxes [[x0, y0, x1, y1], ...]
        gt_bboxes: List of ground truth bounding boxes [[x0, y0, x1, y1], ...]
        matched_pairs: List of (pred_idx, gt_idx, iou) tuples
        unmatched_pred_indices: Indices of unmatched predictions (false positives)
        unmatched_gt_indices: Indices of unmatched ground truths (false negatives)
        gt_labels: Optional list of labels for ground truth boxes
        prompt_text: The prompt used for detection
        show_labels: Whether to show labels on boxes

    Returns:
        Image with drawn bounding boxes and matching lines
    """
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw all ground truth boxes (green)
    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        x0, y0, x1, y1 = gt_bbox
        draw.rectangle([x0, y0, x1, y1], outline="green", width=3)

        if show_labels:
            label_text = f"GT"
            if gt_labels and gt_idx < len(gt_labels):
                label_text += f": {gt_labels[gt_idx]}"
            draw.text((x0, y0 - 18), label_text, fill="green", font=font_small)

    # Draw matched prediction boxes with connecting lines
    for pred_idx, gt_idx, iou in matched_pairs:
        if pred_idx >= len(pred_bboxes):
            continue

        pred_bbox = pred_bboxes[pred_idx]
        gt_bbox = gt_bboxes[gt_idx]

        x0, y0, x1, y1 = pred_bbox

        # Color code based on IoU: red (<0.5), orange (0.5-0.75), lime (>=0.75)
        if iou < 0.5:
            color = "red"
        elif iou < 0.75:
            color = "orange"
        else:
            color = "lime"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        if show_labels:
            draw.text((x0, y1 + 2), f"P: {iou:.2f}", fill=color, font=font_small)

        # Draw connecting line from pred center to gt center
        pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2)
        gt_center = ((gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2)
        draw.line([pred_center, gt_center], fill=color, width=2)

    # Draw unmatched predictions (false positives) in purple
    for pred_idx in unmatched_pred_indices:
        if pred_idx >= len(pred_bboxes):
            continue

        x0, y0, x1, y1 = pred_bboxes[pred_idx]
        draw.rectangle([x0, y0, x1, y1], outline="purple", width=3)

        if show_labels:
            draw.text((x0, y1 + 2), "FP", fill="purple", font=font_small)

    # Draw crosses on unmatched ground truths (false negatives)
    for gt_idx in unmatched_gt_indices:
        if gt_idx >= len(gt_bboxes):
            continue

        x0, y0, x1, y1 = gt_bboxes[gt_idx]
        # Draw X through the box
        draw.line([x0, y0, x1, y1], fill="red", width=2)
        draw.line([x1, y0, x0, y1], fill="red", width=2)

        if show_labels:
            draw.text((x0, y1 + 2), "FN", fill="red", font=font_small)

    # Draw statistics and prompt at bottom
    if show_labels:
        img_width, img_height = img_draw.size

        # Statistics
        stats_text = f"Matched: {len(matched_pairs)} | FP: {len(unmatched_pred_indices)} | FN: {len(unmatched_gt_indices)}"
        draw.text((10, img_height - 40), stats_text, fill="white", font=font_small)

        # Prompt
        if prompt_text:
            text = f"Prompt: {prompt_text}"
            draw.text((10, img_height - 25), text, fill="white", font=font_small)

    return img_draw


class ResultAnalyzer:
    """
    Analyzer for detection results.

    Handles saving visualizations, logs, and performing error analysis.
    """

    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize result analyzer.

        Args:
            output_dir: Base directory for saving results
            verbose: Whether to print verbose output
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create directory structure
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        self.logs_dir = self.output_dir / "logs"
        self.error_dir = self.output_dir / "error_analysis"

        for dir_path in [self.viz_dir, self.metrics_dir, self.logs_dir, self.error_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_visualization(
        self,
        image: Image.Image,
        pred_bbox: Optional[List[float]],
        gt_bbox: List[float],
        iou: float,
        prompt_strategy: str,
        prompt_text: str,
        image_id: str,
        is_failure: bool = False
    ):
        """
        Save visualization of detection result.

        Args:
            image: Original image
            pred_bbox: Predicted bounding box or None
            gt_bbox: Ground truth bounding box
            iou: IoU value
            prompt_strategy: Name of prompt strategy
            prompt_text: The actual prompt used
            image_id: Unique identifier for the image
            is_failure: Whether this is a failure case (IoU < 0.3)
        """
        # Create strategy-specific directory
        strategy_dir = self.viz_dir / prompt_strategy
        strategy_dir.mkdir(exist_ok=True)

        # Create success/failure subdirectories
        if is_failure:
            save_dir = strategy_dir / "failures"
        else:
            save_dir = strategy_dir / "success"
        save_dir.mkdir(exist_ok=True)

        # Draw detection result
        img_with_boxes = draw_detection_result(
            image, pred_bbox, gt_bbox, iou, prompt_text, show_labels=True
        )

        # Save image
        save_path = save_dir / f"{image_id}_iou{iou:.2f}.jpg"
        img_with_boxes.save(save_path, quality=95)

        if self.verbose:
            print(f"    Saved visualization: {save_path}")

    def save_detailed_log(
        self,
        prompt_strategy: str,
        results: List[Dict]
    ):
        """
        Save detailed log for a prompt strategy.

        Args:
            prompt_strategy: Name of prompt strategy
            results: List of result dictionaries for all images
        """
        log_path = self.logs_dir / f"{prompt_strategy}.log"

        with open(log_path, 'w') as f:
            f.write(f"Detailed Evaluation Log for '{prompt_strategy}'\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"Image {i}: {result.get('image_id', 'unknown')}\n")
                f.write(f"  Class: {result.get('class_name', 'unknown')}\n")
                f.write(f"  GT BBox: {result.get('gt_bbox', [])}\n")
                f.write(f"  Predicted BBox: {result.get('pred_bbox', 'None')}\n")
                f.write(f"  IoU: {result.get('iou', 0.0):.4f}\n")
                f.write(f"  Inference Time: {result.get('inference_time', 0.0):.4f}s\n")
                f.write(f"  Num Detections: {result.get('num_detections', 0)}\n")

                if result.get('error'):
                    f.write(f"  Error: {result['error']}\n")

                f.write("\n")

            # Summary statistics
            f.write("=" * 80 + "\n")
            f.write("Summary Statistics\n")
            f.write("=" * 80 + "\n")

            ious = [r.get('iou', 0.0) for r in results]
            times = [r.get('inference_time', 0.0) for r in results if r.get('inference_time')]

            f.write(f"Total Images: {len(results)}\n")
            f.write(f"Mean IoU: {np.mean(ious):.4f}\n")
            f.write(f"Median IoU: {np.median(ious):.4f}\n")
            f.write(f"Min IoU: {np.min(ious):.4f}\n")
            f.write(f"Max IoU: {np.max(ious):.4f}\n")
            f.write(f"Avg Inference Time: {np.mean(times):.4f}s\n") if times else None

            # Count by IoU ranges
            high_iou = sum(1 for iou in ious if iou >= 0.75)
            mid_iou = sum(1 for iou in ious if 0.5 <= iou < 0.75)
            low_iou = sum(1 for iou in ious if 0.3 <= iou < 0.5)
            fail_iou = sum(1 for iou in ious if iou < 0.3)

            f.write(f"\nDetections by IoU:\n")
            f.write(f"  IoU >= 0.75: {high_iou} ({high_iou/len(ious)*100:.1f}%)\n")
            f.write(f"  0.5 <= IoU < 0.75: {mid_iou} ({mid_iou/len(ious)*100:.1f}%)\n")
            f.write(f"  0.3 <= IoU < 0.5: {low_iou} ({low_iou/len(ious)*100:.1f}%)\n")
            f.write(f"  IoU < 0.3: {fail_iou} ({fail_iou/len(ious)*100:.1f}%)\n")

        if self.verbose:
            print(f"  Saved detailed log: {log_path}")

    def save_error_analysis(
        self,
        all_results: Dict[str, List[Dict]],
        iou_threshold: float = 0.3
    ):
        """
        Save error analysis for low IoU cases.

        Args:
            all_results: Dictionary mapping strategy names to result lists
            iou_threshold: IoU threshold below which cases are considered failures
        """
        # Collect failure cases
        failure_cases = []

        for strategy_name, results in all_results.items():
            for result in results:
                iou = result.get('iou', 0.0)
                if iou < iou_threshold:
                    failure_cases.append({
                        "strategy": strategy_name,
                        "image_id": result.get('image_id', 'unknown'),
                        "class_name": result.get('class_name', 'unknown'),
                        "iou": iou,
                        "gt_bbox": result.get('gt_bbox', []),
                        "pred_bbox": result.get('pred_bbox'),
                        "num_detections": result.get('num_detections', 0),
                        "error": result.get('error')
                    })

        # Save as JSON
        json_path = self.error_dir / "low_iou_cases.json"
        with open(json_path, 'w') as f:
            json.dump(failure_cases, f, indent=2)

        # Save summary text
        summary_path = self.error_dir / "failure_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Error Analysis: Cases with IoU < {iou_threshold}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Failure Cases: {len(failure_cases)}\n\n")

            # Group by strategy
            strategy_failures = {}
            for case in failure_cases:
                strategy = case['strategy']
                if strategy not in strategy_failures:
                    strategy_failures[strategy] = []
                strategy_failures[strategy].append(case)

            f.write("Failures by Strategy:\n")
            for strategy, cases in sorted(strategy_failures.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"  {strategy}: {len(cases)} failures\n")

            # Group by class
            f.write("\nFailures by Class:\n")
            class_failures = {}
            for case in failure_cases:
                class_name = case['class_name']
                if class_name not in class_failures:
                    class_failures[class_name] = []
                class_failures[class_name].append(case)

            for class_name, cases in sorted(class_failures.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                f.write(f"  {class_name}: {len(cases)} failures\n")

            # Detection statistics
            f.write("\nDetection Statistics:\n")
            no_detection = sum(1 for c in failure_cases if c['pred_bbox'] is None)
            wrong_detection = sum(1 for c in failure_cases if c['pred_bbox'] is not None)
            f.write(f"  No detection: {no_detection} ({no_detection/len(failure_cases)*100:.1f}%)\n")
            f.write(f"  Wrong detection: {wrong_detection} ({wrong_detection/len(failure_cases)*100:.1f}%)\n")

        if self.verbose:
            print(f"  Saved error analysis: {json_path}")
            print(f"  Saved failure summary: {summary_path}")

    def save_metrics_table(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        strategy_names: Optional[Dict[str, str]] = None
    ):
        """
        Save metrics as both CSV and Markdown tables.

        Args:
            metrics_dict: Dictionary mapping strategy names to metrics
            strategy_names: Optional display names for strategies
        """
        from .detection_metrics import format_metrics_csv, format_metrics_table

        # Save CSV
        csv_path = self.metrics_dir / "summary_table.csv"
        csv_content = format_metrics_csv(metrics_dict, strategy_names)
        with open(csv_path, 'w') as f:
            f.write(csv_content)

        # Save Markdown
        md_path = self.metrics_dir / "summary_table.md"
        md_content = format_metrics_table(metrics_dict, strategy_names)
        with open(md_path, 'w') as f:
            f.write(md_content)

        # Save JSON with full details
        json_path = self.metrics_dir / "per_prompt_details.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        if self.verbose:
            print(f"  Saved metrics CSV: {csv_path}")
            print(f"  Saved metrics Markdown: {md_path}")
            print(f"  Saved metrics JSON: {json_path}")

        # Also print to console
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(md_content)
        print("=" * 80 + "\n")

    def save_multi_object_visualization(
        self,
        image: Image.Image,
        pred_bboxes: List[List[float]],
        gt_bboxes: List[List[float]],
        matched_pairs: List[Tuple[int, int, float]],
        unmatched_pred_indices: List[int],
        unmatched_gt_indices: List[int],
        prompt_strategy: str,
        prompt_text: str,
        image_id: str,
        scene_name: str,
        gt_labels: Optional[List[str]] = None,
        is_failure: bool = False
    ):
        """
        Save visualization for multi-object detection result.

        Args:
            image: Original image
            pred_bboxes: List of predicted bounding boxes
            gt_bboxes: List of ground truth bounding boxes
            matched_pairs: List of (pred_idx, gt_idx, iou) tuples
            unmatched_pred_indices: Indices of false positive predictions
            unmatched_gt_indices: Indices of false negative ground truths
            prompt_strategy: Name of prompt strategy
            prompt_text: The actual prompt used
            image_id: Unique identifier for the image
            scene_name: Name of the scene (e.g., "floor1", "shelf")
            gt_labels: Optional labels for ground truth boxes
            is_failure: Whether this is a failure case
        """
        # Create scene-specific directory structure
        scene_dir = self.viz_dir / scene_name / prompt_strategy
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Create success/failure subdirectories
        if is_failure:
            save_dir = scene_dir / "failures"
        else:
            save_dir = scene_dir / "success"
        save_dir.mkdir(exist_ok=True)

        # Draw multi-object detection result
        img_with_boxes = draw_multi_object_detection_result(
            image, pred_bboxes, gt_bboxes,
            matched_pairs, unmatched_pred_indices, unmatched_gt_indices,
            gt_labels, prompt_text, show_labels=True
        )

        # Calculate average IoU for filename
        avg_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0.0

        # Save image
        save_path = save_dir / f"{image_id}_avgIoU{avg_iou:.2f}.jpg"
        img_with_boxes.save(save_path, quality=95)

        if self.verbose:
            print(f"    Saved visualization: {save_path}")

    def save_visualization_with_segmentation(
        self,
        image: Image.Image,
        pred_bbox: Optional[List[float]],
        gt_bbox: List[float],
        mask: Optional[np.ndarray],
        iou: float,
        prompt_strategy: str,
        prompt_text: str,
        image_id: str,
        is_failure: bool = False
    ):
        """
        Save visualization of detection result with segmentation mask.

        Args:
            image: Original image
            pred_bbox: Predicted bounding box or None
            gt_bbox: Ground truth bounding box
            mask: Segmentation mask (H, W) or None
            iou: IoU value
            prompt_strategy: Name of prompt strategy
            prompt_text: The actual prompt used
            image_id: Unique identifier for the image
            is_failure: Whether this is a failure case
        """
        # Create strategy-specific directory
        strategy_dir = self.viz_dir / prompt_strategy
        strategy_dir.mkdir(exist_ok=True)

        # Create success/failure subdirectories
        if is_failure:
            save_dir = strategy_dir / "failures"
        else:
            save_dir = strategy_dir / "success"
        save_dir.mkdir(exist_ok=True)

        # Draw detection result with segmentation
        img_with_result = draw_detection_with_segmentation(
            image, pred_bbox, gt_bbox, mask, iou, prompt_text, show_labels=True
        )

        # Save image
        save_path = save_dir / f"{image_id}_iou{iou:.2f}.jpg"
        img_with_result.save(save_path, quality=95)

        if self.verbose:
            print(f"    Saved visualization with segmentation: {save_path}")

    def save_multi_object_visualization_with_segmentation(
        self,
        image: Image.Image,
        pred_bboxes: List[List[float]],
        gt_bboxes: List[List[float]],
        masks: Optional[List[np.ndarray]],
        matched_pairs: List[Tuple[int, int, float]],
        unmatched_pred_indices: List[int],
        unmatched_gt_indices: List[int],
        prompt_strategy: str,
        prompt_text: str,
        image_id: str,
        scene_name: str,
        gt_labels: Optional[List[str]] = None,
        is_failure: bool = False
    ):
        """
        Save visualization for multi-object detection with segmentation masks.

        Args:
            image: Original image
            pred_bboxes: List of predicted bounding boxes
            gt_bboxes: List of ground truth bounding boxes
            masks: List of segmentation masks for predictions
            matched_pairs: List of (pred_idx, gt_idx, iou) tuples
            unmatched_pred_indices: Indices of false positive predictions
            unmatched_gt_indices: Indices of false negative ground truths
            prompt_strategy: Name of prompt strategy
            prompt_text: The actual prompt used
            image_id: Unique identifier for the image
            scene_name: Name of the scene
            gt_labels: Optional labels for ground truth boxes
            is_failure: Whether this is a failure case
        """
        # Create scene-specific directory structure
        scene_dir = self.viz_dir / scene_name / prompt_strategy
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Create success/failure subdirectories
        if is_failure:
            save_dir = scene_dir / "failures"
        else:
            save_dir = scene_dir / "success"
        save_dir.mkdir(exist_ok=True)

        # Draw multi-object detection result with segmentation
        img_with_result = draw_multi_object_with_segmentation(
            image, pred_bboxes, gt_bboxes, masks,
            matched_pairs, unmatched_pred_indices, unmatched_gt_indices,
            gt_labels, prompt_text, show_labels=True
        )

        # Calculate average IoU for filename
        avg_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0.0

        # Save image
        save_path = save_dir / f"{image_id}_avgIoU{avg_iou:.2f}.jpg"
        img_with_result.save(save_path, quality=95)

        if self.verbose:
            print(f"    Saved visualization with segmentation: {save_path}")


def draw_recognition_result(
    image: Image.Image,
    recognitions: List[Dict],
    confidence_threshold: float = 0.5,
    show_labels: bool = True,
    show_masks: bool = True,
    mask_alpha: float = 0.3
) -> Image.Image:
    """
    Draw recognition results with bounding boxes, instance IDs, and confidence scores.

    Args:
        image: PIL Image to draw on
        recognitions: List of recognition dictionaries with keys:
            - bbox: [x0, y0, x1, y1]
            - instance_id: str
            - confidence: float
            - mask: np.ndarray (optional)
            - seg_score: float (optional)
        confidence_threshold: Threshold for high/low confidence visualization
        show_labels: Whether to show text labels
        show_masks: Whether to draw segmentation masks
        mask_alpha: Transparency of masks

    Returns:
        Image with drawn recognition results
    """
    # Draw masks first if available
    if show_masks:
        for i, rec in enumerate(recognitions):
            mask = rec.get('mask')
            if mask is not None:
                color = MASK_COLORS[i % len(MASK_COLORS)]
                image = draw_segmentation_mask(
                    image, mask, color, mask_alpha, draw_contour=True, contour_width=2
                )

    # Create a copy to draw boxes and labels
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw each recognition
    for i, rec in enumerate(recognitions):
        bbox = rec['bbox']
        instance_id = rec['instance_id']
        confidence = rec['confidence']

        x0, y0, x1, y1 = bbox

        # Color based on confidence
        if confidence >= confidence_threshold:
            color = "lime"  # High confidence
        else:
            color = "orange"  # Low confidence

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        if show_labels:
            # Draw instance ID and confidence
            label = f"{instance_id}"
            conf_text = f"{confidence:.3f}"

            # Draw label background
            label_bbox = draw.textbbox((x0, y0 - 40), label, font=font)
            draw.rectangle(label_bbox, fill=color)
            draw.text((x0, y0 - 40), label, fill="black", font=font)

            # Draw confidence score
            draw.text((x0, y0 - 20), f"Conf: {conf_text}", fill=color, font=font_small)

            # Draw segmentation score if available
            if 'seg_score' in rec:
                seg_score = rec['seg_score']
                draw.text((x0, y1 + 2), f"Seg: {seg_score:.3f}", fill=color, font=font_small)

    return img_draw


def draw_recognition_result_with_top_k(
    image: Image.Image,
    recognitions: List[Dict],
    confidence_threshold: float = 0.5,
    show_top_k: int = 3,
    show_labels: bool = True,
    show_masks: bool = True,
    mask_alpha: float = 0.3
) -> Image.Image:
    """
    Draw recognition results with top-k alternative matches displayed.

    Args:
        image: PIL Image to draw on
        recognitions: List of recognition dictionaries
        confidence_threshold: Threshold for high/low confidence
        show_top_k: Number of top-k matches to display
        show_labels: Whether to show text labels
        show_masks: Whether to draw segmentation masks
        mask_alpha: Transparency of masks

    Returns:
        Image with drawn recognition results
    """
    # Draw masks first if available
    if show_masks:
        for i, rec in enumerate(recognitions):
            mask = rec.get('mask')
            if mask is not None:
                color = MASK_COLORS[i % len(MASK_COLORS)]
                image = draw_segmentation_mask(
                    image, mask, color, mask_alpha, draw_contour=True, contour_width=2
                )

    # Create a copy to draw boxes and labels
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw each recognition
    for i, rec in enumerate(recognitions):
        bbox = rec['bbox']
        instance_id = rec['instance_id']
        confidence = rec['confidence']
        top_k_matches = rec.get('top_k_matches', [])

        x0, y0, x1, y1 = bbox

        # Color based on confidence
        if confidence >= confidence_threshold:
            color = "lime"
        else:
            color = "orange"

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        if show_labels:
            # Draw instance ID (best match)
            label = f"#{i+1}: {instance_id}"
            conf_text = f"{confidence:.3f}"

            # Draw on top of box
            draw.text((x0, y0 - 20), f"{label} ({conf_text})", fill=color, font=font)

            # Draw top-k matches inside the box (if space available)
            box_height = y1 - y0
            box_width = x1 - x0

            if box_height > 80 and box_width > 100:
                y_offset = y0 + 5
                draw.text((x0 + 5, y_offset), "Top matches:", fill="white", font=font_small)
                y_offset += 15

                for rank, (match_id, score) in enumerate(top_k_matches[:show_top_k]):
                    rank_text = f"  {rank+1}. {match_id}: {score:.3f}"
                    draw.text((x0 + 5, y_offset), rank_text, fill="white", font=font_small)
                    y_offset += 12

                    if y_offset > y1 - 10:
                        break

    return img_draw


class RecognitionVisualizer:
    """
    Visualizer for object recognition results.

    Handles saving visualizations and reports for recognition tasks.
    """

    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize recognition visualizer.

        Args:
            output_dir: Base directory for saving results
            verbose: Whether to print verbose output
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create directory structure
        self.viz_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.viz_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_recognition_visualization(
        self,
        image: Image.Image,
        recognitions: List[Dict],
        image_id: str,
        confidence_threshold: float = 0.5,
        show_top_k: bool = False,
        show_masks: bool = True
    ):
        """
        Save visualization of recognition result.

        Args:
            image: Original image
            recognitions: List of recognition dictionaries
            image_id: Unique identifier for the image
            confidence_threshold: Confidence threshold
            show_top_k: Whether to show top-k matches
            show_masks: Whether to show segmentation masks
        """
        # Choose visualization function
        if show_top_k:
            img_with_results = draw_recognition_result_with_top_k(
                image=image,
                recognitions=recognitions,
                confidence_threshold=confidence_threshold,
                show_top_k=3,
                show_labels=True,
                show_masks=show_masks,
                mask_alpha=0.3
            )
        else:
            img_with_results = draw_recognition_result(
                image=image,
                recognitions=recognitions,
                confidence_threshold=confidence_threshold,
                show_labels=True,
                show_masks=show_masks,
                mask_alpha=0.3
            )

        # Save image
        save_path = self.viz_dir / f"{image_id}_recognition.jpg"
        img_with_results.save(save_path, quality=95)

        if self.verbose:
            print(f"  Saved visualization: {save_path}")

    def save_recognition_report(
        self,
        results: List[Dict],
        report_name: str = "recognition_report"
    ):
        """
        Save detailed recognition report.

        Args:
            results: List of result dictionaries for all images
            report_name: Name for the report file
        """
        report_path = self.reports_dir / f"{report_name}.txt"

        with open(report_path, 'w') as f:
            f.write("Object Recognition Report\n")
            f.write("=" * 80 + "\n\n")

            total_images = len(results)
            total_detections = sum(r.get('num_detections', 0) for r in results)
            total_recognitions = sum(len(r.get('recognitions', [])) for r in results)

            f.write(f"Summary:\n")
            f.write(f"  Total images: {total_images}\n")
            f.write(f"  Total detections: {total_detections}\n")
            f.write(f"  Total recognitions: {total_recognitions}\n")
            f.write(f"  Avg detections per image: {total_detections/total_images:.2f}\n\n")

            f.write("=" * 80 + "\n\n")

            # Per-image details
            for i, result in enumerate(results, 1):
                image_id = result.get('image_id', f'image_{i}')
                num_detections = result.get('num_detections', 0)
                recognitions = result.get('recognitions', [])

                f.write(f"Image {i}: {image_id}\n")
                f.write(f"  Detections: {num_detections}\n")

                if recognitions:
                    f.write(f"  Recognitions:\n")
                    for j, rec in enumerate(recognitions):
                        instance_id = rec.get('instance_id', 'unknown')
                        confidence = rec.get('confidence', 0.0)
                        bbox = rec.get('bbox', [])

                        f.write(f"    {j+1}. {instance_id} (conf={confidence:.4f})\n")
                        f.write(f"       BBox: {[f'{x:.1f}' for x in bbox]}\n")

                        if 'top_k_matches' in rec:
                            f.write(f"       Top-3 matches:\n")
                            for rank, (match_id, score) in enumerate(rec['top_k_matches'][:3], 1):
                                f.write(f"         {rank}. {match_id}: {score:.4f}\n")
                else:
                    f.write(f"  No recognitions\n")

                f.write("\n")

        if self.verbose:
            print(f"  Saved report: {report_path}")

    def save_recognition_summary_json(
        self,
        results: List[Dict],
        summary_name: str = "recognition_summary"
    ):
        """
        Save recognition summary as JSON.

        Args:
            results: List of result dictionaries
            summary_name: Name for the summary file
        """
        import json

        summary_path = self.reports_dir / f"{summary_name}.json"

        # Build summary
        summary = {
            'total_images': len(results),
            'total_detections': sum(r.get('num_detections', 0) for r in results),
            'total_recognitions': sum(len(r.get('recognitions', [])) for r in results),
            'results': []
        }

        for result in results:
            image_result = {
                'image_id': result.get('image_id', 'unknown'),
                'num_detections': result.get('num_detections', 0),
                'recognitions': []
            }

            for rec in result.get('recognitions', []):
                rec_data = {
                    'instance_id': rec.get('instance_id'),
                    'confidence': float(rec.get('confidence', 0.0)),
                    'bbox': [float(x) for x in rec.get('bbox', [])],
                    'seg_score': float(rec.get('seg_score', 0.0)) if 'seg_score' in rec else None,
                }

                if 'top_k_matches' in rec:
                    rec_data['top_k_matches'] = [
                        {'instance_id': match_id, 'score': float(score)}
                        for match_id, score in rec['top_k_matches']
                    ]

                image_result['recognitions'].append(rec_data)

            summary['results'].append(image_result)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"  Saved JSON summary: {summary_path}")
