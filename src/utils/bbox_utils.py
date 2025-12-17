"""
Bounding box utility functions.
"""

import numpy as np


def seg_map_to_bbox(seg_map):
    """
    Convert a segmentation map to a bounding box.

    Args:
        seg_map: Binary segmentation mask [H, W]

    Returns:
        Bounding box [x0, y0, x1, y1]
    """
    h, w = seg_map.shape
    row_nonzero_inds = np.arange(h)[np.where(np.sum(seg_map, axis=1) > 0, True, False)]
    col_nonzero_inds = np.arange(w)[np.where(np.sum(seg_map, axis=0) > 0, True, False)]
    bbox_xyxy = [
        col_nonzero_inds[0],
        row_nonzero_inds[0],
        col_nonzero_inds[-1] + 1,
        row_nonzero_inds[-1] + 1
    ]
    return bbox_xyxy


def bbox_iou(left_bbox_xyxy, right_bbox_xyxy):
    """
    Compute IoU between two bounding boxes.

    Args:
        left_bbox_xyxy: First bbox [x0, y0, x1, y1]
        right_bbox_xyxy: Second bbox [x0, y0, x1, y1]

    Returns:
        IoU value (float)
    """
    int_x_min = max(left_bbox_xyxy[0], right_bbox_xyxy[0])
    int_x_max = min(left_bbox_xyxy[2], right_bbox_xyxy[2])
    int_w = max(0, int_x_max - int_x_min)
    int_y_min = max(left_bbox_xyxy[1], right_bbox_xyxy[1])
    int_y_max = min(left_bbox_xyxy[3], right_bbox_xyxy[3])
    int_h = max(0, int_y_max - int_y_min)
    int_area = int_w * int_h
    left_area = (left_bbox_xyxy[2] - left_bbox_xyxy[0]) * (left_bbox_xyxy[3] - left_bbox_xyxy[1])
    right_area = (right_bbox_xyxy[2] - right_bbox_xyxy[0]) * (right_bbox_xyxy[3] - right_bbox_xyxy[1])
    union_area = left_area + right_area - int_area
    return int_area / union_area


def seg_mask_iou(left_seg_mask, right_seg_mask):
    """
    Compute IoU between two segmentation masks.

    Args:
        left_seg_mask: First mask [H, W]
        right_seg_mask: Second mask [H, W]

    Returns:
        IoU value (float)
    """
    int_area = np.sum(left_seg_mask * right_seg_mask)
    union_area = np.sum(left_seg_mask) + np.sum(right_seg_mask) - int_area
    return int_area / union_area
