"""
=============================================================================
AeroVision AI - Object Detection Foundations: Module 02
=============================================================================
Non-Maximum Suppression (NMS) Algorithm From Scratch

Learning Objectives:
1. Understand why deep detectors produce dense redundant candidate boxes.
2. Implement Greedy Non-Maximum Suppression (NMS) in pure NumPy from scratch.
3. Compare custom NMS outputs against PyTorch torchvision.ops.nms for mathematical parity.
4. Understand Class-Agnostic vs Class-Specific NMS in multi-class aerial drone scenes.
5. Contrast Hard NMS with Soft NMS for dense aerial object clusters.
"""

import numpy as np
import torch
import torchvision.ops

def nms_numpy_from_scratch(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> np.ndarray:
    """
    Greedy Non-Maximum Suppression implemented in pure NumPy.
    
    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2]
        scores: Array of shape (N,) containing confidence scores
        iou_threshold: Overlap threshold above which redundant boxes are suppressed
        score_threshold: Minimum confidence score to be considered
        
    Returns:
        keep_indices: 1D array of indices corresponding to retained boxes
    """
    # Step 1: Filter out boxes below confidence score threshold
    valid_mask = scores >= score_threshold
    if not np.any(valid_mask):
        return np.empty((0,), dtype=int)
        
    indices = np.where(valid_mask)[0]
    boxes = boxes[indices]
    scores = scores[indices]

    # Precalculate box areas: Area = (x2 - x1) * (y2 - y1)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Step 2: Sort candidate boxes descending by confidence score
    order = scores.argsort()[::-1]

    keep = []

    # Step 3: Greedy suppression loop
    while order.size > 0:
        # Pick the box with the highest score
        i = order[0]
        keep.append(indices[i])

        if order.size == 1:
            break

        # Compute intersection of highest-score box with all remaining candidates
        remaining = order[1:]
        inter_x1 = np.maximum(x1[i], x1[remaining])
        inter_y1 = np.maximum(y1[i], y1[remaining])
        inter_x2 = np.minimum(x2[i], x2[remaining])
        inter_y2 = np.minimum(y2[i], y2[remaining])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        # Compute IoU: Intersection / (Area_i + Area_remaining - Intersection)
        union = areas[i] + areas[remaining] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        # Retain only candidate boxes with IoU < iou_threshold
        surviving = np.where(iou < iou_threshold)[0]
        order = order[surviving + 1]

    return np.array(keep, dtype=int)

def demonstrate_nms_parity():
    print("=" * 70)
    print("STEP 1: NON-MAXIMUM SUPPRESSION (NMS) VERIFICATION")
    print("=" * 70)

    # Generate synthetic candidate predictions for 2 vehicles in a drone scene
    # Vehicle 1: Centered at ~ (100, 100) with 4 overlapping predictions
    # Vehicle 2: Centered at ~ (300, 300) with 3 overlapping predictions
    np.random.seed(42)

    boxes = np.array([
        # Vehicle 1 candidates
        [98.0, 95.0, 152.0, 148.0],   # Box 0: High score (0.92)
        [100.0, 100.0, 150.0, 150.0], # Box 1: Medium score (0.85) - heavy overlap
        [95.0, 92.0, 155.0, 152.0],   # Box 2: Lower score (0.78) - heavy overlap
        [90.0, 90.0, 160.0, 160.0],   # Box 3: Low score (0.45) - heavy overlap
        # Vehicle 2 candidates
        [295.0, 290.0, 355.0, 360.0], # Box 4: High score (0.88)
        [300.0, 300.0, 350.0, 350.0], # Box 5: Medium score (0.75) - heavy overlap
        [305.0, 295.0, 360.0, 355.0], # Box 6: Low score (0.60) - heavy overlap
        # Low confidence noise (should be discarded by score threshold)
        [50.0, 50.0, 80.0, 80.0]      # Box 7: Sub-threshold noise (0.15)
    ], dtype=np.float32)

    scores = np.array([0.92, 0.85, 0.78, 0.45, 0.88, 0.75, 0.60, 0.15], dtype=np.float32)

    iou_thresh = 0.50
    score_thresh = 0.25

    print(f"Total Raw Candidates : {len(boxes)}")
    print(f"NMS Parameters       : IoU Thresh = {iou_thresh}, Score Thresh = {score_thresh}")

    # 1. Custom NumPy NMS
    custom_keep = nms_numpy_from_scratch(boxes, scores, iou_thresh, score_thresh)

    # 2. PyTorch TorchVision NMS
    # Note: torchvision.ops.nms takes boxes and scores as torch.Tensor
    t_boxes = torch.from_numpy(boxes)
    t_scores = torch.from_numpy(scores)
    # Apply score threshold first
    score_mask = t_scores >= score_thresh
    tv_indices = torch.where(score_mask)[0]
    tv_keep_relative = torchvision.ops.nms(t_boxes[score_mask], t_scores[score_mask], iou_thresh)
    torchvision_keep = tv_indices[tv_keep_relative].cpu().numpy()

    print(f"\nCustom NumPy NMS Retained Indices    : {custom_keep.tolist()}")
    print(f"TorchVision NMS Retained Indices     : {torchvision_keep.tolist()}")

    # Assert exact index equality
    assert np.array_equal(custom_keep, torchvision_keep), "Custom NMS does not match TorchVision NMS!"
    print(f"[SUCCESS] Custom NMS perfectly matched TorchVision NMS!")

    print(f"\nRetained True Detections:")
    for idx in custom_keep:
        print(f"  Box #{idx}: Box={boxes[idx].tolist()}, Conf={scores[idx]:.2f}")

    print("\n" + "=" * 70)
    print("STEP 2: CLASS-AGNOSTIC VS CLASS-SPECIFIC NMS IN AERIAL SCENES")
    print("=" * 70)
    print("1. Class-Specific NMS (Default):")
    print("   NMS is executed independently per object class. A 'car' and a 'person' standing next to each other")
    print("   will NOT suppress each other, even if their bounding boxes have high overlap.")
    print("2. Class-Agnostic NMS:")
    print("   All classes are suppressed together. Useful when a single object is double-detected as both")
    print("   a 'car' and a 'truck'.")
    print("3. Trick for Batched / Class-Specific NMS:")
    print("   Offset box coordinates by `class_id * max_coordinate`. Boxes of different classes will have zero overlap,")
    print("   enabling a single vectorized NMS call!")
    print()

def main():
    demonstrate_nms_parity()
    print("=" * 70)
    print("PHASE 5 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
