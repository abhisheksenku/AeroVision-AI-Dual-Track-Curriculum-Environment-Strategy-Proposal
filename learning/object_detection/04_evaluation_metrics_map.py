"""
=============================================================================
AeroVision AI - Object Detection Foundations: Module 04
=============================================================================
Evaluation Metrics: Precision, Recall, PR Curves, and mAP@50 / mAP@50:95

Learning Objectives:
1. Understand why standard classification accuracy fails for object detection.
2. Define True Positives, False Positives, and False Negatives using IoU matching.
3. Compute Precision, Recall, and construct the Precision-Recall (PR) Curve.
4. Implement Average Precision (AP) using 11-point interpolation and All-points AUC.
5. Compute Mean Average Precision across multi-class aerial drone objects (mAP@50 and mAP@50:95).
"""

import numpy as np
from typing import List, Dict, Tuple

def compute_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def calculate_ap_for_class(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.50
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Average Precision (AP) for a single class at a specified IoU threshold.
    
    Args:
        pred_boxes: Array of shape (N, 4) in [x1, y1, x2, y2]
        pred_scores: Array of shape (N,) confidence scores
        gt_boxes: Array of shape (M, 4) ground truth boxes
        iou_threshold: Minimum IoU to consider a detection a True Positive
        
    Returns:
        ap: Area under the Precision-Recall curve
        precisions: Precision values across recall thresholds
        recalls: Recall values
    """
    num_gt = len(gt_boxes)
    if num_gt == 0:
        return 0.0, np.array([]), np.array([])
    if len(pred_boxes) == 0:
        return 0.0, np.zeros(num_gt), np.zeros(num_gt)

    # Sort predictions by descending confidence score
    sort_order = np.argsort(pred_scores)[::-1]
    sorted_boxes = pred_boxes[sort_order]
    sorted_scores = pred_scores[sort_order]

    tp = np.zeros(len(sorted_boxes))
    fp = np.zeros(len(sorted_boxes))
    gt_matched = np.zeros(num_gt, dtype=bool)

    # Match each prediction to highest-IoU unmatched ground truth box
    for i, pred in enumerate(sorted_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for g, gt in enumerate(gt_boxes):
            iou = compute_box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g

        if best_iou >= iou_threshold:
            if not gt_matched[best_gt_idx]:
                tp[i] = 1.0  # True Positive (first detection to match GT)
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1.0  # False Positive (duplicate detection)
        else:
            fp[i] = 1.0      # False Positive (poor localization or hallucination)

    # Compute cumulative metrics
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / num_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolated Average Precision (PASCAL VOC standard)
    recall_levels = np.linspace(0.0, 1.0, 11)
    interpolated_precisions = []
    for r in recall_levels:
        prec_at_r = precisions[recalls >= r]
        max_p = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
        interpolated_precisions.append(max_p)
    ap_11 = float(np.mean(interpolated_precisions))

    # All-points continuous AUC AP (COCO standard)
    # Append boundary points
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    m_prec = np.concatenate(([1.0], precisions, [0.0]))
    
    # Compute maximum precision envelope backwards
    for k in range(len(m_prec) - 2, -1, -1):
        m_prec[k] = max(m_prec[k], m_prec[k + 1])
        
    # Find points where recall changes
    recall_change_idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    ap_coco = float(np.sum((m_rec[recall_change_idx + 1] - m_rec[recall_change_idx]) * m_prec[recall_change_idx + 1]))

    return ap_coco, precisions, recalls

def demonstrate_map_evaluation():
    print("=" * 70)
    print("STEP 1: PRECISION-RECALL & mAP CALCULATION FROM SCRATCH")
    print("=" * 70)

    # Ground truth drone vehicles (5 cars on a roadway)
    gt_cars = np.array([
        [100.0, 100.0, 200.0, 200.0],  # Car 1
        [250.0, 150.0, 350.0, 250.0],  # Car 2
        [400.0, 200.0, 500.0, 300.0],  # Car 3
        [600.0, 100.0, 700.0, 200.0],  # Car 4
        [750.0, 250.0, 850.0, 350.0],  # Car 5
    ], dtype=np.float32)

    # Detector predictions (6 candidates with various confidences and IoUs)
    pred_cars = np.array([
        [102.0,  98.0, 198.0, 201.0],  # High score, excellent overlap with Car 1 (TP)
        [248.0, 152.0, 352.0, 248.0],  # High score, excellent overlap with Car 2 (TP)
        [405.0, 205.0, 495.0, 295.0],  # Medium score, good overlap with Car 3 (TP)
        [105.0, 102.0, 195.0, 198.0],  # Duplicate detection of Car 1 (FP)
        [610.0, 110.0, 690.0, 190.0],  # Low score, moderate overlap with Car 4 (TP)
        [900.0, 500.0, 950.0, 550.0],  # False alarm / background hallucination (FP)
        # Note: Car 5 is missed by the detector! (FN)
    ], dtype=np.float32)

    pred_scores = np.array([0.95, 0.91, 0.82, 0.76, 0.58, 0.35], dtype=np.float32)

    # 1. Compute AP@50
    ap_50, precs, recs = calculate_ap_for_class(pred_cars, pred_scores, gt_cars, iou_threshold=0.50)

    print(f"Ground Truth Objects  : {len(gt_cars)}")
    print(f"Detector Predictions  : {len(pred_cars)}")
    print(f"Computed AP@50 (Cars) : {ap_50 * 100.0:.2f}%\n")

    print(f"{'Rank':<5} | {'Score':<6} | {'Outcome':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 50)
    for i in range(len(precs)):
        print(f"{i+1:<5} | {pred_scores[i]:<6.2f} | {'TP/FP':<8} | {precs[i]:<10.3f} | {recs[i]:<8.3f}")

    # 2. Compute mAP@[50:95] across IoU thresholds (0.50, 0.55, ..., 0.95)
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap_values = []
    print(f"\nAP across IoU Thresholds (50% to 95%):")
    for iou_th in iou_thresholds:
        ap_val, _, _ = calculate_ap_for_class(pred_cars, pred_scores, gt_cars, iou_threshold=iou_th)
        ap_values.append(ap_val)
        print(f"  IoU @ {iou_th:.2f}: AP = {ap_val * 100.0:.1f}%")

    map_50_95 = float(np.mean(ap_values))
    print(f"\nmAP@[50:95] (Strict Localization Metric): {map_50_95 * 100.0:.2f}%")
    print("\nTakeaway:")
    print("  mAP@50 measures general object recognition ability.")
    print("  mAP@[50:95] penalizes sloppy bounding box borders, essential for high-precision drone surveying.")
    print()

def main():
    demonstrate_map_evaluation()
    print("=" * 70)
    print("PHASE 5 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
