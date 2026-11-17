"""
=============================================================================
AeroVision AI - Object Detection Foundations: Module 01
=============================================================================
Bounding Box Geometry, Coordinate Formats, and Vectorized IoU

Learning Objectives:
1. Master bounding box coordinate formats: XYXY, XYWH, CXCYWH, and Normalized.
2. Vectorize conversions between coordinate representations with zero copying.
3. Mathematically derive and implement Intersection over Union (IoU) from scratch.
4. Implement vectorized N x M pairwise IoU matrix computation using NumPy broadcasting.
5. Understand modern bounding box loss metrics (GIoU, DIoU, CIoU) used in YOLO11.
"""

import numpy as np
from typing import Tuple

def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Converts [x1, y1, x2, y2] -> [cx, cy, width, height]."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.column_stack([cx, cy, w, h])

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts [cx, cy, width, height] -> [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.column_stack([x1, y1, x2, y2])

def compute_single_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two boxes in [x1, y1, x2, y2] format.
    
    Formula:
      Inter_x1 = max(x1_a, x1_b), Inter_y1 = max(y1_a, y1_b)
      Inter_x2 = min(x2_a, x2_b), Inter_y2 = min(y2_a, y2_b)
      Intersection_Area = max(0, Inter_x2 - Inter_x1) * max(0, Inter_y2 - Inter_y1)
      Union_Area = Area_a + Area_b - Intersection_Area
      IoU = Intersection_Area / Union_Area
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)

def compute_pairwise_iou_vectorized(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU matrix of shape (N, M) between N boxes in boxes_a
    and M boxes in boxes_b using NumPy broadcasting.
    
    Time Complexity: O(N * M) highly vectorized (SIMD CPU execution).
    """
    # boxes_a: (N, 4) -> (N, 1, 4)
    # boxes_b: (M, 4) -> (1, M, 4)
    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]

    # Intersection coordinates: (N, M)
    inter_x1 = np.maximum(a[:, :, 0], b[:, :, 0])
    inter_y1 = np.maximum(a[:, :, 1], b[:, :, 1])
    inter_x2 = np.minimum(a[:, :, 2], b[:, :, 2])
    inter_y2 = np.minimum(a[:, :, 3], b[:, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Individual box areas
    area_a = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
    area_b = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])

    union_area = area_a + area_b - inter_area
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0.0)

    return iou_matrix

def demonstrate_iou_and_geometry():
    print("=" * 70)
    print("STEP 1: BOUNDING BOX COORDINATE CONVERSIONS")
    print("=" * 70)

    # 3 sample drone vehicle detections in [x1, y1, x2, y2]
    original_xyxy = np.array([
        [100.0, 150.0, 200.0, 250.0],  # Box 1: Car 1 (100x100)
        [140.0, 180.0, 240.0, 280.0],  # Box 2: Overlapping Car (100x100)
        [500.0, 600.0, 650.0, 800.0]   # Box 3: Distant Truck (150x200)
    ], dtype=np.float32)

    cxcywh = xyxy_to_cxcywh(original_xyxy)
    restored_xyxy = cxcywh_to_xyxy(cxcywh)

    diff = np.max(np.abs(original_xyxy - restored_xyxy))
    print("Original XYXY Coordinates:\n", original_xyxy)
    print("\nConverted to CXCYWH (Center X, Center Y, Width, Height):\n", np.round(cxcywh, 2))
    print(f"\nRoundtrip Conversion Max Error: {diff:.8f}")
    assert diff < 1e-6, "Coordinate conversion failed!"

    print("\n" + "=" * 70)
    print("STEP 2: VECTORIZED N x M PAIRWISE IoU CALCULATION")
    print("=" * 70)

    # Compute pairwise IoU matrix
    iou_matrix = compute_pairwise_iou_vectorized(original_xyxy, original_xyxy)
    print(f"Computed Pairwise IoU Matrix (Shape {iou_matrix.shape}):\n")
    for i in range(len(original_xyxy)):
        row_str = "  ".join([f"{iou_matrix[i, j]:.4f}" for j in range(len(original_xyxy))])
        print(f"  Box {i+1} IoU with [Box 1, Box 2, Box 3]: [{row_str}]")

    # Verify self-IoU is strictly 1.0
    for i in range(len(original_xyxy)):
        assert np.isclose(iou_matrix[i, i], 1.0), f"Self-IoU must be 1.0, got {iou_matrix[i, i]}"

    # Verify IoU(Box 1, Box 2) matches scalar function
    scalar_iou = compute_single_iou(original_xyxy[0], original_xyxy[1])
    vector_iou = iou_matrix[0, 1]
    print(f"\nIoU(Box 1, Box 2) Scalar Function  : {scalar_iou:.6f}")
    print(f"IoU(Box 1, Box 2) Vectorized Matrix: {vector_iou:.6f}")
    assert np.isclose(scalar_iou, vector_iou, atol=1e-5)
    print(f"Disjoint Box 1 & Box 3 IoU         : {iou_matrix[0, 2]:.4f} (Zero overlap)")

    print("\n" + "=" * 70)
    print("STEP 3: ADVANCED BOUNDING BOX LOSS FORMULATIONS (GIoU vs DIoU vs CIoU)")
    print("=" * 70)
    print("1. Standard IoU Loss: L_IoU = 1 - IoU")
    print("   Problem: If boxes do not overlap (IoU = 0), gradient is zero! (No learning signal)")
    print("2. Generalized IoU (GIoU):")
    print("   L_GIoU = 1 - IoU + (Area_Convex - Area_Union) / Area_Convex")
    print("   Penalizes the empty space inside the smallest enclosing convex hull.")
    print("3. Complete IoU (CIoU) - Used in YOLO11:")
    print("   L_CIoU = 1 - IoU + (rho^2(b, b_gt) / c^2) + alpha * v")
    print("   Where rho is Euclidean center distance, c is enclosing diagonal, and v measures aspect ratio consistency.")
    print("\n[SUCCESS] Bounding box geometry and IoU derivations verified.")

def main():
    demonstrate_iou_and_geometry()
    print("=" * 70)
    print("PHASE 5 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
