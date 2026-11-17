"""
Reference Verification & Solution Script for Module 1 Exercises
"""
import numpy as np

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) of two bounding boxes in xyxy format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    # Compute union area
    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def test_iou_calculations():
    # Test identical boxes
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    iou_identical = compute_iou(box1, box2)
    assert np.isclose(iou_identical, 1.0), f"Expected 1.0, got {iou_identical}"

    # Test non-overlapping boxes
    box3 = [100, 100, 150, 150]
    iou_none = compute_iou(box1, box3)
    assert np.isclose(iou_none, 0.0), f"Expected 0.0, got {iou_none}"

    # Test 50% overlap box
    box4 = [30, 10, 70, 50]
    iou_partial = compute_iou(box1, box4)
    print(f"Partial IoU between {box1} and {box4}: {iou_partial:.4f}")
    assert 0.0 < iou_partial < 1.0

    print("All IoU exercise reference tests passed successfully!")

if __name__ == "__main__":
    test_iou_calculations()
