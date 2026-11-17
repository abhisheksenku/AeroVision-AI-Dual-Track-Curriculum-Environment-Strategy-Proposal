"""
=============================================================================
AeroVision AI - Computer Vision Foundations: Module 04
=============================================================================
Aspect Ratio Preservation & Letterbox Preprocessing for YOLO

Learning Objectives:
1. Understand why naive image resizing degrades object detection accuracy.
2. Implement the Letterbox algorithm (rescaling + symmetric padding).
3. Derive the mathematical inverse coordinate mapping (Un-letterboxing).
4. Prove zero-distortion coordinate fidelity from detector space back to native frame space.
"""

import cv2
import numpy as np
from typing import Tuple

def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes image with aspect ratio preservation and symmetric padding.
    
    Returns:
        padded_img: Array of shape (target_size[1], target_size[0], 3)
        scale_ratio: Rescaling factor applied (min(w_ratio, h_ratio))
        (dw, dh): Symmetric padding offsets (horizontal, vertical)
    """
    h_orig, w_orig = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale ratio
    r = min(target_w / w_orig, target_h / h_orig)

    # Resized dimensions before padding
    new_w = int(round(w_orig * r))
    new_h = int(round(h_orig * r))

    # Padding required to reach target dimensions
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0

    # Resize using INTER_AREA if downscaling, INTER_LINEAR if upscaling
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Integer padding values
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Apply constant border
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=fill_color
    )

    return padded, r, (dw, dh)

def unletterbox_coordinates(
    boxes_letterboxed: np.ndarray,
    scale_ratio: float,
    padding: Tuple[float, float],
    original_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Mathematically projects bounding boxes from letterbox (640x640) space
    back to original video frame coordinates (e.g. 1920x1080).
    
    Formula:
        x_orig = (x_letterbox - dw) / r
        y_orig = (y_letterbox - dh) / r
    
    Args:
        boxes_letterboxed: Array of shape (N, 4) in [x1, y1, x2, y2]
        scale_ratio: Scale factor r returned by letterbox
        padding: (dw, dh) padding offsets returned by letterbox
        original_shape: (height, width) of original frame
        
    Returns:
        boxes_orig: Array of shape (N, 4) clipped to [0, width] and [0, height]
    """
    dw, dh = padding
    orig_h, orig_w = original_shape
    
    boxes_orig = boxes_letterboxed.copy().astype(np.float32)
    
    # Invert horizontal offset and scale
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - dw) / scale_ratio
    
    # Invert vertical offset and scale
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - dh) / scale_ratio
    
    # Clip coordinates to valid image bounds
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)
    
    return boxes_orig

def demonstrate_letterbox_roundtrip():
    """
    Demonstrates and mathematically proves zero round-trip projection error.
    """
    print("=" * 70)
    print("STEP 1: LETTERBOX PREPROCESSING & INVERSE COORDINATE MAPPING")
    print("=" * 70)

    # Simulate a Full HD 1080p drone frame: 1920x1080 (16:9 widescreen)
    orig_w, orig_h = 1920, 1080
    synthetic_frame = np.full((orig_h, orig_w, 3), 100, dtype=np.uint8)

    # Suppose a ground vehicle is located at (x1=800, y1=500, x2=950, y2=620) in 1080p
    gt_box = np.array([[800.0, 500.0, 950.0, 620.0]], dtype=np.float32)

    # 1. Apply Letterboxing to standard 640x640 YOLO target
    target_size = (640, 640)
    padded_img, r, (dw, dh) = letterbox_image(synthetic_frame, target_size=target_size)

    print(f"Original Video Frame : {orig_w}x{orig_h} (Aspect Ratio = {orig_w/orig_h:.3f})")
    print(f"Target YOLO Input    : {target_size[0]}x{target_size[1]} (Aspect Ratio = 1.000)")
    print(f"Applied Scale Ratio r: {r:.4f}")
    print(f"Computed Padding     : Horizontal dw = {dw:.1f} px, Vertical dh = {dh:.1f} px")
    print(f"Padded Output Shape  : {padded_img.shape} (Matches 640x640)")

    # 2. Forward projection: map ground truth box into letterbox space
    letterboxed_box = gt_box.copy()
    letterboxed_box[:, [0, 2]] = (letterboxed_box[:, [0, 2]] * r) + dw
    letterboxed_box[:, [1, 3]] = (letterboxed_box[:, [1, 3]] * r) + dh

    print(f"\nGround Truth Box (1080p space): [x1={gt_box[0,0]}, y1={gt_box[0,1]}, x2={gt_box[0,2]}, y2={gt_box[0,3]}]")
    print(f"Letterbox Model Space (640x640) : [x1={letterboxed_box[0,0]:.1f}, y1={letterboxed_box[0,1]:.1f}, x2={letterboxed_box[0,2]:.1f}, y2={letterboxed_box[0,3]:.1f}]")

    # 3. Inverse projection: un-letterbox back to native frame space
    restored_box = unletterbox_coordinates(
        letterboxed_box,
        scale_ratio=r,
        padding=(dw, dh),
        original_shape=(orig_h, orig_w)
    )

    print(f"Restored Native Box           : [x1={restored_box[0,0]:.1f}, y1={restored_box[0,1]:.1f}, x2={restored_box[0,2]:.1f}, y2={restored_box[0,3]:.1f}]")

    # Measure coordinate discrepancy
    max_error = np.max(np.abs(gt_box - restored_box))
    print(f"Roundtrip Numerical Error     : {max_error:.8f} pixels")
    assert max_error < 1e-4, f"Roundtrip coordinate error exceeds tolerance: {max_error}"

    print("\n[SUCCESS] Letterbox forward and inverse projections are mathematically exact.")
    print()

def main():
    demonstrate_letterbox_roundtrip()
    print("=" * 70)
    print("PHASE 4 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
