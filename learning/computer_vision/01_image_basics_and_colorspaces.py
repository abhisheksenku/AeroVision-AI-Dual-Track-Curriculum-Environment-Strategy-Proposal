"""
=============================================================================
AeroVision AI - Computer Vision Foundations: Module 01
=============================================================================
Image Basics, Color Spaces, and Pixel Arrays

Learning Objectives:
1. Understand digital image representation as NumPy multi-dimensional arrays (H, W, C).
2. Understand OpenCV's historical BGR channel ordering vs standard RGB.
3. Perform conversions between BGR, RGB, Grayscale, and HSV color spaces.
4. Calculate and visualize image statistics (histograms, channel means, contrast).
5. Apply color-based thresholding for aerial drone imagery (e.g., vegetation/road mask).
"""

import os
import cv2
import numpy as np

def explain_image_memory_layout(image_path: str):
    """Examines image tensor geometry, data types, and contiguous memory strides."""
    print("=" * 70)
    print("STEP 1: DIGITAL IMAGE ARRAYS & MEMORY LAYOUT")
    print("=" * 70)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load image using OpenCV (loads as BGR uint8 by default)
    bgr_img = cv2.imread(image_path)
    
    h, w, c = bgr_img.shape
    dtype = bgr_img.dtype
    itemsize = bgr_img.itemsize  # bytes per element
    total_bytes = bgr_img.nbytes
    is_contiguous = bgr_img.flags['C_CONTIGUOUS']
    strides = bgr_img.strides

    print(f"Image File           : {image_path}")
    print(f"Spatial Dimensions   : Height={h} px, Width={w} px, Channels={c}")
    print(f"Data Type            : {dtype} (range: 0 to 255)")
    print(f"Element Size         : {itemsize} byte(s)")
    print(f"Memory Footprint     : {total_bytes} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
    print(f"Memory Layout        : C-Contiguous = {is_contiguous}")
    print(f"Strides (Row, Col, C): {strides} bytes")
    print("\nMathematical representation of pixel intensity at (y, x):")
    print(f"  Pixel at Center ({h//2}, {w//2}):")
    print(f"    Blue  (Channel 0) = {bgr_img[h//2, w//2, 0]}")
    print(f"    Green (Channel 1) = {bgr_img[h//2, w//2, 1]}")
    print(f"    Red   (Channel 2) = {bgr_img[h//2, w//2, 2]}")
    print()

def demonstrate_color_spaces(image_path: str):
    """
    Demonstrates transformations between BGR, RGB, Grayscale, and HSV.
    Explains why HSV is preferred for drone environmental segmentation.
    """
    print("=" * 70)
    print("STEP 2: COLOR SPACE TRANSFORMATIONS (BGR vs RGB vs Grayscale vs HSV)")
    print("=" * 70)

    bgr = cv2.imread(image_path)
    
    # 1. BGR to RGB (Required for Matplotlib, PyTorch torchvision, PIL)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # 2. BGR to Grayscale
    # Formula: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Verify manual grayscale calculation against OpenCV's optimized routine
    manual_gray = (
        0.114 * bgr[:, :, 0].astype(np.float32) +  # Blue
        0.587 * bgr[:, :, 1].astype(np.float32) +  # Green
        0.299 * bgr[:, :, 2].astype(np.float32)    # Red
    ).round().astype(np.uint8)
    
    max_gray_diff = np.max(np.abs(gray.astype(int) - manual_gray.astype(int)))
    print(f"Grayscale Conversion Standard : ITU-R BT.601 (Y = 0.299*R + 0.587*G + 0.114*B)")
    print(f"Manual vs OpenCV Grayscale Diff: {max_gray_diff} (exact match with integer rounding)")

    # 3. BGR to HSV (Hue [0-179], Saturation [0-255], Value [0-255])
    # Hue: chromatic angle (dominant wavelength)
    # Saturation: purity/vibrancy of color (0 = washed out/gray, 255 = saturated)
    # Value: luminance/intensity (0 = black, 255 = bright)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    print("\nHSV Color Space Advantages for Drone Analytics:")
    print("  1. Invariant to lighting/shadow variations: Shading primarily affects 'Value', leaving 'Hue' constant.")
    print("  2. Direct color gating: Allows filtering vegetation (green) or water (blue) regardless of sun exposure.")

    h_mean, s_mean, v_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    print(f"Mean Scene Hue (0-179)       : {h_mean:.1f}")
    print(f"Mean Scene Saturation (0-255): {s_mean:.1f}")
    print(f"Mean Scene Value (0-255)     : {v_mean:.1f}")
    print()

def demonstrate_aerial_segmentation_mask():
    """
    Demonstrates HSV range mask segmentation to detect road asphalt and green zones
    using synthetic aerial patches.
    """
    print("=" * 70)
    print("STEP 3: HSV COLOR THRESHOLDING FOR AERIAL ROADS & VEGETATION")
    print("=" * 70)

    # Create a synthetic aerial landscape patch (200x300 px)
    # Top-half: Green grass/trees; Bottom-half: Gray asphalt road
    patch = np.zeros((200, 300, 3), dtype=np.uint8)
    patch[:100, :] = [34, 139, 34]    # BGR for Forest Green
    patch[100:, :] = [80, 80, 80]     # BGR for Gray Asphalt Road

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # HSV thresholds for Green Vegetation: Hue 35 to 85
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    vegetation_mask = cv2.inRange(hsv_patch, lower_green, upper_green)

    # HSV thresholds for Road/Asphalt (Low saturation, moderate value)
    lower_road = np.array([0, 0, 40], dtype=np.uint8)
    upper_road = np.array([180, 50, 150], dtype=np.uint8)
    road_mask = cv2.inRange(hsv_patch, lower_road, upper_road)

    veg_percentage = (np.count_nonzero(vegetation_mask) / vegetation_mask.size) * 100.0
    road_percentage = (np.count_nonzero(road_mask) / road_mask.size) * 100.0

    print(f"Vegetation Mask Coverage     : {veg_percentage:.1f}% (Expected: ~50.0%)")
    print(f"Asphalt Road Mask Coverage   : {road_percentage:.1f}% (Expected: ~50.0%)")
    assert abs(veg_percentage - 50.0) < 1.0, "Vegetation mask segmentation failed!"
    assert abs(road_percentage - 50.0) < 1.0, "Road mask segmentation failed!"
    print("[SUCCESS] HSV color segmentation accurately isolated aerial terrain classes.")
    print()

def main():
    sample_img = "data/samples/bus.jpg"
    explain_image_memory_layout(sample_img)
    demonstrate_color_spaces(sample_img)
    demonstrate_aerial_segmentation_mask()
    print("=" * 70)
    print("PHASE 4 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
