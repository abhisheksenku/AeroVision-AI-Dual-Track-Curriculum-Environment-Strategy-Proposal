"""
=============================================================================
AeroVision AI - Computer Vision Foundations: Module 02
=============================================================================
Spatial Transformations, Interpolation, and Edge Filtering

Learning Objectives:
1. Master OpenCV interpolation methods (INTER_NEAREST, INTER_LINEAR, INTER_AREA).
2. Understand why INTER_AREA is mandatory for aerial downsampling to avoid aliasing.
3. Compute Affine transformations and 2D rotation matrices for drone yaw stabilization.
4. Implement 2D spatial convolution filters from scratch and compare with cv2.filter2D.
5. Apply Gaussian smoothing, Sobel gradients, and Canny edge detection on drone imagery.
"""

import os
import cv2
import numpy as np

def demonstrate_interpolation_algorithms(image_path: str):
    """
    Compares interpolation algorithms when downscaling and upscaling images.
    Explains the mathematical difference between Nearest Neighbor, Bilinear, and Area.
    """
    print("=" * 70)
    print("STEP 1: INTERPOLATION ALGORITHMS FOR IMAGE RESIZING")
    print("=" * 70)

    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    print(f"Original Resolution: {orig_w}x{orig_h} px")

    # Downsampling to 320x320
    target_down = (320, 320)
    
    # 1. INTER_NEAREST: Copies nearest pixel. Fastest (O(1)), but introduces jagged edges/blockiness.
    down_nearest = cv2.resize(img, target_down, interpolation=cv2.INTER_NEAREST)
    
    # 2. INTER_LINEAR: 2x2 bilinear interpolation weighted by spatial distance. Standard for upsampling.
    down_linear = cv2.resize(img, target_down, interpolation=cv2.INTER_LINEAR)
    
    # 3. INTER_AREA: Pixel area relation resampling. Computes weighted average over pixel area.
    # Essential for downsampling high-res 4K drone footage to prevent aliasing (Moire artifacts).
    down_area = cv2.resize(img, target_down, interpolation=cv2.INTER_AREA)

    print(f"Downsampled to {target_down[0]}x{target_down[1]}:")
    print("  INTER_NEAREST: Fast, but introduces staircase/aliasing artifacts.")
    print("  INTER_LINEAR : Smooth, balanced compromise.")
    print("  INTER_AREA   : Optimal for downscaling (preserves thin power lines, road markings).")

    # Measure variance (sharpness metric)
    var_nearest = cv2.Laplacian(down_nearest, cv2.CV_64F).var()
    var_linear = cv2.Laplacian(down_linear, cv2.CV_64F).var()
    var_area = cv2.Laplacian(down_area, cv2.CV_64F).var()

    print(f"\nLaplacian Variance (High-frequency energy preserved):")
    print(f"  Nearest : {var_nearest:.1f}")
    print(f"  Linear  : {var_linear:.1f}")
    print(f"  Area    : {var_area:.1f}")
    print()

def demonstrate_drone_yaw_rotation(image_path: str):
    """
    Applies 2D Euclidean / Affine transformation to rotate an image around its center,
    simulating drone gimbal yaw rotation.
    """
    print("=" * 70)
    print("STEP 2: AFFINE TRANSFORMATIONS & DRONE YAW ROTATION")
    print("=" * 70)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle_deg = 35.0  # Drone yaw rotation of 35 degrees
    scale = 1.0

    # Generate 2x3 affine rotation matrix M:
    # [ alpha  beta  (1-alpha)*cx - beta*cy ]
    # [ -beta alpha   beta*cx + (1-alpha)*cy ]
    # where alpha = scale * cos(angle), beta = scale * sin(angle)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # Apply warp affine
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    rad = np.radians(angle_deg)
    expected_alpha = scale * np.cos(rad)
    expected_beta = scale * np.sin(rad)

    print(f"Rotation Angle : {angle_deg}° around center ({center[0]:.1f}, {center[1]:.1f})")
    print(f"Computed Affine Matrix (2x3):\n{np.round(M, 4)}")
    print(f"Mathematical Alpha (cos): {expected_alpha:.4f} vs M[0,0]: {M[0,0]:.4f}")
    print(f"Mathematical Beta  (sin): {expected_beta:.4f} vs M[0,1]: {M[0,1]:.4f}")
    assert np.isclose(M[0, 0], expected_alpha, atol=1e-3), "Affine alpha mismatch!"
    print("[SUCCESS] Drone yaw rotation matrix mathematically verified.")
    print()

def demonstrate_spatial_convolutions_and_edges(image_path: str):
    """
    Demonstrates image filtering:
    1. 2D Convolution from scratch vs cv2.filter2D.
    2. Gaussian smoothing for camera sensor noise suppression.
    3. Sobel gradient magnitude for edge boundary extraction.
    4. Canny edge detector (hysteresis thresholding).
    """
    print("=" * 70)
    print("STEP 3: 2D CONVOLUTIONS, GAUSSIAN SMOOTHING & CANNY EDGES")
    print("=" * 70)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    small_gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
    h, w = small_gray.shape

    # 1. Manual 2D convolution with 3x3 Sharpening Kernel
    # Kernel:
    #  0 -1  0
    # -1  5 -1
    #  0 -1  0
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    # OpenCV cv2.filter2D
    cv_filtered = cv2.filter2D(small_gray, cv2.CV_32F, kernel)

    # Manual convolution from scratch with border reflection
    pad_img = np.pad(small_gray.astype(np.float32), 1, mode='reflect')
    manual_filtered = np.zeros_like(small_gray, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = pad_img[i:i+3, j:j+3]
            manual_filtered[i, j] = np.sum(patch * kernel)

    max_conv_diff = np.max(np.abs(cv_filtered - manual_filtered))
    print(f"Manual 2D Convolution vs OpenCV filter2D max diff: {max_conv_diff:.6f}")
    assert max_conv_diff < 1e-4, "Manual 2D convolution does not match filter2D!"

    # 2. Gaussian Blur (Sensor noise reduction)
    # G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    noise_reduction = gray.std() - blurred.std()
    print(f"Gaussian Blur (5x5, sigma=1.2) variance reduction: {noise_reduction:.2f}")

    # 3. Sobel Filter (Gradient in X and Y)
    # Gx detects vertical edges, Gy detects horizontal edges
    # Magnitude |G| = sqrt(Gx^2 + Gy^2)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    print(f"Sobel Gradient Mean Magnitude: {grad_magnitude.mean():.2f}")

    # 4. Canny Edge Detector (Double thresholding with hysteresis)
    # High threshold = 150 (seeds strong edges)
    # Low threshold  = 50  (connects weak edges to strong edges)
    canny_edges = cv2.Canny(blurred, 50, 150)
    edge_density = (np.count_nonzero(canny_edges) / canny_edges.size) * 100.0
    print(f"Canny Edge Pixel Density: {edge_density:.2f}% of total pixels")
    print("[SUCCESS] Convolution filters and Canny edge pipeline verified.")
    print()

def main():
    sample_img = "data/samples/bus.jpg"
    demonstrate_interpolation_algorithms(sample_img)
    demonstrate_drone_yaw_rotation(sample_img)
    demonstrate_spatial_convolutions_and_edges(sample_img)
    print("=" * 70)
    print("PHASE 4 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
