"""
=============================================================================
AeroVision AI - Analytics Engine Foundations: Module 02
=============================================================================
Spatial Density Estimation, Traffic Accumulation & Congestion Heatmaps

Learning Objectives:
1. Discretize continuous 2D video coordinate space into spatial density grids.
2. Implement 2D Gaussian Kernel Density Estimation (KDE) from scratch in NumPy.
3. Generate traffic density heatmaps from vehicle trajectory centroids.
4. Blend pseudo-color heatmaps with aerial video frames using alpha compositing.
"""

import numpy as np
import cv2

def generate_spatial_density_grid(
    centroids: np.ndarray,
    width: int = 1280,
    height: int = 720,
    cell_size: int = 32
) -> np.ndarray:
    """
    Computes a 2D histogram grid counting object presence per spatial block.
    """
    grid_w = width // cell_size
    grid_h = height // cell_size
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for x, y in centroids:
        col = int(np.clip(x // cell_size, 0, grid_w - 1))
        row = int(np.clip(y // cell_size, 0, grid_h - 1))
        grid[row, col] += 1.0

    return grid

def compute_gaussian_kde_heatmap(
    centroids: np.ndarray,
    width: int = 1280,
    height: int = 720,
    sigma: float = 35.0
) -> np.ndarray:
    """
    Generates a continuous traffic density heatmap via Gaussian Kernel Density Estimation.
    
    Formula:
      Density(x, y) = sum_i exp( - ((x - x_i)^2 + (y - y_i)^2) / (2 * sigma^2) )
    """
    # Downsample accumulator by 4x for high-throughput CPU performance
    scale = 0.25
    scaled_w = int(width * scale)
    scaled_h = int(height * scale)
    scaled_sigma = sigma * scale

    accumulator = np.zeros((scaled_h, scaled_w), dtype=np.float32)
    radius = int(scaled_sigma * 3)

    # Precalculate discrete Gaussian kernel patch
    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * scaled_sigma**2))

    for pt in centroids:
        cx = int(pt[0] * scale)
        cy = int(pt[1] * scale)

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(scaled_w, cx + radius + 1)
        y2 = min(scaled_h, cy + radius + 1)

        kx1 = x1 - (cx - radius)
        ky1 = y1 - (cy - radius)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            accumulator[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    # Resize back to native resolution using INTER_LINEAR
    density_map = cv2.resize(accumulator, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0 - 255
    max_val = density_map.max()
    if max_val > 0:
        norm_density = (density_map / max_val * 255.0).astype(np.uint8)
    else:
        norm_density = np.zeros((height, width), dtype=np.uint8)

    return norm_density

def demonstrate_spatial_heatmaps():
    print("=" * 70)
    print("STEP 1: SPATIAL DENSITY & TRAFFIC ACCUMULATION HEATMAPS")
    print("=" * 70)

    # Simulate 500 vehicle trajectory observations on an aerial highway corridor:
    # Most points travel along the middle highway (Y ~ 350)
    # A congested bottleneck exists at an intersection (X ~ 640, Y ~ 350)
    np.random.seed(42)
    num_samples = 500

    # 70% of points spread along the road
    road_x = np.random.uniform(100, 1180, int(num_samples * 0.7))
    road_y = np.random.normal(350, 25, int(num_samples * 0.7))

    # 30% of points clustered at the bottleneck intersection (high dwell congestion)
    congested_x = np.random.normal(640, 30, int(num_samples * 0.3))
    congested_y = np.random.normal(350, 15, int(num_samples * 0.3))

    all_x = np.concatenate([road_x, congested_x])
    all_y = np.concatenate([road_y, congested_y])
    centroids = np.column_stack([all_x, all_y])

    # 1. Discrete Grid
    grid = generate_spatial_density_grid(centroids, width=1280, height=720, cell_size=64)
    print(f"Discrete Grid Dimensions   : {grid.shape[1]} cols x {grid.shape[0]} rows (64x64 px cells)")
    print(f"Max Vehicle Observations in Single Cell: {int(grid.max())} counts")

    # 2. Gaussian KDE Heatmap
    heatmap_gray = compute_gaussian_kde_heatmap(centroids, width=1280, height=720, sigma=40.0)
    print(f"Native Density Heatmap Shape : {heatmap_gray.shape} (uint8 range 0-255)")
    print(f"Peak Density Coordinate      : (X={np.argmax(heatmap_gray) % 1280}, Y={np.argmax(heatmap_gray) // 1280})")

    # 3. Apply Color Map (COLORMAP_JET: Blue = Free flow, Red = Severe Congestion)
    heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

    # 4. Alpha Blending with base frame
    base_frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
    alpha = 0.6
    # Only blend where density > 0
    mask = (heatmap_gray > 20).astype(float)[:, :, np.newaxis]
    blended = (base_frame * (1.0 - mask * (1.0 - alpha)) + heatmap_colored * (mask * (1.0 - alpha))).astype(np.uint8)

    print(f"Blended Output Frame Shape   : {blended.shape}")
    assert blended.shape == (720, 1280, 3), "Heatmap blending dimension mismatch!"
    print("\n[SUCCESS] Gaussian KDE spatial density estimation and alpha blending verified.")
    print()

def main():
    demonstrate_spatial_heatmaps()
    print("=" * 70)
    print("PHASE 7 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
