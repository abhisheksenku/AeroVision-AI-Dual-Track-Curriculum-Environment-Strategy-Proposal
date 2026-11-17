"""
=============================================================================
AeroVision AI - Frontend Engineering Foundations: Module 03
=============================================================================
Responsive HTML5 Canvas Coordinate Projections & Aspect-Ratio Fitting

Learning Objectives:
1. Understand the coordinate mismatch between native video frames and client DOM display sizes.
2. Implement CSS object-fit: contain mathematical projection equations.
3. Map bounding box pixel coordinates from 1080p/720p native space to dynamic client canvas viewports.
"""

from typing import Tuple, List
import numpy as np

def project_native_box_to_canvas(
    native_box: Tuple[float, float, float, float],
    native_dim: Tuple[int, int],
    container_dim: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Projects bounding box [x1, y1, x2, y2] from native video resolution
    to responsive HTML5 canvas container using CSS 'object-fit: contain' logic.
    
    Args:
        native_box: (x1, y1, x2, y2) in native video frame coordinates
        native_dim: (width, height) of native video (e.g. 1280, 720)
        container_dim: (width, height) of client browser container element
        
    Returns:
        canvas_box: (x1, y1, x2, y2) in client canvas display coordinates
    """
    x1, y1, x2, y2 = native_box
    w_nat, h_nat = native_dim
    w_con, h_con = container_dim

    # Scale factor
    scale = min(w_con / w_nat, h_con / h_nat)

    # Offset for letterbox bars inside container
    rendered_w = w_nat * scale
    rendered_h = h_nat * scale
    offset_x = (w_con - rendered_w) / 2.0
    offset_y = (h_con - rendered_h) / 2.0

    # Project coordinates
    c_x1 = x1 * scale + offset_x
    c_y1 = y1 * scale + offset_y
    c_x2 = x2 * scale + offset_x
    c_y2 = y2 * scale + offset_y

    return (round(c_x1, 2), round(c_y1, 2), round(c_x2, 2), round(c_y2, 2))

def demonstrate_canvas_projection():
    print("=" * 70)
    print("STEP 1: RESPONSIVE CANVAS COORDINATE PROJECTION (OBJECT-FIT: CONTAIN)")
    print("=" * 70)

    # Native Drone Video: 1280x720 (16:9)
    native_dim = (1280, 720)
    # Target vehicle box on highway: x1=400, y1=200, x2=550, y2=320
    native_box = (400.0, 200.0, 550.0, 320.0)

    # Scenario A: Browser resized to laptop window (960x540) - Exact 16:9 aspect ratio
    client_dim_a = (960, 540)
    proj_a = project_native_box_to_canvas(native_box, native_dim, client_dim_a)
    print(f"Scenario A (960x540 exact 16:9):")
    print(f"  Native Box : {native_box}")
    print(f"  Canvas Box : {proj_a} (Scale: {960/1280:.2f}x, Zero Offset)")
    assert proj_a == (300.0, 150.0, 412.5, 240.0)

    # Scenario B: Ultrawide display or tall column container (800x600) - Pillarboxed!
    # Height is the bottleneck: 600/720 = 0.833, Width rendered = 1280 * 0.833 = 1066.6 > 800
    # So Width is the bottleneck: 800/1280 = 0.625, Height rendered = 720 * 0.625 = 450 (150px vertical black bars)
    client_dim_b = (800, 600)
    proj_b = project_native_box_to_canvas(native_box, native_dim, client_dim_b)
    print(f"\nScenario B (800x600 with Vertical Letterbox Offset):")
    print(f"  Native Box : {native_box}")
    print(f"  Canvas Box : {proj_b} (Scale: 0.625x, Vertical Offset: 75px)")

    # Assert vertical offset is applied: offset_y = (600 - 450) / 2 = 75px
    # c_y1 = 200 * 0.625 + 75 = 125 + 75 = 200.0
    assert proj_b[1] == 200.0
    assert proj_b[0] == 400.0 * 0.625

    print("\n[SUCCESS] Responsive canvas coordinate mapping mathematically verified.")
    print()

def main():
    demonstrate_canvas_projection()
    print("=" * 70)
    print("PHASE 9 - MODULE 03: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
