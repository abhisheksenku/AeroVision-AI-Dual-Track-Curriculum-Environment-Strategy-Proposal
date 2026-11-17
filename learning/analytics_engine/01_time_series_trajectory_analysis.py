"""
=============================================================================
AeroVision AI - Analytics Engine Foundations: Module 01
=============================================================================
Time-Series Trajectory Analysis, Moving Averages, and Dwell Time

Learning Objectives:
1. Process spatial tracking sequences (t, x, y, v) as multivariate time series.
2. Implement Exponential Moving Average (EMA) smoothing for noisy velocity vectors.
3. Compute compass heading angles (degrees and cardinal directions: N, E, S, W).
4. Detect stationary dwell time and loitering events from trajectory data.
"""

import numpy as np
from typing import List, Tuple, Dict

def smooth_trajectory_ema(points: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Applies Exponential Moving Average (EMA) to smooth noisy spatial coordinates.
    Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    smoothed = np.zeros_like(points)
    smoothed[0] = points[0]
    for t in range(1, len(points)):
        smoothed[t] = alpha * points[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed

def compute_heading_bearing(dx: float, dy: float) -> Tuple[float, str]:
    """
    Computes compass heading angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
    given pixel displacement (dx, dy). Note: Image Y points downward!
    """
    # In image coordinates, y increases downward.
    # Convert to mathematical Cartesian where positive Y is North:
    cartesian_dx = dx
    cartesian_dy = -dy

    angle_rad = np.arctan2(cartesian_dx, cartesian_dy) # 0 is North (up)
    angle_deg = np.degrees(angle_rad) % 360.0

    # Map to 8-point compass
    compass_sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    sector_idx = int(round(angle_deg / 45.0))
    cardinal = compass_sectors[sector_idx]

    return float(angle_deg), cardinal

def detect_dwell_and_stoppage(
    timestamps: np.ndarray,
    velocities: np.ndarray,
    speed_threshold: float = 2.0,     # km/h
    min_dwell_duration: float = 1.0   # seconds
) -> List[Dict]:
    """
    Identifies stationary periods (stoppages or loitering) where velocity < threshold
    for at least min_dwell_duration consecutive seconds.
    """
    stoppages = []
    is_stopped = False
    stop_start_t = 0.0

    for t, v in zip(timestamps, velocities):
        if v < speed_threshold:
            if not is_stopped:
                is_stopped = True
                stop_start_t = t
        else:
            if is_stopped:
                dwell_time = t - stop_start_t
                if dwell_time >= min_dwell_duration:
                    stoppages.append({
                        "start_time_sec": round(stop_start_t, 2),
                        "end_time_sec": round(t, 2),
                        "dwell_duration_sec": round(dwell_time, 2)
                    })
                is_stopped = False

    # Check if still stopped at the end of trajectory
    if is_stopped:
        dwell_time = timestamps[-1] - stop_start_t
        if dwell_time >= min_dwell_duration:
            stoppages.append({
                "start_time_sec": round(stop_start_t, 2),
                "end_time_sec": round(timestamps[-1], 2),
                "dwell_duration_sec": round(dwell_time, 2)
            })

    return stoppages

def demonstrate_trajectory_time_series():
    print("=" * 70)
    print("STEP 1: TIME-SERIES TRAJECTORY SMOOTHING & HEADING")
    print("=" * 70)

    # 100-step trajectory (30 FPS, ~3.33 seconds)
    # Vehicle drives East (dx > 0), stops for 1.2 seconds, then resumes East
    dt = 1.0 / 30.0
    timestamps = np.array([i * dt for i in range(100)])

    raw_x = []
    raw_y = []
    cur_x, cur_y = 100.0, 300.0
    np.random.seed(42)

    for i in range(100):
        # Frame 0-30: Driving East at 30 km/h (~166 px/s)
        # Frame 31-70: Stationary stop (traffic light or obstruction)
        # Frame 71-99: Driving East at 35 km/h
        if i < 30 or i >= 70:
            speed_px_f = 5.5 # px per frame
        else:
            speed_px_f = 0.0 # stopped

        cur_x += speed_px_f + np.random.normal(0, 0.2) # realistic sensor jitter
        cur_y += 0.0 + np.random.normal(0, 0.2)
        raw_x.append(cur_x)
        raw_y.append(cur_y)

    points = np.column_stack([raw_x, raw_y])
    smoothed = smooth_trajectory_ema(points, alpha=0.35)

    # Calculate smoothed velocities
    diffs = np.diff(smoothed, axis=0)
    speeds_px_s = np.hypot(diffs[:, 0], diffs[:, 1]) / dt
    # Assume 5 cm/px GSD -> 0.05 m/px -> * 3.6 for km/h
    speeds_kmh = np.concatenate(([0.0], speeds_px_s * 0.05 * 3.6))

    # Compute heading during movement
    heading_deg, cardinal = compute_heading_bearing(diffs[10, 0], diffs[10, 1])
    print(f"Vehicle Motion Heading : {heading_deg:.1f}° ({cardinal} - Eastbound Flow)")
    assert cardinal == "E", f"Expected Eastbound heading, got {cardinal}"

    # Detect stoppage events
    stoppages = detect_dwell_and_stoppage(timestamps, speeds_kmh, speed_threshold=3.0, min_dwell_duration=1.0)
    print(f"\nDetected Stoppage Events ({len(stoppages)} found):")
    for s in stoppages:
        print(f"  Stoppage: {s['start_time_sec']}s to {s['end_time_sec']}s (Duration: {s['dwell_duration_sec']}s)")

    assert len(stoppages) == 1, "Expected exactly 1 stoppage event!"
    assert abs(stoppages[0]["dwell_duration_sec"] - 1.3) < 0.2, "Stoppage duration calculation inaccurate!"
    print("\n[SUCCESS] Time-series trajectory smoothing and dwell-time detection verified.")
    print()

def main():
    demonstrate_trajectory_time_series()
    print("=" * 70)
    print("PHASE 7 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
