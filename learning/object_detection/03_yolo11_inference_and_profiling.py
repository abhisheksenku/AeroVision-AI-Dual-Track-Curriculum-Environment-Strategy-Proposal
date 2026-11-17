"""
=============================================================================
AeroVision AI - Object Detection Foundations: Module 03
=============================================================================
YOLO11 Inference, CPU Latency Profiling, and Aerial CV Physics

Learning Objectives:
1. Run CPU inference using YOLO11n on realistic imagery.
2. Calculate Ground Sampling Distance (GSD) and object pixel footprints across drone altitudes.
3. Understand why aerial imagery requires specialized computer vision architectures.
4. Profile inference latency, standard deviation, and throughput (FPS) on Intel Core 5 CPU.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

def calculate_aerial_gsd_and_footprint():
    """
    Computes Ground Sampling Distance (GSD) and apparent pixel footprint
    of vehicles and pedestrians at varying drone flight altitudes.
    
    Formula:
      GSD_horizontal = (Flight_Altitude * Sensor_Width) / (Focal_Length * Image_Width)
                     = (Flight_Altitude * Pixel_Pitch) / Focal_Length
    """
    print("=" * 70)
    print("STEP 1: AERIAL SENSOR OPTICS & GROUND SAMPLING DISTANCE (GSD)")
    print("=" * 70)

    # Typical commercial drone camera specs (e.g. DJI Mavic 3 Enterprise / Matrice)
    focal_length_mm = 12.0       # 12mm lens
    sensor_width_mm = 17.3       # 4/3-inch CMOS sensor
    sensor_height_mm = 13.0
    image_w_px = 3840            # 4K resolution
    image_h_px = 2160

    pixel_pitch_um = (sensor_width_mm / image_w_px) * 1000.0 # ~4.5 micrometers

    # Standard real-world target dimensions (meters)
    car_length_m = 4.5
    car_width_m = 1.8
    person_height_m = 1.75
    person_width_m = 0.5

    altitudes_m = [20, 50, 100, 150]

    print(f"Drone Camera Specs: Focal Length={focal_length_mm}mm, Sensor={sensor_width_mm}x{sensor_height_mm}mm, 4K ({image_w_px}x{image_h_px})")
    print(f"Sensor Pixel Pitch: {pixel_pitch_um:.2f} um/pixel\n")
    print(f"{'Altitude':<10} | {'GSD (cm/px)':<12} | {'Car (px)':<15} | {'Person (px)':<15} | {'Detection Feasibility'}")
    print("-" * 75)

    for alt in altitudes_m:
        # GSD in cm per pixel
        gsd_cm = (alt * sensor_width_mm / (focal_length_mm * image_w_px)) * 100.0
        
        # Dimensions in pixels on native 4K sensor
        car_px_l = (car_length_m * 100.0) / gsd_cm
        car_px_w = (car_width_m * 100.0) / gsd_cm
        
        person_px_h = (person_height_m * 100.0) / gsd_cm
        person_px_w = (person_width_m * 100.0) / gsd_cm

        if alt <= 50:
            status = "EXCELLENT (Standard CNN)"
        elif alt <= 100:
            status = "CHALLENGING (Needs P2/P3 High-Res)"
        else:
            status = "CRITICAL (Requires Tiling/SAHI)"

        print(f"{alt:>4} m     | {gsd_cm:>6.2f}       | {car_px_l:>5.0f} x {car_px_w:<4.0f}    | {person_px_h:>5.0f} x {person_px_w:<4.0f}    | {status}")

    print("\nTakeaway: At 150m altitude, a pedestrian occupies only ~14x4 pixels on a 4K frame.")
    print("If downsampled directly to 640x640, the pedestrian shrinks to ~2x1 pixels and disappears!")
    print("AeroVision AI utilizes high-resolution multi-scale feature pyramids (P3, P4, P5) to mitigate this.")
    print()

def benchmark_yolo11_cpu(model_path: str, sample_image_path: str, num_runs: int = 10):
    """
    Profiles YOLO11n inference latency, variance, and FPS on CPU.
    """
    print("=" * 70)
    print("STEP 2: YOLO11n CPU INFERENCE & THROUGHPUT BENCHMARK")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    if not os.path.exists(sample_image_path):
        raise FileNotFoundError(f"Sample image not found at: {sample_image_path}")

    # Load model
    model = YOLO(model_path)
    img = cv2.imread(sample_image_path)

    print(f"Loaded Weights   : {model_path}")
    print(f"Inference Device : CPU")
    print(f"Benchmark Image  : {sample_image_path} ({img.shape[1]}x{img.shape[0]} px)")
    print(f"Target Classes   : COCO (80 classes: person, bicycle, car, motorcycle, bus, truck, etc.)")

    # Warmup run (compiles PyTorch kernel graphs and caches memory allocations)
    _ = model.predict(source=img, device="cpu", verbose=False)

    latencies = []
    detection_counts = []

    for i in range(num_runs):
        t0 = time.perf_counter()
        results = model.predict(
            source=img,
            conf=0.25,
            iou=0.45,
            device="cpu",
            verbose=False
        )
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        
        boxes = results[0].boxes
        num_dets = len(boxes) if boxes is not None else 0
        detection_counts.append(num_dets)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"\nBenchmark Results across {num_runs} consecutive runs:")
    print(f"  Mean Latency     : {mean_lat:.2f} ms")
    print(f"  Std Deviation    : {std_lat:.2f} ms")
    print(f"  Min / Max Latency: {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  Throughput       : {fps:.1f} FPS")
    print(f"  Detections Found : {detection_counts[0]} objects in test scene")

    # Display detected objects
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = model.names
        print(f"\nDetected Entities:")
        for idx in range(min(5, len(boxes))):
            cls_id = int(boxes.cls[idx].cpu().numpy())
            conf = float(boxes.conf[idx].cpu().numpy())
            xyxy = boxes.xyxy[idx].cpu().numpy()
            print(f"  - {names[cls_id]:<10} (Conf: {conf:.2f}) at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

    print("\n[SUCCESS] YOLO11n CPU latency and inference pipeline verified.")

def main():
    calculate_aerial_gsd_and_footprint()
    benchmark_yolo11_cpu(
        model_path="models/weights/yolo11n.pt",
        sample_image_path="data/samples/bus.jpg",
        num_runs=10
    )
    print("=" * 70)
    print("PHASE 5 - MODULE 03: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
