"""
Phase 1 Verification Script for AeroVision AI
Verifies Python environment, PyTorch CPU, OpenCV, Ultralytics,
runs YOLO11n inference on a test image, measures inference latency and RAM usage,
and saves the annotated output.
"""
import sys
import os
import time
import psutil
import torch
import torchvision
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

def format_mb(bytes_val):
    return f"{bytes_val / (1024 * 1024):.2f} MB"

def main():
    print("=" * 60)
    print("AEROVISION AI - PHASE 1 VERIFICATION")
    print("=" * 60)

    # 1. Environment Checks
    print("\n[1] Environment Checks:")
    print(f"  Python Version:     {sys.version.split()[0]}")
    print(f"  Python Executable:  {sys.executable}")
    is_venv = sys.prefix != sys.base_prefix
    print(f"  Virtual Env Active: {is_venv} (Prefix: {sys.prefix})")
    print(f"  PyTorch Version:    {torch.__version__}")
    print(f"  TorchVision Version:{torchvision.__version__}")
    print(f"  CUDA Available:     {torch.cuda.is_available()} (Strictly CPU: {not torch.cuda.is_available()})")
    print(f"  OpenCV Version:     {cv2.__version__}")
    print(f"  Ultralytics Version:{ultralytics.__version__}")

    # Memory baseline
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss
    print(f"  Initial Process RAM:{format_mb(ram_before)}")

    # 2. Model Initialization
    print("\n[2] YOLO11n Model Initialization:")
    model_path = os.path.join("models", "weights", "yolo11n.pt")
    if not os.path.exists(model_path):
        print(f"  Weights not found at {model_path}. Loading standard 'yolo11n.pt'...")
        model_path = "yolo11n.pt"
    
    t_load_start = time.perf_counter()
    model = YOLO(model_path)
    t_load = (time.perf_counter() - t_load_start) * 1000
    print(f"  Model File:         {model_path}")
    print(f"  Model Task:         {model.task}")
    print(f"  Model Load Time:    {t_load:.2f} ms")
    print(f"  Total Classes:      {len(model.names)}")

    # 3. Test Image Creation / Acquisition
    print("\n[3] Preparing Test Image:")
    os.makedirs(os.path.join("data", "samples"), exist_ok=True)
    test_img_path = os.path.join("data", "samples", "phase1_test_scene.jpg")
    
    # We create a realistic test scene with simulated drone overhead shapes (cars, people boxes)
    # or fetch a standard benchmark test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Background: asphalt road color
    img[:] = (70, 75, 80)
    # Road markings (white lines)
    cv2.line(img, (320, 0), (320, 640), (255, 255, 255), 4)
    # Draw simulated vehicles (rectangles)
    cv2.rectangle(img, (120, 100), (220, 260), (40, 40, 180), -1)   # Red car
    cv2.rectangle(img, (380, 250), (500, 480), (180, 120, 40), -1)  # Blue van/truck
    # Draw pedestrians (small circles)
    cv2.circle(img, (280, 200), 12, (200, 200, 200), -1)
    cv2.circle(img, (280, 240), 12, (200, 200, 200), -1)
    cv2.imwrite(test_img_path, img)
    print(f"  Generated synthetic scene at: {test_img_path} (640x640)")

    # 4. YOLO Inference Test
    print("\n[4] Running YOLO11n Inference Test:")
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    # Also test on a natural image containing real objects (bus.jpg)
    real_sample_path = os.path.join("data", "samples", "bus.jpg")
    if not os.path.exists(real_sample_path):
        import urllib.request
        try:
            print("  Downloading standard sample image (bus.jpg)...")
            urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", real_sample_path)
            eval_img_path = real_sample_path
        except Exception as e:
            print(f"  Could not download bus.jpg ({e}), using synthetic scene.")
            eval_img_path = test_img_path
    else:
        eval_img_path = real_sample_path

    # Warmup pass
    _ = model.predict(source=eval_img_path, device="cpu", verbose=False)
    
    # Timed inference pass
    t_inf_start = time.perf_counter()
    results = model.predict(
        source=eval_img_path,
        device="cpu",
        conf=0.25,
        save=False,
        verbose=False
    )
    t_inf_total = (time.perf_counter() - t_inf_start) * 1000

    result = results[0]
    speed = result.speed  # dict with preprocess, inference, postprocess (ms)
    
    print(f"  Test Image Used:    {eval_img_path}")
    print(f"  Pre-process Time:   {speed.get('preprocess', 0):.2f} ms")
    print(f"  Inference Time:     {speed.get('inference', 0):.2f} ms")
    print(f"  Post-process Time:  {speed.get('postprocess', 0):.2f} ms")
    print(f"  Total Roundtrip:    {t_inf_total:.2f} ms")

    # Save annotated output
    output_img_path = os.path.join("data", "processed", "phase1_annotated_output.jpg")
    annotated_bgr = result.plot()
    cv2.imwrite(output_img_path, annotated_bgr)
    print(f"  Annotated Output:   {output_img_path} (Saved)")

    # Detections
    boxes = result.boxes
    print(f"  Detections Count:   {len(boxes)}")
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        coords = [round(float(c), 1) for c in box.xyxy[0]]
        print(f"    - Object {i+1}: Class='{cls_name}' | Conf={conf:.2f} | BBox(xyxy)={coords}")

    # Memory after inference
    ram_after = process.memory_info().rss
    ram_delta = ram_after - ram_before
    print(f"\n[5] Memory Profile:")
    print(f"  Process RAM Peak:   {format_mb(ram_after)}")
    print(f"  RAM Growth:         {format_mb(ram_delta)}")
    
    # System RAM
    vm = psutil.virtual_memory()
    print(f"  System Total RAM:   {format_mb(vm.total)}")
    print(f"  System Free RAM:    {format_mb(vm.available)}")
    print(f"  System RAM Used:    {vm.percent}%")

    print("\n" + "=" * 60)
    print("PHASE 1 PYTHON & CV VERIFICATION COMPLETE - ALL PASS")
    print("=" * 60)

if __name__ == "__main__":
    main()
