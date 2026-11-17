"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 10: YOLO11 Architecture Deconstruction & Forward Pass Mechanics
=============================================================================

Domain Problem:
Architectural Anatomy of AeroVision AI's Core Detection Engine:
In this module, we dissect the actual production weights `models/weights/yolo11n.pt`
under PyTorch. We deconstruct its three macro systems:
1. CSPDarknet Backbone (Downsampling to P3, P4, P5 multi-scale feature maps)
2. PAN-FPN Neck (Feature aggregation blending fine spatial details with deep semantic context)
3. Decoupled Anchor-Free Detection Head (Direct regression via DFL + class logits)

Learning Objectives:
1. Load and inspect the PyTorch neural network graph inside `yolo11n.pt`.
2. Understand the Anchor-Free paradigm vs historical Anchor-Based YOLO models.
3. Deconstruct the output tensor shape (1, 84, 8400):
   - Why 84? (4 bounding box coordinates [cx, cy, w, h] + 80 COCO class probabilities).
   - Why 8400? The mathematical sum of spatial grid cells across P3, P4, and P5:
     (80x80) + (40x40) + (20x20) = 6,400 + 1,600 + 400 = 8,400 candidate anchor points!
4. Inspect the modern C3k2 and SPPF (Spatial Pyramid Pooling - Fast) modules.
5. Trace a single frame through the raw PyTorch model without high-level wrappers.
6. Demystify Distribution Focal Loss (DFL): How continuous coordinates are represented as probability distributions.
7. Connection to AeroVision AI: Calibrating confidence thresholds and NMS in Phase 4/5.
"""

import os
import torch
import torch.nn as nn
from ultralytics import YOLO

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def main():
    weights_path = "models/weights/yolo11n.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    section_header("1. LOADING YOLOV11N PYTORCH NEURAL GRAPH")
    yolo_wrapper = YOLO(weights_path)
    pt_model = yolo_wrapper.model  # Extract underlying torch.nn.Module
    
    total_params = sum(p.numel() for p in pt_model.parameters())
    trainable_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"Loaded Checkpoint:         {weights_path}")
    print(f"PyTorch Class Type:        {type(pt_model)}")
    print(f"File Size on Disk:         {model_size_mb:.2f} MB")
    print(f"Total Parameters:          {total_params:,} (~2.6M)")
    print(f"Trainable Parameters:      {trainable_params:,}")
    print(f"Total Registered Modules:  {len(list(pt_model.modules())):,}")

    # -------------------------------------------------------------------------
    section_header("2. MACRO ARCHITECTURE: BACKBONE, NECK & HEAD")
    # In YOLO11, the top-level layers are stored in pt_model.model (nn.Sequential)
    sequential_layers = pt_model.model
    print(f"Total Top-Level Stages:    {len(sequential_layers)} sequential blocks\n")
    
    print(f"{'Stage':<6} | {'Layer Type':<22} | {'Input From':<14} | {'Parameters':<10} | {'Purpose in Drone Vision'}")
    print("-" * 88)
    
    for i, layer in enumerate(sequential_layers):
        l_type = layer.__class__.__name__
        n_params = sum(p.numel() for p in layer.parameters())
        from_idx = str(getattr(layer, 'f', -1))
        
        # Categorize purpose based on stage index
        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if i == 0:
                purpose = "Stem: 640x640 -> 320x320 Downsample"
            elif i in [2, 4, 6]:
                purpose = f"Downsampling Conv (Stride 2)"
            elif i in [1, 3, 5, 7, 8]:
                purpose = "C3k2 / Bottleneck Feature Extraction"
            elif i == 10:
                purpose = "SPPF: Multi-scale Context Pooling"
            else:
                purpose = "Backbone Feature Processing"
        elif i in range(11, 23):
            purpose = "PAN-FPN Neck: Feature Concat & Upsampling"
        else:
            purpose = "Detect Head: Decoupled Box & Class Prediction"
            
        print(f"[{i:>2d}]   | {l_type:<22} | {from_idx:<14} | {n_params:>10,d} | {purpose}")

    # -------------------------------------------------------------------------
    section_header("3. FORWARD PASS EXECUTION & OUTPUT TENSOR DECONSTRUCTION")
    # Standard input size: (Batch=1, Channels=3, Height=640, Width=640)
    x_dummy = torch.randn(1, 3, 640, 640)
    
    pt_model.eval()
    with torch.no_grad():
        raw_output = pt_model(x_dummy)
        
    # The output is a tuple containing the decoded predictions tensor
    predictions_tensor = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    
    print(f"Raw PyTorch Output Tensor Shape: {predictions_tensor.shape}")
    print("Deconstructing Dimensions: (Batch=1, Features=84, Anchors=8400)")
    
    print("\n1. Why Exactly 84 Features per Prediction?")
    print("   - Index [0..3]:  4 Bounding Box Coordinates: [center_x, center_y, width, height]")
    print("   - Index [4..83]: 80 Class Probabilities for COCO objects (person, car, truck, bus...)")
    print("   - Total: 4 + 80 = 84 numbers per spatial candidate box!")

    print("\n2. Why Exactly 8400 Candidate Anchor Points?")
    print("   YOLO11 predicts objects across 3 multi-scale pyramid feature maps:")
    p3_cells = 80 * 80  # Stride 8:  640 / 8 = 80 -> 6,400 cells
    p4_cells = 40 * 40  # Stride 16: 640 / 16 = 40 -> 1,600 cells
    p5_cells = 20 * 20  # Stride 32: 640 / 32 = 20 ->   400 cells
    total_cells = p3_cells + p4_cells + p5_cells
    
    print(f"   - P3 Small Object Head  (Stride  8):  80 x 80 = {p3_cells:>5,d} candidate boxes (High spatial res)")
    print(f"   - P4 Medium Object Head (Stride 16):  40 x 40 = {p4_cells:>5,d} candidate boxes (Balanced context)")
    print(f"   - P5 Large Object Head  (Stride 32):  20 x 20 = {p5_cells:>5,d} candidate boxes (High receptive field)")
    print(f"   - Sum Across All Multi-Scale Heads:             {total_cells:>5,d} Total Prediction Vectors!")
    assert predictions_tensor.shape[2] == total_cells, "Spatial grid sum mismatch!"
    print("   --> Mathematical verification PASSED: 6,400 + 1,600 + 400 = 8,400.")

    # -------------------------------------------------------------------------
    section_header("4. THE ANCHOR-FREE ADVANTAGE FOR DRONE ANALYTICS")
    print("""
Historical Anchor-Based YOLO (v2 through v7):
  - Every grid cell evaluated 3 to 9 predefined anchor boxes of fixed aspect ratios [w, h].
  - Flaw for Drones: Small aerial objects (e.g., a person walking at 120m altitude covers only
    6x6 pixels, with IoU < 0.1 against standard ground-level anchor priors).
  - High sensitivity to anchor hyperparameter tuning; required K-Means clustering on dataset boxes.

Modern Anchor-Free YOLO11:
  - Eliminates predefined anchor boxes completely.
  - Predicts distance offsets directly from each grid cell center [cx, cy] to the 4 box edges:
    left (l), top (t), right (r), bottom (b).
  - Uses Distribution Focal Loss (DFL): Instead of regressing a single noisy coordinate,
    it models coordinates as a discrete probability distribution over possible offsets,
    providing exceptional robustness to motion blur and variable drone flight altitudes!
""")

    # -------------------------------------------------------------------------
    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Explain what the tensor shape `[1, 84, 8400]` means in YOLO11 inference.
A1: - `1`: Batch size (1 image evaluated).
    - `84`: The feature channels for each prediction. The first 4 channels represent the bounding box
      regression parameters [center_x, center_y, width, height]. The remaining 80 channels represent
      the class logits / probabilities for the 80 COCO dataset classes.
    - `8400`: The total number of candidate prediction anchors across the three multi-scale detection heads:
      - Head 1 (P3, stride 8):  80 x 80 = 6,400 anchors for small objects.
      - Head 2 (P4, stride 16): 40 x 40 = 1,600 anchors for medium objects.
      - Head 3 (P5, stride 32): 20 x 20 =   400 anchors for large objects.
      Total: 6,400 + 1,600 + 400 = 8,400 anchors.
      During post-processing, Non-Maximum Suppression (NMS) filters these 8,400 candidates down
      to the few actual detected objects (e.g. 5 objects in bus.jpg) by removing low-confidence
      and highly overlapping redundant boxes.

Q2: What is SPPF (Spatial Pyramid Pooling - Fast) and why is it placed at the end of the backbone?
A2: SPPF aggregates spatial context across multiple receptive field scales without reducing spatial resolution:
    - It passes the deep P5 feature map through consecutive 5x5 MaxPooling layers with stride 1.
    - Two consecutive 5x5 maxpools are mathematically equivalent to a 9x9 maxpool; three consecutive 5x5 pools
      are equivalent to a 13x13 maxpool, but execute ~2.5x faster due to smaller kernel cache locality.
    - The original feature map is concatenated with the pooled representations.
    - Placed at the end of the backbone (Stage 10), SPPF gives the deepest feature map a global receptive field,
      allowing the network to understand broad environmental context (e.g. distinguishing a parking lot from a highway)
      before features enter the PAN-FPN neck.
""")

if __name__ == "__main__":
    main()
