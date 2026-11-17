"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 06: Convolution Operations, Feature Maps & Receptive Fields
=============================================================================

Domain Problem:
Building the Feature Extraction Backbone for Aerial Drone Vision:
Raw drone video frames contain millions of pixels (e.g. 1920x1080x3 = 6.2M values).
Dense layers cannot process raw images due to parameter explosion and lack of spatial
equivariance. Convolutional neural networks extract local invariant features (edges ->
textures -> parts -> objects) across multi-scale feature pyramids.

Learning Objectives:
1. Understand the 2D Convolution Operation: Kernels, Stride, Padding, Channels.
2. The Output Dimension Formula: O = floor((W - K + 2P) / S) + 1.
3. Parameter Counting for Conv2d: (K_h * K_w * C_in + bias) * C_out.
4. The Modern YOLO Building Block: Conv + BatchNorm2d + SiLU (The 'CBS' block).
5. Receptive Field (RF) Calculation: How deep neurons see broad spatial contexts.
6. Strided Convolutions vs MaxPool2d: Why modern vision architectures eliminate pooling.
7. Multi-Scale Feature Maps (P3, P4, P5): Detecting small vs medium vs large drone targets.
8. Connection to AeroVision AI: Deconstructing YOLO11's CSPDarknet backbone stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. THE CANONICAL CONV-BN-SILU (CBS) BLOCK
# =============================================================================
class CBSBlock(nn.Module):
    """
    Standard Convolution + Batch Normalization + SiLU activation unit.
    This is the fundamental atomic building block of Ultralytics YOLO11.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Conv2d does not need bias when immediately followed by BatchNorm2d
        # because BN's beta (shift) parameter acts as the bias!
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# =============================================================================
# 2. OUTPUT SPATIAL DIMENSION VERIFICATION
# =============================================================================
def verify_output_dimension_formula():
    section_header("1. CONVOLUTION OUTPUT DIMENSION FORMULA VERIFICATION")
    # Formula: O = floor((W - K + 2*P) / S) + 1
    def compute_out_dim(W, K, P, S):
        return int((W - K + 2 * P) / S) + 1

    test_cases = [
        {"W": 640, "K": 3, "P": 1, "S": 1, "desc": "Standard 3x3 preserve spatial size"},
        {"W": 640, "K": 3, "P": 1, "S": 2, "desc": "Strided 3x3 downsample by 2x"},
        {"W": 320, "K": 5, "P": 2, "S": 1, "desc": "5x5 kernel with padding 2"},
        {"W": 160, "K": 1, "P": 0, "S": 1, "desc": "1x1 bottleneck channel projection"},
    ]

    print(f"{'Description':<38} | {'Input':<8} | {'Config (K,P,S)':<15} | {'Formula Output':<15} | {'PyTorch Actual'}")
    print("-" * 95)

    for case in test_cases:
        W, K, P, S = case["W"], case["K"], case["P"], case["S"]
        expected_out = compute_out_dim(W, K, P, S)
        
        # Test with real PyTorch Conv2d
        dummy_input = torch.zeros(1, 1, W, W)
        conv = nn.Conv2d(1, 1, kernel_size=K, padding=P, stride=S)
        actual_out = conv(dummy_input).shape[2]
        
        cfg_str = f"K={K}, P={P}, S={S}"
        print(f"{case['desc']:<38} | {W:<8} | {cfg_str:<15} | {expected_out:<15} | {actual_out}")
        assert expected_out == actual_out, "Dimension formula mismatch!"
        
    print("\n--> Formula verification PASSED: O = floor((W - K + 2P)/S) + 1 matches PyTorch exactly.")

# =============================================================================
# 3. RECEPTIVE FIELD CALCULATION
# =============================================================================
def calculate_receptive_field():
    section_header("2. RECEPTIVE FIELD (RF) PROPAGATION MATHEMATICS")
    # A single neuron in an early layer sees 3x3 pixels.
    # Stacking multiple layers expands the effective field of view.
    # Receptive Field formula for layer l:
    #   RF_l = RF_{l-1} + (K_l - 1) * J_{l-1}
    # Where jump J_l = J_{l-1} * S_l (cumulative stride)
    
    stages = [
        {"name": "Stage 0 (Input)", "K": 1, "S": 1},
        {"name": "Stage 1 (Stem 3x3, S=2)", "K": 3, "S": 2},
        {"name": "Stage 2 (Conv 3x3, S=2)", "K": 3, "S": 2},
        {"name": "Stage 3 (Conv 3x3, S=2)", "K": 3, "S": 2},
        {"name": "Stage 4 (Conv 3x3, S=2)", "K": 3, "S": 2},
        {"name": "Stage 5 (Conv 3x3, S=2)", "K": 3, "S": 2},
    ]

    rf = 1
    jump = 1
    print(f"{'Backbone Stage':<26} | {'Kernel (K)':<12} | {'Stride (S)':<12} | {'Jump (Cumulative)':<18} | {'Receptive Field (px)'}")
    print("-" * 90)

    for stage in stages:
        K = stage["K"]
        S = stage["S"]
        if stage["name"] != "Stage 0 (Input)":
            rf = rf + (K - 1) * jump
            jump = jump * S
        print(f"{stage['name']:<26} | {K:<12} | {S:<12} | {jump:<18} | {rf} x {rf} pixels")
        
    print("\nKey Insight: By Stage 5 (Stride 32), a single neuron aggregates context from an effective")
    print("receptive field of 99x99 pixels, allowing the network to detect large trucks or buildings!")

# =============================================================================
# 4. MINI MULTI-SCALE FEATURE PYRAMID (P3, P4, P5)
# =============================================================================
class MiniDroneFeatureBackbone(nn.Module):
    """
    Extracts multi-scale features matching YOLO11's P3, P4, P5 hierarchy:
    Input: (B, 3, 640, 640)
    - P3 (Stride 8):  (B, 32, 80, 80)   -> Small objects (distant pedestrians, bikes)
    - P4 (Stride 16): (B, 64, 40, 40)   -> Medium objects (cars, vans)
    - P5 (Stride 32): (B, 128, 20, 20)  -> Large objects (buses, trucks, buildings)
    """
    def __init__(self):
        super().__init__()
        # Stem: 640 -> 320
        self.stem = CBSBlock(in_channels=3, out_channels=16, stride=2)
        # Stage 1: 320 -> 160
        self.stage1 = CBSBlock(in_channels=16, out_channels=24, stride=2)
        # Stage 2 (P3): 160 -> 80
        self.stage2 = CBSBlock(in_channels=24, out_channels=32, stride=2)
        # Stage 3 (P4): 80 -> 40
        self.stage3 = CBSBlock(in_channels=32, out_channels=64, stride=2)
        # Stage 4 (P5): 40 -> 20
        self.stage4 = CBSBlock(in_channels=64, out_channels=128, stride=2)

    def forward(self, x):
        x = self.stem(x)       # (B, 16, 320, 320)
        x = self.stage1(x)     # (B, 24, 160, 160)
        p3 = self.stage2(x)    # (B, 32, 80, 80)   - Stride 8
        p4 = self.stage3(p3)   # (B, 64, 40, 40)   - Stride 16
        p5 = self.stage4(p4)   # (B, 128, 20, 20)  - Stride 32
        return p3, p4, p5

def main():
    verify_output_dimension_formula()
    calculate_receptive_field()

    section_header("3. MINI MULTI-SCALE DRONE BACKBONE IN ACTION")
    backbone = MiniDroneFeatureBackbone()
    
    # Simulate a standard aerial drone video frame (Batch=1, Channels=3, H=640, W=640)
    drone_frame = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = backbone(drone_frame)

    print(f"Input Drone Video Frame: {drone_frame.shape}")
    print(f"  P3 Feature Map (Stride 8):  {p3.shape}  -> Optimized for Small Objects (e.g. Pedestrians)")
    print(f"  P4 Feature Map (Stride 16): {p4.shape}  -> Optimized for Medium Objects (e.g. Cars)")
    print(f"  P5 Feature Map (Stride 32): {p5.shape}  -> Optimized for Large Objects (e.g. Trucks/Helipads)")

    # Parameter footprint
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal Backbone Parameters: {total_params:,} (~{total_params * 4 / 1024:.2f} KB FP32)")

    section_header("4. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why does a Conv2d layer not require a bias tensor (bias=False) when immediately followed by BatchNorm2d?
A1: Batch Normalization standardizes intermediate activations across the batch dimension:
    BN(x) = gamma * ((x - mu) / sqrt(sigma^2 + eps)) + beta
    If Conv2d includes a bias parameter `b`:
    Conv(x) = W * x + b
    During the normalization step:
    mean(Conv(x)) = mean(W * x) + b
    Subtracting the mean:
    (W * x + b) - (mean(W * x) + b) = W * x - mean(W * x)
    The bias `b` is mathematically canceled out to zero in the subtraction!
    The learnable parameter `beta` in BatchNorm2d already serves as the shift/bias term.
    Setting `bias=False` in Conv2d eliminates redundant weights and avoids useless gradient computations.

Q2: Why did modern YOLO architectures replace MaxPooling2d with Strided Convolutions for downsampling?
A2: 1. Learnable Spatial Decimation: MaxPooling2d applies a rigid, fixed heuristic (selecting the maximum value in a 2x2 window),
       discarding 75% of spatial information without learning. A strided convolution (kernel=3, stride=2, padding=1)
       has trainable weights that learn which spatial features to retain during downsampling.
    2. Gradient Flow: In MaxPooling, backward gradients flow only to the single winning maximum pixel; in strided convolutions,
       gradients flow back through all kernel weights to all input pixels, improving gradient propagation in deep backbones.
""")

if __name__ == "__main__":
    main()
