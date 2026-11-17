"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 03: Neural Network Architectures, Activations & Initializations
=============================================================================

Domain Problem:
Designing a Multi-Task Drone Detection Head:
In aerial object detection, a deep neural network processes intermediate spatial
feature maps and branches into two simultaneous heads:
1. Classification Head: Predicts object class probabilities (pedestrian, car, truck, drone).
2. Bounding Box Head: Regresses 4 spatial coordinates (cx, cy, w, h).

Learning Objectives:
1. Subclassing `nn.Module` and implementing custom forward passes.
2. Layer mechanics: `nn.Linear`, weights, biases, and internal shapes.
3. Why Non-Linearity is Essential: Mathematically proving linear collapse (W2 * W1 = W_eff).
4. Modern Activation Functions: ReLU, LeakyReLU, Sigmoid, and SiLU (Swish) used in YOLO11.
5. Parameter Counting: Total vs Trainable parameters.
6. Weight Initialization: Xavier/Glorot vs Kaiming/He initialization.
7. Multi-Task Neural Network Architecture: Joint classification & box regression.
8. Connection to AeroVision AI: Deconstructing YOLO11's decoupled detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. MATHEMATICAL PROOF: LINEAR COLLAPSE WITHOUT ACTIVATIONS
# =============================================================================
class LinearCollapseNetwork(nn.Module):
    """3 linear layers without any non-linear activation functions."""
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x):
        # x -> W1 -> W2 -> W3 -> out
        return self.fc3(self.fc2(self.fc1(x)))

def verify_linear_collapse():
    # If there are no activations: y = ((x * W1^T) * W2^T) * W3^T = x * (W3 * W2 * W1)^T
    # This means a 3-layer deep network is strictly equivalent to a single linear layer!
    net = LinearCollapseNetwork(in_features=4, hidden=16, out_features=2)
    x = torch.randn(5, 4)

    # Multi-layer forward pass
    y_multi = net(x)

    # Equivalent single effective weight matrix: W_eff = W3 @ W2 @ W1
    W1 = net.fc1.weight  # shape: (16, 4)
    W2 = net.fc2.weight  # shape: (16, 16)
    W3 = net.fc3.weight  # shape: (2, 16)
    W_eff = W3 @ W2 @ W1 # shape: (2, 4)

    # Single-layer forward pass
    y_single = x @ W_eff.T

    discrepancy = torch.max(torch.abs(y_multi - y_single)).item()
    print(f"Max discrepancy between 3-layer linear net and 1-layer matrix: {discrepancy:.8f}")
    assert discrepancy < 1e-5, "Linear collapse proof failed!"
    print("--> Mathematically PROVED: Deep networks without non-linearities collapse to 1 layer!")

# =============================================================================
# 2. ACTIVATION FUNCTIONS: FROM RELU TO SILU (SWISH)
# =============================================================================
def compare_activations():
    z = torch.linspace(-4.0, 4.0, 9)
    print(f"Input z:             {[round(val.item(), 2) for val in z]}")
    
    # 1. ReLU: max(0, z)
    relu = F.relu(z)
    print(f"ReLU(z):             {[round(val.item(), 2) for val in relu]}")
    
    # 2. LeakyReLU: max(0.1*z, z) -> prevents dying neurons
    lrelu = F.leaky_relu(z, negative_slope=0.1)
    print(f"LeakyReLU(z, 0.1):   {[round(val.item(), 2) for val in lrelu]}")
    
    # 3. Sigmoid: 1 / (1 + exp(-z)) -> output probability
    sig = torch.sigmoid(z)
    print(f"Sigmoid(z):          {[round(val.item(), 2) for val in sig]}")
    
    # 4. SiLU (Swish): z * sigmoid(z) -> YOLO11 standard activation
    silu = F.silu(z)
    print(f"SiLU/Swish(z):       {[round(val.item(), 2) for val in silu]}")
    print("\nWhy YOLO uses SiLU: Smooth gradient everywhere, non-monotonic (slight dip below 0 for negative z),")
    print("which regularizes small negative activations and accelerates deep gradient flow.")

# =============================================================================
# 3. MULTI-TASK DRONE DETECTION HEAD ARCHITECTURE
# =============================================================================
class DroneDecoupledHead(nn.Module):
    """
    Decoupled detection head mirroring YOLO11's architecture:
    A shared feature embedding branches into:
    1. Bounding Box Regression Head (outputs 4 normalized coordinates)
    2. Classification Head (outputs num_classes class logits)
    """
    def __init__(self, in_features=128, hidden_dim=64, num_classes=5):
        super().__init__()
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Branch 1: Box Regression Head [cx, cy, w, h]
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # Normalized coordinates bounded to [0.0, 1.0]
        )
        
        # Branch 2: Class Head (raw logits)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights with Kaiming / He normal
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming He normal initialization for SiLU/ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feat = self.trunk(x)
        pred_boxes = self.box_head(feat)      # Shape: (Batch, 4) in [0, 1]
        pred_logits = self.cls_head(feat)     # Shape: (Batch, num_classes)
        return pred_boxes, pred_logits

# =============================================================================
# 4. EXECUTION & VERIFICATION
# =============================================================================
def main():
    section_header("1. LINEAR COLLAPSE VERIFICATION")
    verify_linear_collapse()

    section_header("2. ACTIVATION FUNCTION BEHAVIOR & SILU (YOLO STANDARD)")
    compare_activations()

    section_header("3. DECOUPLED DRONE DETECTION HEAD INITIALIZATION")
    head = DroneDecoupledHead(in_features=128, hidden_dim=64, num_classes=5)
    
    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Drone Decoupled Head Parameter Summary:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Layer breakdown
    print("\nLayer Architecture & Parameter Tensor Shapes:")
    for name, param in head.named_parameters():
        print(f"  {name:<30} shape={str(list(param.shape)):<18} params={param.numel():>6d}")

    # Simulated batch forward pass: 4 drone object crops
    x_crops = torch.randn(4, 128)  # Batch of 4 visual feature vectors
    boxes, logits = head(x_crops)

    section_header("4. FORWARD PASS OUTPUTS & MULTI-TASK PREDICTIONS")
    print(f"Input Feature Tensor Shape:    {x_crops.shape}")
    print(f"Predicted Box Coordinates:     {boxes.shape} (Range [{boxes.min().item():.3f}, {boxes.max().item():.3f}])")
    print(f"Predicted Class Logits Shape:  {logits.shape}")
    
    probs = F.softmax(logits, dim=-1)
    print("\nSample Detections for Crop 0:")
    print(f"  Box Coordinates (cx, cy, w, h): {[round(v, 3) for v in boxes[0].tolist()]}")
    print(f"  Class Probabilities:            {[round(v, 3) for v in probs[0].tolist()]}")
    print(f"  Top Predicted Class:            {torch.argmax(probs[0]).item()} with confidence {probs[0].max().item():.3f}")

    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why does YOLO11 use a "decoupled" head (separate branches for box regression and classification) rather than a shared head?
A1: In earlier YOLO architectures (YOLOv1 through YOLOv5), a single linear/conv layer predicted both box coordinates [x, y, w, h]
    and class probabilities simultaneously from the same shared parameters.
    However, object classification and bounding box regression are fundamentally conflicting tasks:
    - Classification requires translation-invariant features (a car is a car regardless of where it appears in the crop).
    - Localization requires translation-equivariant features (the box coordinates must change precisely with object shift).
    Forcing both tasks to share the final convolutional weights creates negative task interference.
    Decoupled heads (introduced in YOLOX and perfected in YOLOv8/11) split the representation into dedicated sub-networks,
    accelerating convergence and boosting mAP by 1.5 - 2.5%.

Q2: Why is Kaiming/He initialization preferred over Xavier/Glorot initialization for modern CNNs?
A2: Xavier initialization assumes linear or symmetric zero-centered activations (e.g., tanh, sigmoid) and maintains variance
    Var(W) = 2 / (fan_in + fan_out).
    However, rectifier activations (ReLU, LeakyReLU, SiLU) set half of the negative inputs to zero, cutting the output variance in half.
    Stacking 20 ReLU layers with Xavier initialization causes signal variance to shrink exponentially by (0.5)^20,
    leading to vanishing forward signals.
    Kaiming initialization accounts for this by doubling the weight variance: Var(W) = 2 / fan_in,
    guaranteeing that signal variance remains constant across arbitrarily deep networks.
""")

if __name__ == "__main__":
    main()
