"""
=============================================================================
AeroVision AI - Module 02: Model Quantization & Performance Benchmarking
=============================================================================
Teaches post-training dynamic quantization (PTQ) for edge CPU deployment:
1. Mathematical principles of affine integer quantization (FP32 -> INT8).
2. Dynamic quantization of PyTorch linear projection layers.
3. Measuring disk serialization footprint reduction.
4. Profiling inference latency, memory delta, and numerical degradation.
=============================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization


# =============================================================================
# 1. MATHEMATICAL QUANTIZATION FORMULATION (AFFINE MAPPING)
# =============================================================================
def quantize_fp32_to_int8_manual(tensor: np.ndarray):
    """
    Simulates affine asymmetric quantization from float32 to int8:
    Scale S = (max - min) / (255)
    Zero-point Z = round(-min / S) - 128
    q = clip(round(x / S) + Z, -128, 127)
    """
    t_min = float(np.min(tensor))
    t_max = float(np.max(tensor))

    # Avoid zero division
    scale = max((t_max - t_min) / 255.0, 1e-8)
    zero_point = int(np.round(-t_min / scale)) - 128
    zero_point = max(-128, min(127, zero_point))

    # Quantize
    q = np.clip(np.round(tensor / scale) + zero_point, -128, 127).astype(np.int8)

    # Dequantize: x_approx = (q - Z) * S
    dequant = (q.astype(np.float32) - zero_point) * scale
    return q, dequant, scale, zero_point


# =============================================================================
# 2. DEFINING DEEP DRONE DETECTOR HEAD MODEL
# =============================================================================
class DroneDetectorHead(nn.Module):
    """
    Multi-layer detection head for regression and classification:
    Linear(512, 1024) -> ReLU -> Linear(1024, 512) -> ReLU -> Linear(512, 10)
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim // 2, 8)  # 8 drone object classes
        self.regressor = nn.Linear(hidden_dim // 2, 4)   # 4 box coords

    def forward(self, x: torch.Tensor):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        cls_scores = self.classifier(x)
        bbox_coords = self.regressor(x)
        return cls_scores, bbox_coords


# =============================================================================
# 3. QUANTIZATION AND BENCHMARKING
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: AFFINE ASYMMETRIC QUANTIZATION MATHEMATICS")
    print("=" * 70)
    sample_weights = np.array([-2.45, -0.80, 0.05, 1.25, 3.80], dtype=np.float32)
    q_vals, deq_vals, scale, zp = quantize_fp32_to_int8_manual(sample_weights)

    print(f"Original FP32 Weights : {sample_weights}")
    print(f"Calculated Scale (S)  : {scale:.6f}")
    print(f"Calculated ZeroPt (Z) : {zp}")
    print(f"Quantized INT8 Tensor : {q_vals.tolist()} (Type: {q_vals.dtype})")
    print(f"Dequantized FP32 Appr : {np.round(deq_vals, 3).tolist()}")
    
    quant_error = np.mean(np.abs(sample_weights - deq_vals))
    print(f"Mean Quantization Err : {quant_error:.6f}")
    assert quant_error < 0.05

    print("\n" + "=" * 70)
    print("STEP 2: PYTORCH DYNAMIC POST-TRAINING QUANTIZATION (PTQ)")
    print("=" * 70)
    torch.manual_seed(42)
    fp32_model = DroneDetectorHead().eval()

    # Dynamic post-training quantization targets Linear weights to qint8
    int8_model = quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    print(f"FP32 Model Layer 1 : {type(fp32_model.fc1)}")
    print(f"INT8 Model Layer 1 : {type(int8_model.fc1)}")
    print("✓ Dynamic quantization successfully converted FP32 Linear to DynamicQuantizedLinear.")

    print("\n" + "=" * 70)
    print("STEP 3: SERIALIZATION FILE SIZE COMPARISON")
    print("=" * 70)
    os.makedirs(os.path.join("models", "export"), exist_ok=True)
    fp32_path = os.path.join("models", "export", "detector_fp32.pt")
    int8_path = os.path.join("models", "export", "detector_int8.pt")

    torch.save(fp32_model.state_dict(), fp32_path)
    torch.save(int8_model.state_dict(), int8_path)

    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    int8_size_kb = os.path.getsize(int8_path) / 1024.0
    reduction_percent = ((fp32_size_kb - int8_size_kb) / fp32_size_kb) * 100.0

    print(f"FP32 Model File Size : {fp32_size_kb:.2f} KB")
    print(f"INT8 Model File Size : {int8_size_kb:.2f} KB")
    print(f"Storage Footprint Cut: {reduction_percent:.1f}% Reduction")

    assert reduction_percent > 50.0, "Expected at least 50% storage compression"

    print("\n" + "=" * 70)
    print("STEP 4: INFERENCE LATENCY & NUMERICAL FIDELITY BENCHMARK")
    print("=" * 70)
    batch_input = torch.randn(32, 512)

    # Warmup
    for _ in range(10):
        _ = fp32_model(batch_input)
        _ = int8_model(batch_input)

    # FP32 Latency
    t0 = time.perf_counter()
    for _ in range(100):
        fp32_cls, fp32_box = fp32_model(batch_input)
    fp32_time_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    # INT8 Latency
    t0 = time.perf_counter()
    for _ in range(100):
        int8_cls, int8_box = int8_model(batch_input)
    int8_time_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    # Accuracy / Output difference
    diff_cls = torch.mean(torch.abs(fp32_cls - int8_cls)).item()
    diff_box = torch.mean(torch.abs(fp32_box - int8_box)).item()

    print(f"FP32 Latency (Batch 32) : {fp32_time_ms:.3f} ms")
    print(f"INT8 Latency (Batch 32) : {int8_time_ms:.3f} ms")
    print(f"Class Score Mean Delta  : {diff_cls:.6f}")
    print(f"Box Regression Mean Del : {diff_box:.6f}")

    assert diff_cls < 0.10, "Quantization divergence exceeded allowable margin"
    print("✓ Numerical delta is within acceptable domain tolerance (< 0.10).")

    print("\n[SUCCESS] Module 02 Model Quantization verified successfully!")


if __name__ == "__main__":
    main()
