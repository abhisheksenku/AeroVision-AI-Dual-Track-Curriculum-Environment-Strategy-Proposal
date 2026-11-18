"""
=============================================================================
AeroVision AI - Module 01: Model Export & ONNX Runtime Inference
=============================================================================
Demonstrates production model serialization and hardware-optimized inference:
1. Defining a convolutional drone vision feature extractor in PyTorch.
2. Exporting PyTorch computation graph to Open Neural Network Exchange (ONNX).
3. Initializing ONNX Runtime with CPUExecutionProvider.
4. Verifying mathematical parity between PyTorch and ONNX Runtime outputs.
5. Benchmarking throughput (FPS) and latency speedup.
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DEFINE DRONE VISION CONVNET BACKBONE
# =============================================================================
class AeroVisionDroneBackbone(nn.Module):
    """
    Lightweight CNN backbone simulating aerial drone object feature extraction:
    Conv2D -> BatchNorm -> SiLU -> MaxPool -> Residual Block -> GlobalAvgPool -> Linear
    """
    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, num_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 4)  # Bounding box coordinates regression [cx, cy, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage(feat)
        return self.head(feat)


# =============================================================================
# 2. MODEL EXPORT TO ONNX
# =============================================================================
def export_pytorch_to_onnx(model: nn.Module, export_path: str, dummy_input: torch.Tensor):
    """
    Exports PyTorch model graph to standard ONNX representation with dynamic axes.
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_tensor"],
        output_names=["bbox_predictions"],
        dynamic_axes={
            "input_tensor": {0: "batch_size"},
            "bbox_predictions": {0: "batch_size"}
        },
        dynamo=False
    )
    file_size_kb = os.path.getsize(export_path) / 1024.0
    return file_size_kb


# =============================================================================
# 3. VERIFY NUMERICAL PARITY & BENCHMARK
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: INITIALIZING PYTORCH DRONE VISION BACKBONE")
    print("=" * 70)
    torch.manual_seed(42)
    np.random.seed(42)

    model = AeroVisionDroneBackbone()
    model.eval()

    # Input: 1 frame of resolution 640x640 with 3 color channels
    dummy_input = torch.randn(1, 3, 320, 320, dtype=torch.float32)

    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    print(f"PyTorch Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input Shape             : {list(dummy_input.shape)}")
    print(f"Output Shape            : {list(pytorch_output.shape)}")

    print("\n" + "=" * 70)
    print("STEP 2: EXPORTING MODEL GRAPH TO ONNX")
    print("=" * 70)
    export_path = os.path.join("models", "export", "drone_backbone.onnx")
    size_kb = export_pytorch_to_onnx(model, export_path, dummy_input)
    print(f"Exported ONNX Model to  : {export_path}")
    print(f"Serialized File Size    : {size_kb:.2f} KB")

    print("\n" + "=" * 70)
    print("STEP 3: EXECUTING INFERENCE IN ONNX RUNTIME")
    print("=" * 70)
    # Configure ONNX Runtime Session Options
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(
        export_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ort_inputs = {input_name: dummy_input.numpy()}
    ort_output = session.run([output_name], ort_inputs)[0]

    print(f"ONNX Runtime Execution Provider: {session.get_providers()[0]}")
    print(f"ONNX Output Tensor Shape       : {list(ort_output.shape)}")

    print("\n" + "=" * 70)
    print("STEP 4: MATHEMATICAL NUMERICAL PARITY AUDIT")
    print("=" * 70)
    max_abs_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_abs_diff = np.mean(np.abs(pytorch_output - ort_output))
    print(f"Max Absolute Error  : {max_abs_diff:.8e}")
    print(f"Mean Absolute Error : {mean_abs_diff:.8e}")

    assert max_abs_diff < 1e-4, f"Numerical parity failed! Max diff {max_abs_diff} >= 1e-4"
    print("✓ Numerical parity verified! ONNX Runtime predictions match PyTorch within float32 tolerance.")

    print("\n" + "=" * 70)
    print("STEP 5: INFERENCE LATENCY BENCHMARKING (50 ITERATIONS)")
    print("=" * 70)
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)
        _ = session.run([output_name], ort_inputs)

    # Benchmark PyTorch
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    pytorch_time_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

    # Benchmark ONNX Runtime
    t0 = time.perf_counter()
    for _ in range(50):
        _ = session.run([output_name], ort_inputs)
    ort_time_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

    speedup = pytorch_time_ms / ort_time_ms
    print(f"PyTorch Eager CPU Latency   : {pytorch_time_ms:.2f} ms ({1000.0/pytorch_time_ms:.1f} FPS)")
    print(f"ONNX Runtime CPU Latency    : {ort_time_ms:.2f} ms ({1000.0/ort_time_ms:.1f} FPS)")
    print(f"Acceleration Speedup Factor : {speedup:.2f}x")

    print("\n[SUCCESS] Module 01 Model Export & ONNX Runtime verified successfully!")


if __name__ == "__main__":
    main()
