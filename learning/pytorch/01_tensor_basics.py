"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 01: Tensor Semantics, Strides, Shapes & Broadcasting
=============================================================================

Domain Problem:
Transforming raw camera frames and bounding box coordinates into deep learning tensors:
1. Drone Frame Re-Ordering:
   OpenCV decodes video frames into NumPy arrays of shape (Height, Width, Channels: BGR).
   PyTorch CNN backbones strictly expect (Batch, Channels: RGB, Height, Width).
   Understanding strides, permute, and contiguous memory is essential to prevent silent data corruption.
2. Bounding Box Grid Broadcasting:
   In YOLO detection heads, anchor grid offsets [grid_x, grid_y] must be broadcast
   across thousands of predicted bounding boxes simultaneously.

Learning Objectives:
1. Tensor creation, dtypes (float32, int64, uint8), and device placement (CPU).
2. Understand Strides and Memory Layout: How multi-dimensional tensors are flattened in RAM.
3. Reshaping mechanics: .view() vs .reshape() vs .permute().
4. Why .contiguous() is required after transposition/permutation.
5. Tensor Broadcasting rules: Vectorized expansion without allocating duplicate RAM.
6. Zero-copy NumPy <-> PyTorch interoperability (shared underlying memory buffers).
7. Matrix multiplication: torch.matmul, @, and batch matrix multiplication (bmm).
8. Connection to AeroVision AI: Preprocessing 4K drone video frames for the YOLO11 input layer.
"""

import torch
import numpy as np

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def main():
    section_header("1. TENSOR CREATION, DTYPES & CPU MEMORY")
    # PyTorch default floating point type is float32 (single precision)
    t_float = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32)
    t_int   = torch.tensor([1, 2, 3], dtype=torch.int64)
    t_img   = torch.randint(0, 256, (3, 640, 640), dtype=torch.uint8)  # Standard 8-bit image

    print(f"Float Tensor:  shape={t_float.shape}, dtype={t_float.dtype}, device={t_float.device}")
    print(f"Int Tensor:    shape={t_int.shape}, dtype={t_int.dtype}")
    print(f"Image Tensor:  shape={t_img.shape}, dtype={t_img.dtype}, size={t_img.numel():,} elements")
    print(f"Memory allocated by 640x640x3 uint8 image: {t_img.element_size() * t_img.numel() / 1024:.2f} KB")

    # -------------------------------------------------------------------------
    section_header("2. TENSOR STRIDES & PHYSICAL MEMORY LAYOUT")
    # In C-contiguous layout, elements along the last dimension are stored adjacent in RAM.
    # Stride indicates how many physical memory locations to skip to move 1 step along a dimension.
    sample_2d = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("2D Matrix (3 rows x 4 cols):")
    print(sample_2d)
    print(f"Shape:  {sample_2d.shape}")
    print(f"Stride: {sample_2d.stride()} -> (Skip 4 elements for next row, skip 1 element for next col)")
    print(f"Is contiguous in memory: {sample_2d.is_contiguous()}")

    # Transposition swaps strides without copying data
    transposed = sample_2d.T
    print("\nTransposed Matrix (4 rows x 3 cols):")
    print(f"Shape:  {transposed.shape}")
    print(f"Stride: {transposed.stride()} -> (Skip 1 element for next row, skip 4 elements for next col!)")
    print(f"Is contiguous in memory: {transposed.is_contiguous()} (False: non-contiguous view)")

    # -------------------------------------------------------------------------
    section_header("3. RESHAPING: VIEW VS RESHAPE VS PERMUTE & CONTIGUOUS")
    # .view() requires contiguous memory; .reshape() will copy memory if non-contiguous
    try:
        # This will fail on non-contiguous transposed tensor
        _ = transposed.view(12)
        print("transposed.view(12) succeeded.")
    except RuntimeError as e:
        print(f"transposed.view(12) raised RuntimeError as expected:")
        print(f"  --> {e}")
        print("Fix: Call .contiguous() first to allocate a newly aligned physical memory buffer:")
        fixed = transposed.contiguous().view(12)
        print(f"  Fixed shape: {fixed.shape}, is_contiguous={fixed.is_contiguous()}")

    # D-Vision Transformation: HWC (OpenCV) -> BCHW (PyTorch Model Input)
    # OpenCV Frame: (Height=480, Width=640, Channels=3)
    cv_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # 1. Convert NumPy to PyTorch Tensor:
    torch_frame = torch.from_numpy(cv_frame)
    # 2. Permute dimensions: (H, W, C) -> (C, H, W)
    chw_frame = torch_frame.permute(2, 0, 1)
    # 3. Add batch dimension: (1, C, H, W) via unsqueeze(0)
    batch_frame = chw_frame.unsqueeze(0).float() / 255.0  # Normalize to [0.0, 1.0]

    print("\nDrone Frame Dimensional Transformation Pipeline:")
    print(f"  1. Raw OpenCV Frame:   {cv_frame.shape} (HWC, uint8)")
    print(f"  2. Permuted Tensor:    {chw_frame.shape} (CHW, uint8)")
    print(f"  3. PyTorch Model Input:{batch_frame.shape} (BCHW, float32, range [{batch_frame.min():.2f}, {batch_frame.max():.2f}])")

    # -------------------------------------------------------------------------
    section_header("4. BROADCASTING: YOLOV11 BOUNDING BOX GRID SHIFT")
    # In YOLO, predictions are made across an 80x80 spatial feature grid.
    # Feature map has shape: (Batch=1, Anchors=3, Height=80, Width=80, Coordinates=4)
    # We want to add grid offsets [grid_x, grid_y] to box centers [cx, cy].
    grid_y, grid_x = torch.meshgrid(torch.arange(80), torch.arange(80), indexing="ij")
    # Shape: (80, 80, 2) -> grid coordinates [x, y]
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # (80, 80, 2)
    # Expand dims to match YOLO head: (1, 1, 80, 80, 2)
    grid = grid.unsqueeze(0).unsqueeze(0)

    # Simulated raw network regression outputs: (Batch=1, Anchors=3, Height=80, Width=80, 2)
    pred_offsets = torch.randn(1, 3, 80, 80, 2)

    # Broadcasting: (1, 3, 80, 80, 2) + (1, 1, 80, 80, 2) -> (1, 3, 80, 80, 2)
    # Grid is virtually expanded across Anchors=3 without duplicating memory!
    absolute_box_centers = pred_offsets + grid

    print("YOLO Grid Broadcasting Mechanics:")
    print(f"  Network Raw Offsets:   {pred_offsets.shape}")
    print(f"  Spatial Anchor Grid:   {grid.shape}")
    print(f"  Broadcast Sum Result:  {absolute_box_centers.shape}")
    print(f"  Center at (col=10, row=20): {absolute_box_centers[0, 0, 20, 10].tolist()}")

    # -------------------------------------------------------------------------
    section_header("5. ZERO-COPY INTEROPERABILITY: NUMPY <-> PYTORCH")
    np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    pt_tensor = torch.from_numpy(np_arr)  # Shares underlying memory buffer!

    print(f"Initial NumPy:   {np_arr}")
    print(f"Initial PyTorch: {pt_tensor}")

    # Modifying PyTorch modifies NumPy in-place without memory allocation!
    pt_tensor[0] = 999.0
    print(f"After modifying PyTorch in-place:")
    print(f"  NumPy Array:   {np_arr}  <-- In-place updated!")
    print(f"  PyTorch Tensor:{pt_tensor}")
    assert np_arr[0] == 999.0, "Memory sharing validation failed!"

    # -------------------------------------------------------------------------
    section_header("6. MATRIX MULTIPLICATION & BATCH TENSOR MATH")
    # In attention heads and fully-connected layers:
    # A: (Batch=4, M=16, K=32), B: (Batch=4, K=32, N=64)
    # Result: (Batch=4, M=16, N=64) via torch.bmm (Batch Matrix Multiplication)
    A = torch.randn(4, 16, 32)
    B = torch.randn(4, 32, 64)
    C = torch.bmm(A, B)

    print(f"Batch Tensor A:       {A.shape}")
    print(f"Batch Tensor B:       {B.shape}")
    print(f"Batch Matmul Result:  {C.shape} (Computed in parallel across 4 batches)")

    # -------------------------------------------------------------------------
    section_header("7. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What is the difference between `tensor.view()` and `tensor.reshape()` in PyTorch?
A1: - `tensor.view()` creates a new view of the existing tensor sharing the exact same memory buffer.
      It strictly requires the underlying tensor to be contiguous in memory (i.e., tensor.is_contiguous() == True).
      If the tensor was transposed, sliced, or permuted, `.view()` will throw a RuntimeError.
    - `tensor.reshape()` is safer: it returns a contiguous view if possible (zero memory overhead),
      or automatically creates a cloned contiguous copy in memory if the tensor was non-contiguous.

Q2: Why do PyTorch Computer Vision models use BCHW format while TensorFlow/Keras defaults to BHWC?
A2: - PyTorch's BCHW layout (Batch, Channels, Height, Width) groups all pixels of a single color channel contiguously in RAM.
      This format is heavily optimized for cuDNN and Intel oneDNN 2D convolution kernels, which slide 2D filters across
      planar feature maps with maximal cache locality.
    - TensorFlow's default BHWC layout (Batch, Height, Width, Channels) groups RGB values of a single pixel contiguously.
      While intuitive for image loading, intermediate deep convolutional representations perform significantly faster
      in channel-first (BCHW / NCHW) layout.
""")

if __name__ == "__main__":
    main()
