# Phase 4: Computer Vision Foundations

This module covers foundational computer vision concepts and engineering practices necessary for high-throughput drone video analytics.

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_image_basics_and_colorspaces.py` | Image Basics & Color Spaces | Pixel geometry, C-contiguous strides, BGR vs RGB history, Grayscale formula ($Y = 0.299R + 0.587G + 0.114B$), HSV color gating for aerial vegetation and asphalt. |
| `02_spatial_transformations_and_filters.py` | Spatial Transforms & Filters | Interpolation methods (`INTER_AREA` vs `INTER_LINEAR`), 2D Affine rotation matrices ($M = \begin{bmatrix} \alpha & \beta & (1-\alpha)c_x - \beta c_y \\ -\beta & \alpha & \beta c_x + (1-\alpha)c_y \end{bmatrix}$), 2D spatial convolutions, Gaussian blur, Sobel gradient magnitude, Canny edge detection. |
| `03_video_decoding_and_frame_extraction.py` | Video Decoding & Streaming | Container vs Codec, VideoCapture, FPS/FourCC metadata, uncompressed memory analysis (proof that 4K video exceeds 16GB RAM in 22 seconds), $O(1)$ memory generator stream pattern, temporal frame striding. |
| `04_aspect_ratio_letterboxing.py` | Aspect Ratio Letterboxing | Distortion problem of naive non-uniform stretching, scale factor $r = \min(w_{target}/w, h_{target}/h)$, symmetric border padding ($dw, dh$), exact inverse coordinate mapping (un-letterboxing) with zero roundtrip error. |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/computer_vision/01_image_basics_and_colorspaces.py
python learning/computer_vision/02_spatial_transformations_and_filters.py
python learning/computer_vision/03_video_decoding_and_frame_extraction.py
python learning/computer_vision/04_aspect_ratio_letterboxing.py
```

---

## Computer Vision Engineer Interview Questions

1. **Why does OpenCV read images in BGR format by default rather than standard RGB?**
   *Answer*: Historically, early camera sensors and hardware frame grabbers stored image byte streams in BGR (or BGRA) order in memory. When OpenCV was initially created at Intel in 1999, BGR was retained for zero-copy compatibility with Windows bitmap formats (DIBs). Modern deep learning libraries (PyTorch, TensorFlow) expect RGB, necessitating `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

2. **Why is `INTER_AREA` strictly required when downsampling aerial high-resolution imagery?**
   *Answer*: When an image is downscaled drastically (e.g. 4K $3840 \times 2160 \to 640 \times 640$), nearest-neighbor or bilinear interpolation samples isolated sparse pixel coordinates, skipping high-frequency spatial structures like thin power lines, guard rails, and road lane markings. This produces severe aliasing and Moire patterns. `INTER_AREA` computes a weighted average of all pixel values over the source pixel area, effectively acting as an anti-aliasing low-pass filter during downsampling.

3. **How does the generator pattern in Python solve the video memory hazard?**
   *Answer*: Loading an entire video file into memory reads every decoded frame into an uncompressed NumPy array. For 4K 30 FPS video, memory grows by ~746 MB every second. By wrapping `cv2.VideoCapture` in a Python generator (`yield frame`), only the current frame is held in RAM ($O(1)$ space complexity). After processing, the frame object is garbage collected or overwritten, enabling indefinite processing of multi-hour drone flights on modest 16 GB hardware.
