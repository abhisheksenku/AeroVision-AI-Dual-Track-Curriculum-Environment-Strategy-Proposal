# Phase 5: Object Detection with YOLO11

This module covers deep learning object detection algorithms, bounding box mathematics, Non-Maximum Suppression (NMS), YOLO11 inference, and aerial drone evaluation metrics (mAP@50 and mAP@50:95).

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_bounding_box_geometry_and_iou.py` | Bounding Box Geometry & IoU | Coordinate formats ($XYXY \leftrightarrow CXCYWH$), vectorized pairwise $N \times M$ IoU matrix broadcasting, Complete IoU ($L_{CIoU} = 1 - IoU + \frac{\rho^2}{c^2} + \alpha v$). |
| `02_nms_algorithm_from_scratch.py` | NMS Algorithm From Scratch | Dense candidate filtering, greedy score sorting, IoU thresholding, mathematical parity verification against `torchvision.ops.nms`, Class-Specific vs Class-Agnostic NMS. |
| `03_yolo11_inference_and_profiling.py` | YOLO11 Inference & Drone Optics | Ground Sampling Distance ($GSD = \frac{Altitude \times PixelPitch}{FocalLength}$), target pixel footprint analysis, nadir vs oblique angles, CPU latency and throughput profiling (<70 ms / ~15 FPS). |
| `04_evaluation_metrics_map.py` | Detection Metrics & mAP | True Positives ($IoU \ge 0.50$), Precision, Recall, PR Curve construction, 11-point interpolation, COCO continuous AUC, mAP@50 and strict mAP@[50:95] calculation from scratch. |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/object_detection/01_bounding_box_geometry_and_iou.py
python learning/object_detection/02_nms_algorithm_from_scratch.py
python learning/object_detection/03_yolo11_inference_and_profiling.py
python learning/object_detection/04_evaluation_metrics_map.py
```

---

## Object Detection Engineer Interview Questions

1. **Why is IoU alone insufficient as a loss function for bounding box regression?**
   *Answer*: If a predicted box and ground truth box do not intersect ($\text{IoU} = 0$), the gradient $\nabla \mathcal{L}_{IoU} = 0$. The model receives zero backpropagation signal regarding the direction or distance to the ground truth. Advanced loss formulations like Generalized IoU (GIoU), Distance IoU (DIoU), and Complete IoU (CIoU) incorporate Euclidean distance between centers ($\rho$), diagonal length of enclosing bounding box ($c$), and aspect ratio penalties ($\alpha v$), ensuring informative non-zero gradients even with zero spatial overlap.

2. **How does YOLO11 achieve anchor-free object detection?**
   *Answer*: Traditional YOLO versions (YOLOv3/v4/v7) relied on pre-defined anchor boxes of fixed aspect ratios clustered from dataset distributions. YOLO11 uses an anchor-free task-aligned decoupled head. It directly predicts distance offsets $(l, t, r, b)$ from feature map grid points to bounding box edges, combined with Distribution Focal Loss (DFL). This eliminates hyperparameter tuning of anchor dimensions and significantly improves small object recall in aerial scenes.

3. **What is the difference between mAP@50 and mAP@50:95?**
   *Answer*: mAP@50 evaluates detections at an IoU threshold of 0.50. A predicted box with 51% overlap is counted as an exact True Positive, meaning loose or sloppy boxes are not penalized. In contrast, mAP@50:95 (the primary COCO metric) averages mAP over 10 IoU thresholds from 0.50 to 0.95 with steps of 0.05. It heavily penalizes inaccurate borders, which is critical for aerial survey missions and infrastructure boundary inspection.
