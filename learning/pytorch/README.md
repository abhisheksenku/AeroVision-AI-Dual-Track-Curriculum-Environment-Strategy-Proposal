# AeroVision AI – Deep Learning with PyTorch Foundations

This directory (`learning/pytorch/`) contains the complete 10-module deep learning engineering curriculum for **Phase 2: Deep Learning with PyTorch**.

All scripts run strictly on CPU inside the primary virtual environment (`.venv`), optimized for Intel Core architectures with AVX2 and oneDNN instructions.

---

## Complete Module Index & Curriculum Mapping

| Module | Script | Topic & Key Focus | Core Architectural & Mathematical Concept | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PT-01** | `01_tensor_basics.py` | Tensor Semantics & Memory | Strides, memory layout, `.view()` vs `.reshape()`, `.permute()`, `.contiguous()`, zero-copy NumPy sharing, YOLO grid broadcasting. | ✅ Verified |
| **PT-02** | `02_autograd_graph.py` | Autograd & Computational Graphs | Dynamic DAG (*define-by-run*), `grad_fn`, leaf vs non-leaf, gradient accumulation, `torch.no_grad()`, `inference_mode()`, float64 finite-difference check ($< 10^{-7}$). | ✅ Verified |
| **PT-03** | `03_neural_network.py` | Neural Network Design | Subclassing `nn.Module`, mathematical proof of linear collapse, activation mechanics (ReLU to SiLU), Kaiming He initialization, decoupled drone detection head. | ✅ Verified |
| **PT-04** | `04_forward_backward_propagation.py` | Backpropagation from Scratch | 5-step training loop, analytical multivariate chain rule manual math matched to PyTorch Autograd down to machine epsilon ($0.0000000000$). | ✅ Verified |
| **PT-05** | `05_loss_and_optimizers.py` | Loss Functions & Optimizers | Smooth L1 (Huber) outlier clipping, `BCEWithLogitsLoss` log-sum-exp trick, Adam vs AdamW decoupled weight decay proof, Cosine Annealing scheduler. | ✅ Verified |
| **PT-06** | `06_cnn_architecture.py` | Convolutions & Receptive Fields | 2D convolution mechanics, output dimension formula $O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$, CBS block, receptive field propagation, multi-scale P3/P4/P5 pyramid. | ✅ Verified |
| **PT-07** | `07_image_classification.py` | Datasets, Loaders & Training Loop | Custom `Dataset` (`__len__`, `__getitem__`), `DataLoader` batching/shuffling, `train()` vs `eval()` modes, early stopping checkpoint save, converged to 100% accuracy. | ✅ Verified |
| **PT-08** | `08_transfer_learning.py` | Transfer Learning & Freezing | Freezing backbone layers (`param.requires_grad = False`), slashing active gradient computation by 96.4%, 2-tier differential learning rates (1e-5 vs 1e-3). | ✅ Verified |
| **PT-09** | `09_model_training_evaluation.py` | Evaluation & Model Persistence | Confusion Matrix, per-class Precision/Recall/F1 from scratch, safety-critical Recall prioritization, saving/loading `state_dict()` and comprehensive checkpoints. | ✅ Verified |
| **PT-10** | `10_yolo_architecture_deep_dive.py` | YOLO11 Neural Graph Deconstruction | Inspecting `models/weights/yolo11n.pt`, 24 sequential stages, SPPF context aggregation, anchor-free coordinates, and deconstructing `(1, 84, 8400)` output dimensions. | ✅ Verified |

---

## Execution Instructions

All 10 modules are self-contained and run directly within `.venv`:

```powershell
# 1. Activate primary virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run Module 01: Tensor Semantics & Memory Strides
python learning/pytorch/01_tensor_basics.py

# 3. Run Module 02: Autograd & Dynamic Computational Graphs
python learning/pytorch/02_autograd_graph.py

# 4. Run Module 03: Neural Network Architectures & Decoupled Heads
python learning/pytorch/03_neural_network.py

# 5. Run Module 04: Forward/Backward Propagation from Scratch
python learning/pytorch/04_forward_backward_propagation.py

# 6. Run Module 05: Loss Functions, Optimizers & AdamW
python learning/pytorch/05_loss_and_optimizers.py

# 7. Run Module 06: Convolutions & Receptive Field Propagation
python learning/pytorch/06_cnn_architecture.py

# 8. Run Module 07: Custom Datasets, DataLoaders & Full Training Loop
python learning/pytorch/07_image_classification.py

# 9. Run Module 08: Transfer Learning & Layer Freezing
python learning/pytorch/08_transfer_learning.py

# 10. Run Module 09: Multi-Class Evaluation & Model Checkpoints
python learning/pytorch/09_model_training_evaluation.py

# 11. Run Module 10: YOLO11 Neural Graph Deconstruction
python learning/pytorch/10_yolo_architecture_deep_dive.py
```

---

## Key Mathematical & Engineering Insights for AI Engineer Interviews

### 1. Why `(1, 84, 8400)` in YOLO11?
- `84`: 4 bounding box coordinates $[cx, cy, w, h] + 80$ COCO class logits.
- `8400`: The sum of spatial grid candidate anchors across 3 multi-scale pyramid levels:
  $$\text{P3 (Small Objects, Stride 8): } 80 \times 80 = 6,400$$
  $$\text{P4 (Medium Objects, Stride 16): } 40 \times 40 = 1,600$$
  $$\text{P5 (Large Objects, Stride 32): } 20 \times 20 = 400$$
  $$\text{Total: } 6,400 + 1,600 + 400 = 8,400 \text{ spatial candidates}$$

### 2. Why AdamW Over Standard Adam with L2 Regularization?
In standard Adam, the weight penalty $\lambda W$ was added directly to the gradient vector $g_t$, which meant it was subsequently divided by $\sqrt{v_t}$ (running variance of historical squared gradients). This caused weights with large historical gradients to experience suppressed weight decay, while weights with small gradients were over-regularized. AdamW decouples weight decay, applying $W \leftarrow W - \alpha \lambda W$ directly, preserving true $L_2$ regularization across all layers.

### 3. Why `model.train()` vs `model.eval()`?
`model.train()` enables Dropout and sets BatchNorm layers to update their moving average running statistics from the batch. `model.eval()` disables Dropout and freezes BatchNorm to use its fixed population statistics (`running_mean`, `running_var`), ensuring deterministic inference even when evaluating a single frame ($B=1$).
