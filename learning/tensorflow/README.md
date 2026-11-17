# AeroVision AI – TensorFlow & Keras Learning Track

Welcome to the **TensorFlow & Keras Learning Track** of AeroVision AI.

While the core AeroVision AI platform is powered by PyTorch and Ultralytics YOLO11, this track provides comprehensive mastery over Google's **TensorFlow 2.16+ and Keras 3** framework, static graph optimization, and edge deployment via **TensorFlow Lite (TFLite)**.

---

## 1. Dual-Environment Management & Activation

To ensure zero dependency conflicts and protect our PyTorch/YOLO11 setup, this track runs in an isolated virtual environment (`.venv_tf`).

### Primary Environment (AeroVision Platform & PyTorch Track):
```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1

# Direct Python Execution:
.\.venv\Scripts\python.exe <script_path>
```
*Stack: Python 3.11.9, PyTorch 2.15 CPU, TorchVision, OpenCV 5.1, Ultralytics YOLO11, NumPy 2.4.6*

### Secondary Environment (TensorFlow & Keras Track):
```powershell
# In PowerShell:
.\.venv_tf\Scripts\Activate.ps1

# Direct Python Execution:
.\.venv_tf\Scripts\python.exe <script_path>
```
*Stack: Python 3.11.9, TensorFlow-CPU 2.16.1, Keras 3.15.1, NumPy 1.26.4*

---

## 2. In-Depth Comparison: PyTorch vs TensorFlow / Keras

| Feature | PyTorch (AeroVision Core) | TensorFlow / Keras (This Track) |
| :--- | :--- | :--- |
| **Execution Model** | **Dynamic Eager Execution** (*define-by-run*). Graphs are constructed dynamically per forward pass. | **Hybrid**: Eager by default + Static Graph compilation via `@tf.function` / XLA. |
| **Automatic Differentiation** | `loss.backward()` accumulates gradients directly into `tensor.grad`. | Context manager `with tf.GradientTape() as tape:` records operations explicitly. |
| **Model Definition API** | Subclassing `torch.nn.Module` with explicit `forward()` methods. | Multi-tier: High-level `keras.Sequential`, graph-like `keras.Functional`, or `keras.Model` subclassing. |
| **Data Ingestion Pipeline** | `torch.utils.data.Dataset` + `DataLoader` with Python workers. | `tf.data.Dataset` with C++ native streaming, prefetching (`AUTOTUNE`), and map parallelization. |
| **Edge & Embedded Deployment** | TorchScript, ONNX Runtime, OpenVINO, TensorRT. | **TensorFlow Lite (TFLite)**, TFLite Micro (MCU), Coral Edge TPU. |
| **Industry Standing** | Dominant in AI Research, Computer Vision (YOLO), Generative AI. | Dominant in Enterprise Production, Google Cloud Vertex AI, Embedded Edge systems. |

---

## 3. Curriculum Syllabus

| Script | Topic | Description | Status |
| :--- | :--- | :--- | :--- |
| **`01_tensor_basics.py`** | Tensor Operations & Math | Creation, indexing, slicing, broadcasting, linear algebra, NumPy interop | 🟢 Completed |
| **`02_keras_neural_network.py`** | Keras Model Architecture | Sequential vs Functional APIs, Dense layers, activations, loss functions | ⚪ Next |
| **`03_cnn_image_classification.py`** | CNNs & Feature Maps | `Conv2D`, `MaxPooling2D`, receptive fields, synthetic image classification | ⚪ Pending |
| **`04_model_training.py`** | Training Loops & Callbacks | `model.fit()`, custom training loops with `GradientTape`, early stopping | ⚪ Pending |
| **`05_transfer_learning.py`** | Transfer Learning & Fine-Tuning | Pretrained MobileNetV2, layer freezing, drone classification adaptation | ⚪ Pending |
| **`06_model_evaluation.py`** | Evaluation, SavedModel & TFLite | Metrics, confusion matrices, model export, INT8 post-training quantization | ⚪ Pending |

---

## 4. Key AI Engineer Interview Questions: PyTorch vs TensorFlow

### Question 1: "How does gradient computation differ between PyTorch and TensorFlow?"
> **Model Answer**:
> - In **PyTorch**, autograd is deeply tied to tensor objects. Setting `requires_grad=True` on a tensor tracks all operations applied to it. Calling `loss.backward()` traverses the backward computational graph, populating `.grad` attributes on leaf tensors.
> - In **TensorFlow**, automatic differentiation is handled by `tf.GradientTape`. Any operation performed within the `with tf.GradientTape() as tape:` block is recorded onto the tape. Gradients are then explicitly retrieved via `grads = tape.gradient(loss, model.trainable_variables)`. This makes gradient inspection and custom gradient modifications (such as gradient clipping or multi-loss weighting) explicit and modular.

### Question 2: "Why is TensorFlow Lite (TFLite) frequently preferred for micro-edge drone microcontrollers?"
> **Model Answer**:
> "TFLite is designed specifically for resource-constrained embedded targets. It features:
> 1. A tiny C++ runtime interpreter (<1 MB binary footprint compared to full PyTorch LibTorch at ~100MB+).
> 2. Direct hardware acceleration delegation (NNAPI on Android, GPU delegates, Hexagon DSPs, and Edge TPUs).
> 3. Native Post-Training Quantization (PTQ) converting FP32 weights to INT8 with minimal accuracy loss, reducing memory by 75% and enabling execution on low-power microcontrollers (e.g. flight controllers) without an operating system."
