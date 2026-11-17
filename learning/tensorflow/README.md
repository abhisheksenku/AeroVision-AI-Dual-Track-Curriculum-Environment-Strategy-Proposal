# AeroVision AI – Phase 3: TensorFlow & Keras (Isolated Track)

This directory (`learning/tensorflow/`) houses the complete 6-module curriculum for **Phase 3: TensorFlow and Keras**, running within the secondary isolated virtual environment (`.venv_tf`).

It is designed to give you complete dual-framework fluency (PyTorch + TensorFlow) and masters **TensorFlow Lite (TFLite) Edge Quantization** for ultra-low-power drone microcontrollers.

---

## Complete Module Index & Curriculum Mapping

| Script | Topic & Key Focus | Core Engineering & Architectural Concept | Status |
| :--- | :--- | :--- | :--- |
| `01_tensor_basics.py` | TF Tensors & AutoDiff | `tf.constant`, `tf.Variable`, tensor ranks 0-4, broadcasting bounding boxes, `tf.GradientTape` auto-differentiation. | ✅ Verified |
| `02_keras_neural_network.py` | Sequential vs Functional API | Directed Acyclic Graphs (DAGs), Multi-Input Multi-Output (MIMO) sensor fusion (flight dynamics + environmental streams). | ✅ Verified |
| `03_cnn_image_classification.py` | CNNs & BHWC Memory Ordering | `Conv2D`, `BatchNormalization`, `MaxPooling2D`, residual skip connections (`layers.add`), GlobalAveragePooling2D vs Flatten. | ✅ Verified |
| `04_model_training.py` | Model Training & Callbacks | `model.compile()`, `model.fit()`, `EarlyStopping(restore_best_weights=True)`, `ReduceLROnPlateau`, test accuracy 100%. | ✅ Verified |
| `05_transfer_learning.py` | Transfer Learning & Freezing | Headless MobileNetV2, `base_model.trainable = False`, the `training=False` BatchNorm trap, 2-stage fine-tuning. | ✅ Verified |
| `06_model_evaluation.py` | Edge Quantization & TFLite | Native Keras v3 `.keras`, FlatBuffer `.tflite` export, Post-Training Dynamic Range INT8 Quantization, and standalone `tf.lite.Interpreter`. | ✅ Verified |

---

## Activation & Execution Instructions

This module operates strictly within the dedicated `.venv_tf` environment:

```powershell
# 1. Activate the secondary isolated TensorFlow virtual environment
.\.venv_tf\Scripts\Activate.ps1

# 2. Run Module 01: TensorFlow Tensors & GradientTape
python learning/tensorflow/01_tensor_basics.py

# 3. Run Module 02: Keras Sequential vs Functional MIMO Architecture
python learning/tensorflow/02_keras_neural_network.py

# 4. Run Module 03: 2D Convolutions, ResNet Blocks & BHWC Memory Layout
python learning/tensorflow/03_cnn_image_classification.py

# 5. Run Module 04: Model Training, EarlyStopping & Learning Curves
python learning/tensorflow/04_model_training.py

# 6. Run Module 05: Transfer Learning & Layer Freezing
python learning/tensorflow/05_transfer_learning.py

# 7. Run Module 06: Model Persistence & TensorFlow Lite INT8 Quantization
python learning/tensorflow/06_model_evaluation.py
```

---

## Architectural Comparison: PyTorch vs TensorFlow / Keras

| Feature | PyTorch (AeroVision AI Platform) | TensorFlow / Keras (Isolated Track) |
| :--- | :--- | :--- |
| **Primary File Location** | Root & `backend/`, `learning/pytorch/` | `learning/tensorflow/` |
| **Virtual Environment** | `.\.venv` (Python 3.11, PyTorch 2.15 CPU, NumPy 2.4) | `.\.venv_tf` (Python 3.11, TF-CPU 2.16.1, NumPy 1.26) |
| **Execution Philosophy** | Dynamic graph (*define-by-run*) by default | Eager execution + `@tf.function` static graph tracing |
| **Autodiff Engine** | `loss.backward()` accumulates on `.grad` | `with tf.GradientTape() as tape:` records graph |
| **Image Tensor Ordering** | Channel-First: $(B, C, H, W)$ | Channel-Last: $(B, H, W, C)$ |
| **Edge Deployment Target** | ONNX Runtime, OpenVINO, TorchScript | **TensorFlow Lite (TFLite)** FlatBuffers, Coral Edge TPU |
| **Weight Quantization** | `torch.ao.quantization` (FX Graph / Eager) | `tf.lite.TFLiteConverter` Post-Training INT8 Quantization |
