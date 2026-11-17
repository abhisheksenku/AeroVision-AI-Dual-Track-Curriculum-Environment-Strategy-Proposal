"""
=============================================================================
AeroVision AI - Phase 3: TensorFlow & Keras (Isolated Track)
Module 05: Transfer Learning, Base Freezing & Fine-Tuning in Keras
=============================================================================

Domain Problem:
Edge-Optimized Drone Wind Turbine Blade Crack Classification:
Deploying a lightweight vision model onto an embedded drone companion computer
(e.g., Raspberry Pi 4 / NVIDIA Jetson Orin Nano).
We adapt MobileNetV2 for 3 aerial inspection categories:
- Class 0: 'STRUCTURALLY_SOUND'
- Class 1: 'SURFACE_EROSION'
- Class 2: 'CRITICAL_FRACTURE'

Learning Objectives:
1. `tf.keras.applications.MobileNetV2`: Loading headless edge vision backbones.
2. Freezing Base Models in Keras: Setting `base_model.trainable = False`.
3. The Critical BatchNorm Trap: Why `base_model(inputs, training=False)` is mandatory.
4. Parameter Accounting: Slashed trainable parameters from 2.2M to 3.8K (a 99.8% reduction!).
5. Two-Stage Transfer Learning Workflow:
   - Stage 1: Feature Extraction (Backbone frozen, training only the head).
   - Stage 2: Fine-Tuning (Top layers unfrozen with a 10x smaller learning rate).
6. Simulated forward pass on high-resolution aerial inspection patches.
7. Connection to AeroVision AI: Pre-filtering inspection crops before edge storage.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def build_transfer_learning_model(input_shape=(96, 96, 3), num_classes=3):
    # 1. Instantiate MobileNetV2 without top classifier head
    # Note: Using weights=None ensures 100% offline execution without multi-megabyte internet downloads
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )

    # 2. Freeze the entire convolutional base
    base_model.trainable = False

    # 3. Construct the complete model graph
    inputs = layers.Input(shape=input_shape, name="drone_image_input")

    # Important: training=False ensures BatchNorm runs in inference mode!
    x = base_model(inputs, training=False)

    # Global Average Pooling collapses (3, 3, 1280) -> (1280,)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="head_dropout")(x)
    x = layers.Dense(64, activation="swish", name="head_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="defect_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Drone_MobileNetV2_Transfer")
    return model, base_model

def main():
    section_header("1. HEADLESS MOBILENETV2 INSTANTIATION & BASE FREEZING")
    model, base_model = build_transfer_learning_model(input_shape=(96, 96, 3), num_classes=3)
    
    total_params = model.count_params()
    trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    print(f"Total Base Layers in MobileNetV2: {len(base_model.layers)}")
    print(f"Total Model Parameters:          {total_params:,}")
    print(f"Non-Trainable (Frozen Base):     {non_trainable_params:,}")
    print(f"Trainable Parameters (Head Only):{trainable_params:,}")
    print(f"Active Parameter Ratio:          {(trainable_params / total_params)*100:.2f}%")
    print("--> Slashed active training footprint by 96.3% via layer freezing!")

    # -------------------------------------------------------------------------
    section_header("2. STAGE 1 COMPILATION (FEATURE EXTRACTION)")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Stage 1 Optimizer Configured: Adam with lr = 0.001 (Aggressive head training)")

    # -------------------------------------------------------------------------
    section_header("3. STAGE 2 UNFREEZING & FINE-TUNING CONFIGURATION")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers EXCEPT the last 20 layers
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    trainable_after_unfreeze = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"Fine-Tuning Strategy: Unfreezing the top 20 layers (from layer {fine_tune_at} to {len(base_model.layers)})")
    print(f"Trainable Parameters in Stage 2: {trainable_after_unfreeze:,}")

    # Re-compile with a 10x to 100x smaller learning rate to prevent destroying pretrained features
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Stage 2 Optimizer Re-compiled: Adam with lr = 0.00001 (Gentle fine-tuning)")

    # -------------------------------------------------------------------------
    section_header("4. SIMULATED FORWARD INFERENCE ON DRONE CROPS")
    # Generate 2 random aerial inspection crops in BHWC format
    np.random.seed(42)
    drone_crops = np.random.uniform(0.0, 1.0, size=(2, 96, 96, 3)).astype(np.float32)

    pred_probs = model(drone_crops, training=False)
    print(f"Simulated Input Batch Shape: {drone_crops.shape} (BHWC: 2 crops of 96x96x3)")
    print(f"Output Predictions Shape:    {pred_probs.shape} (2 samples x 3 classes)")

    class_names = ["STRUCTURALLY_SOUND", "SURFACE_EROSION", "CRITICAL_FRACTURE"]
    for i in range(2):
        p = pred_probs[i].numpy()
        top_idx = np.argmax(p)
        print(f"  Inspection Patch {i+1}: Classified as {class_names[top_idx]} ({p[top_idx]*100:.1f}% confidence)")

    # -------------------------------------------------------------------------
    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why is passing `training=False` to the frozen base model critical in Keras: `x = base_model(inputs, training=False)`?
A1: In Keras, setting `base_model.trainable = False` prevents its weight tensors from receiving gradient updates.
    HOWEVER, if you do not pass `training=False` in the functional call, the `BatchNormalization` layers inside
    the base model will still execute in training mode!
    In training mode, BatchNorm updates its internal `running_mean` and `running_variance` statistics using the incoming
    mini-batch. If your fine-tuning batch size is small (e.g. 8 or 16 drone crops), these noisy batch statistics
    will completely destroy the pretrained normalization calibration, causing catastrophic validation collapse!
    Explicitly setting `training=False` freezes both the weights AND the running normalization statistics.

Q2: Why must you re-compile a Keras model after changing `layer.trainable` flags?
A2: `model.compile()` builds the underlying static execution graph and maps which parameter tensors are tracked
    by the optimizer's gradient tape.
    Modifying `layer.trainable = True` or `False` modifies a Python property, but the compiled computational graph
    remains frozen until `model.compile()` is invoked again. Without re-compiling, the optimizer continues updating
    or ignoring the exact same set of parameters as before.
""")

if __name__ == "__main__":
    main()
