"""
=============================================================================
AeroVision AI - Phase 3: TensorFlow & Keras (Isolated Track)
Module 03: Convolutional Neural Networks, ResNet Blocks & BHWC Memory Layout
=============================================================================

Domain Problem:
Aerial Drone Infrastructure Defect Classification:
Classifying 32x32 drone inspection image patches:
- Class 0: 'STRUCTURAL_CONCRETE'
- Class 1: 'CORROSION_METAL'
- Class 2: 'SOLAR_CELL'
- Class 3: 'VEGETATION_SOIL'

Learning Objectives:
1. Building 2D CNNs in Keras: `Conv2D`, `BatchNormalization`, `MaxPooling2D`.
2. The Critical Memory Layout Difference:
   - TensorFlow Default: Channel-Last  -> (Batch, Height, Width, Channels: BHWC)
   - PyTorch Default:    Channel-First -> (Batch, Channels, Height, Width: BCHW)
3. Implementing Residual / Skip Connections in Keras Functional API (`layers.add`).
4. Global Average Pooling 2D: Eliminating parameter explosion from dense flattening.
5. Forward inference and receptive field verification.
6. Connection to AeroVision AI: Edge CNN backbones on embedded drone compute.
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

# =============================================================================
# 1. RESIDUAL CNN BLOCK IN KERAS FUNCTIONAL API
# =============================================================================
def resnet_identity_block(input_tensor, filters, kernel_size=3):
    """
    Implements a classic ResNet residual block:
    F(x) + x -> Skip connection preserving gradient flow.
    """
    # Main path
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # Additive residual merge
    out = layers.add([x, shortcut])
    out = layers.Activation("swish")(out)
    return out

# =============================================================================
# 2. BUILDING THE RESIDUAL DRONE INSPECTION CNN
# =============================================================================
def build_drone_inspection_cnn(num_classes=4):
    # Input shape: (Height=32, Width=32, Channels=3) -> BHWC layout
    inputs = layers.Input(shape=(32, 32, 3), name="aerial_image_input")

    # Stem: 32x32x3 -> 32x32x16
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_act")(x)

    # Stage 1: Residual block at 32x32 resolution
    x = resnet_identity_block(x, filters=16)

    # Downsample Stage 2: 32x32 -> 16x16
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    x = resnet_identity_block(x, filters=32)

    # Downsample Stage 3: 16x16 -> 8x8
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    x = resnet_identity_block(x, filters=64)

    # Global Average Pooling: 8x8x64 -> (Batch, 64)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="classifier_dropout")(x)

    # Classifier head
    outputs = layers.Dense(num_classes, activation="softmax", name="defect_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Drone_Inspection_ResNet")
    return model

# =============================================================================
# 3. VERIFICATION & EXECUTION
# =============================================================================
def main():
    section_header("1. BHWC VS BCHW TENSOR LAYOUT COMPARISON")
    print("Tensor Dimension Ordering Comparison:")
    print("  Framework       | Format Name    | Dimension Sequence       | Example (Batch 1, 640x640 RGB)")
    print("  -----------------------------------------------------------------------------------------")
    print("  TensorFlow/Keras| Channels Last  | (Batch, Height, Width, C)| (1, 640, 640, 3)")
    print("  PyTorch         | Channels First | (Batch, C, Height, Width)| (1, 3, 640, 640)")
    print("\nWhy it matters: OpenCV frames are already in HWC format. In Keras, image tensors can be fed")
    print("directly without calling `.permute(2, 0, 1)`, simplifying preprocessing pipelines.")

    # -------------------------------------------------------------------------
    section_header("2. MODEL SUMMARY & RESNET ARCHITECTURE AUDIT")
    model = build_drone_inspection_cnn(num_classes=4)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal Network Parameters: {total_params:,} (~{total_params * 4 / 1024:.2f} KB FP32)")

    # -------------------------------------------------------------------------
    section_header("3. FORWARD INFERENCE ON SYNTHETIC DRONE CROPS")
    # Generate 3 random aerial crops in BHWC format
    np.random.seed(42)
    sample_crops = np.random.uniform(0.0, 1.0, size=(3, 32, 32, 3)).astype(np.float32)

    pred_probs = model(sample_crops)
    print(f"Input Tensor Shape:  {sample_crops.shape} (BHWC: Batch=3, Height=32, Width=32, Channels=3)")
    print(f"Output Tensor Shape: {pred_probs.shape} (Batch=3, Classes=4)")

    class_names = ["STRUCTURAL_CONCRETE", "CORROSION_METAL", "SOLAR_CELL", "VEGETATION_SOIL"]
    for i in range(3):
        probs = pred_probs[i].numpy()
        top_idx = np.argmax(probs)
        print(f"  Crop {i+1}: Diagnosed as {class_names[top_idx]:<22} (Confidence: {probs[top_idx]*100:.2f}%)")

    # -------------------------------------------------------------------------
    section_header("4. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why do Residual Connections (Skip Connections: y = F(x) + x) solve the Vanishing Gradient problem?
A1: In standard deep networks, gradients flow backward through repeated matrix multiplications via the chain rule:
    dL/dx_0 = dL/dx_L * (W_L * W_{L-1} * ... * W_1)
    If singular values of weight matrices are < 1.0, gradients shrink exponentially to zero as depth increases.
    In a ResNet block, the forward equation is: x_{l+1} = x_l + F(x_l, W_l)
    Taking the derivative with respect to x_l:
    d(x_{l+1}) / d(x_l) = 1 + d(F) / d(x_l)
    Because of the constant "+ 1" term, gradient signal can flow directly backward across the identity shortcut
    to all earlier layers without being multiplied by weight matrices, preventing vanishing gradients even at 100+ layers!

Q2: What is the difference between `layers.Flatten()` and `layers.GlobalAveragePooling2D()`?
A2: - `layers.Flatten()` reshapes a 3D feature map (e.g. 8x8x64) into a 1D vector of length 4,096.
      Connecting this to a dense layer of 64 units creates 4,096 * 64 = 262,144 weights.
      This causes parameter explosion, heavy memory consumption, and massive overfitting.
    - `layers.GlobalAveragePooling2D()` computes the spatial average across each 8x8 channel,
      reducing (8, 8, 64) down to (1, 1, 64) with ZERO trainable parameters.
      Connecting this to a 64-unit dense layer requires only 64 * 64 = 4,096 weights (a 98.4% parameter reduction!).
""")

if __name__ == "__main__":
    main()
