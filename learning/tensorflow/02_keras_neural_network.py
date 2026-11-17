"""
=============================================================================
AeroVision AI - Phase 3: TensorFlow & Keras (Isolated Track)
Module 02: Keras Sequential vs Functional API & Multi-Task Architectures
=============================================================================

Domain Problem:
Multi-Sensor Drone Telemetry Fusion:
Modern industrial drones ingest data from multiple physically disparate sources:
1. Stream A: Flight Dynamics (Speed, Altitude, Battery Voltage) -> Shape: (3,)
2. Stream B: Environmental Sensors (Wind Gust, Air Temperature, Barometric Delta) -> Shape: (3,)

We need a Multi-Input, Multi-Output (MIMO) neural network that:
- Ingests both sensor streams through dedicated preprocessing branches.
- Merges intermediate representations.
- Outputs simultaneous predictions:
  - Output 1: Flight Mission Risk (Classification: 3 classes - Safe, Caution, Abort).
  - Output 2: Predicted Power Consumption (Regression: Watts).

Learning Objectives:
1. Understand Keras 3 architecture and multi-backend philosophy.
2. The Sequential API: Rapid prototyping for simple linear layer stacks.
3. The Functional API: Expressing complex architectures as Directed Acyclic Graphs (DAGs).
4. Multi-Input & Multi-Output (MIMO) network construction.
5. Activation functions in Keras: ReLU, Sigmoid, Softmax, and Swish (SiLU).
6. Layer inspections: Model summaries, parameter counts, and layer connectivity.
7. Connection to AeroVision AI: Multi-task heads for drone analytics.
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
# 1. SEQUENTIAL API (LINEAR STACK)
# =============================================================================
def build_sequential_model():
    section_header("1. KERAS SEQUENTIAL API: SIMPLE LINEAR STACK")
    # Sequential is ideal for simple feedforward networks where each layer
    # receives exactly one input tensor and produces one output tensor.
    model = keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation="swish", name="dense_hidden_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dense(16, activation="swish", name="dense_hidden_2"),
        layers.Dense(3, activation="softmax", name="classification_output")
    ], name="Drone_Sequential_Classifier")
    
    print(f"Sequential Model Name: {model.name}")
    model.summary()
    return model

# =============================================================================
# 2. FUNCTIONAL API (DIRECTED ACYCLIC GRAPH - MULTI-INPUT / MULTI-OUTPUT)
# =============================================================================
def build_functional_mimo_model():
    section_header("2. KERAS FUNCTIONAL API: MULTI-INPUT / MULTI-OUTPUT (MIMO)")
    # The Functional API treats layers as callable functions: output = layer(input)
    # This enables non-linear topologies: skip connections, branching, and multi-sensor fusion.

    # Branch 1: Flight Dynamics (Speed, Altitude, Battery)
    input_dynamics = layers.Input(shape=(3,), name="input_flight_dynamics")
    x1 = layers.Dense(16, activation="swish", name="dyn_dense_1")(input_dynamics)
    x1 = layers.BatchNormalization(name="dyn_bn")(x1)

    # Branch 2: Environmental Sensors (Wind, Temp, Pressure)
    input_environment = layers.Input(shape=(3,), name="input_environment_sensors")
    x2 = layers.Dense(16, activation="swish", name="env_dense_1")(input_environment)
    x2 = layers.BatchNormalization(name="env_bn")(x2)

    # Sensor Fusion: Concatenate intermediate embeddings
    fused = layers.concatenate([x1, x2], name="sensor_fusion_concat")  # Shape: (Batch, 32)
    fused_dense = layers.Dense(32, activation="swish", name="shared_representation")(fused)

    # Head 1: Flight Risk Classification (Softmax over 3 classes: Safe, Caution, Abort)
    risk_output = layers.Dense(3, activation="softmax", name="output_risk_class")(fused_dense)

    # Head 2: Power Consumption Regression (Linear activation predicting continuous Watts)
    power_output = layers.Dense(1, activation="linear", name="output_power_watts")(fused_dense)

    # Assemble complete MIMO model
    mimo_model = keras.Model(
        inputs=[input_dynamics, input_environment],
        outputs=[risk_output, power_output],
        name="Drone_Sensor_Fusion_MIMO"
    )

    print(f"MIMO Model Name: {mimo_model.name}")
    mimo_model.summary()
    return mimo_model

# =============================================================================
# 3. FORWARD INFERENCE & TENSOR FLOW VERIFICATION
# =============================================================================
def main():
    # 1. Build and inspect Sequential model
    seq_model = build_sequential_model()
    
    # 2. Build and inspect Functional MIMO model
    mimo_model = build_functional_mimo_model()

    section_header("3. FORWARD PASS SIMULATION WITH MULTI-SENSOR BATCH")
    # Simulate a batch of 4 drone flight telemetry records
    np.random.seed(42)
    batch_dynamics = np.random.uniform(5.0, 50.0, size=(4, 3)).astype(np.float32)
    batch_environment = np.random.uniform(0.0, 20.0, size=(4, 3)).astype(np.float32)

    # Run forward inference
    risk_preds, power_preds = mimo_model([batch_dynamics, batch_environment])

    print(f"Batch Size Evaluated: 4 drone records")
    print(f"Risk Class Probabilities Shape: {risk_preds.shape}")
    print(f"Power Regression Shape:          {power_preds.shape}")

    class_names = ["SAFE", "CAUTION", "ABORT"]
    print("\nSample Multi-Task Predictions for Drone 0:")
    probs = risk_preds[0].numpy()
    top_class = np.argmax(probs)
    print(f"  Risk Class Probabilities: [Safe: {probs[0]:.3f}, Caution: {probs[1]:.3f}, Abort: {probs[2]:.3f}]")
    print(f"  Diagnosed Status:         {class_names[top_class]} (Confidence: {probs[top_class]*100:.1f}%)")
    print(f"  Predicted Power Demand:   {power_preds[0, 0].numpy():.2f} Watts")

    section_header("4. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: When MUST you use the Keras Functional API instead of the Sequential API?
A1: The Sequential API is strictly limited to single-input, single-output linear pipelines.
    You MUST use the Functional API whenever you encounter:
    1. Multi-Input Networks: e.g., combining camera video frames with drone GPS telemetry.
    2. Multi-Output Networks: e.g., predicting bounding box coordinates AND object class probabilities.
    3. Non-Linear Graph Topologies: Residual / Skip connections (ResNet: x + F(x)), Inception modules,
       or DenseNet feature reuse.
    4. Shared Layers: Using the same convolutional weight layer across multiple inputs (Siamese networks).

Q2: What is the purpose of `layers.BatchNormalization` in a Keras Dense or Conv2D network?
A2: Batch Normalization standardizes intermediate activations to zero mean and unit variance across the mini-batch,
    then applies learnable scaling (gamma) and shifting (beta) parameters.
    Benefits:
    1. Mitigates Internal Covariate Shift: Stabilizes hidden layer distribution as preceding weights update.
    2. Accelerates Convergence: Allows higher learning rates without vanishing or exploding gradients.
    3. Mild Regularization: The small stochastic noise introduced by mini-batch mean/variance calculation
       acts as a regularizer, often reducing the necessity of heavy Dropout.
""")

if __name__ == "__main__":
    main()
