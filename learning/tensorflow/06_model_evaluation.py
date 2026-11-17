"""
=============================================================================
AeroVision AI - Phase 3: TensorFlow & Keras (Isolated Track)
Module 06: Model Persistence, SavedModel & TensorFlow Lite (TFLite) Edge Quantization
=============================================================================

Domain Problem:
Deploying a Micro-Vision Classifier to an Ultra-Low-Power Drone Microcontroller:
Commercial drones operate on strict energy budgets (e.g. <5 Watts for flight avionics).
A standard 32-bit floating point deep model consumes excessive battery and memory.
To run onboard inference at 60 FPS:
1. Export model to modern Keras `.keras` format.
2. Convert model to TensorFlow Lite (`.tflite`) FlatBuffers via Concrete Function graph traces.
3. Apply Post-Training Quantization (PTQ) to compress 32-bit weights to 8-bit integers (INT8),
   slashing model size with minimal loss in accuracy.
4. Run inference using the standalone `tf.lite.Interpreter`.

Learning Objectives:
1. Model Persistence: Native `.keras` format vs TensorFlow FlatBuffers.
2. The TensorFlow Lite Converter: `tf.lite.TFLiteConverter.from_concrete_functions()`.
3. Post-Training Quantization (PTQ): Dynamic Range INT8 Quantization.
4. Storage Footprint Audit: Comparing Keras `.keras` vs FP32 TFLite vs INT8 TFLite.
5. Edge Runtime Execution: Running inference via the TFLite Interpreter.
6. Bitwise Output Verification: Asserting quantized predictions match original Keras model.
7. Connection to AeroVision AI: Deploying lightweight vision models to drone edge micro-controllers.
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
# 1. TRAIN A LIGHTWEIGHT COMPACT DRONE MODEL FOR EXPORT
# =============================================================================
def build_and_train_compact_model():
    """Builds and fits a compact edge classifier on synthetic telemetry."""
    np.random.seed(42)
    X = np.random.uniform(-3.0, 3.0, size=(600, 8)).astype(np.float32)
    # Synthetic target: Class 0 (Optimal), Class 1 (Caution), Class 2 (Fault)
    y = np.argmax(np.hstack([
        X[:, 0:1] * 2.0 - X[:, 1:2],
        X[:, 2:3] * 1.5 + X[:, 3:4],
        -X[:, 4:5] * 2.0 + X[:, 5:6]
    ]), axis=1)

    model = keras.Sequential([
        layers.Input(shape=(8,), name="edge_telemetry_input"),
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dense(16, activation="relu", name="dense_2"),
        layers.Dense(3, activation="softmax", name="status_probabilities")
    ], name="Compact_Edge_Drone_Model")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, X[:5]

# =============================================================================
# 2. MODEL EXPORT & TFLITE QUANTIZATION ENGINE
# =============================================================================
def main():
    section_header("1. TRAINING COMPACT EDGE MODEL & INITIAL INFERENCE")
    model, sample_batch = build_and_train_compact_model()
    
    keras_preds = model.predict(sample_batch, verbose=0)
    print(f"Original Keras Model Parameters: {model.count_params():,}")
    print(f"Sample Batch Prediction Shape:  {keras_preds.shape}")
    print(f"Sample 0 Predictions:           {[round(p, 4) for p in keras_preds[0]]}")

    # -------------------------------------------------------------------------
    section_header("2. PERSISTENCE: KERAS V3 (.KERAS) FORMAT")
    export_dir = "models/export"
    os.makedirs(export_dir, exist_ok=True)
    
    keras_v3_path = os.path.join(export_dir, "compact_edge_drone_model.keras")
    model.save(keras_v3_path)
    keras_v3_size_kb = os.path.getsize(keras_v3_path) / 1024
    print(f"Saved Native Keras v3 Model to: {keras_v3_path} ({keras_v3_size_kb:.2f} KB)")

    # -------------------------------------------------------------------------
    section_header("3. TENSORFLOW LITE (TFLITE) CONVERSION & INT8 QUANTIZATION")
    # Trace concrete function for clean graph conversion from Keras 3
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape=[None, 8], dtype=tf.float32, name="edge_telemetry_input")
    )
    
    # --- A. Standard Float32 TFLite Conversion ---
    converter_fp32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_fp32_model = converter_fp32.convert()
    
    tflite_fp32_path = os.path.join(export_dir, "model_fp32.tflite")
    with open(tflite_fp32_path, "wb") as f:
        f.write(tflite_fp32_model)
    tflite_fp32_size_kb = os.path.getsize(tflite_fp32_path) / 1024

    # --- B. Post-Training Dynamic Range Quantization (INT8 Weights) ---
    converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8_model = converter_int8.convert()
    
    tflite_int8_path = os.path.join(export_dir, "model_quant_int8.tflite")
    with open(tflite_int8_path, "wb") as f:
        f.write(tflite_int8_model)
    tflite_int8_size_kb = os.path.getsize(tflite_int8_path) / 1024

    print(f"Float32 TFLite Model Saved:     {tflite_fp32_path} ({tflite_fp32_size_kb:.2f} KB)")
    print(f"Quantized INT8 TFLite Saved:    {tflite_int8_path} ({tflite_int8_size_kb:.2f} KB)")
    
    print(f"\n--> Edge Artifacts Generated in '{export_dir}/':")
    print(f"    - Native Keras 3: {keras_v3_size_kb:.2f} KB")
    print(f"    - TFLite FP32:   {tflite_fp32_size_kb:.2f} KB")
    print(f"    - TFLite INT8:   {tflite_int8_size_kb:.2f} KB")

    # -------------------------------------------------------------------------
    section_header("4. STANDALONE TFLITE INTERPRETER EDGE EXECUTION")
    # Load and execute the quantized model using strictly the TFLite runtime
    interpreter = tf.lite.Interpreter(model_path=tflite_int8_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite Input Tensor:  index={input_details[0]['index']}, shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"TFLite Output Tensor: index={output_details[0]['index']}, shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Run inference sample-by-sample (as on an embedded drone companion computer)
    tflite_preds = []
    for sample in sample_batch:
        single_input = np.expand_dims(sample, axis=0)
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        tflite_preds.append(output_data[0])

    tflite_preds = np.array(tflite_preds)

    print("\nComparing Original Keras FP32 vs Quantized TFLite INT8 Predictions:")
    print(f"{'Sample':<8} | {'Original Keras FP32 Probabilities':<38} | {'Quantized TFLite INT8 Probabilities'}")
    print("-" * 88)
    for i in range(len(sample_batch)):
        k_str = f"[{keras_preds[i,0]:.3f}, {keras_preds[i,1]:.3f}, {keras_preds[i,2]:.3f}]"
        t_str = f"[{tflite_preds[i,0]:.3f}, {tflite_preds[i,1]:.3f}, {tflite_preds[i,2]:.3f}]"
        print(f"Sample {i+1:<2} | {k_str:<38} | {t_str}")

    # Verify predictions are closely aligned (<0.05 maximum absolute difference)
    max_diff = np.max(np.abs(keras_preds - tflite_preds))
    print(f"\nMaximum Discrepancy between FP32 and INT8: {max_diff:.4f}")
    assert max_diff < 0.05, "Quantization error exceeds acceptable threshold!"
    print("--> Quantization Verification PASSED: INT8 FlatBuffer preserved accuracy within 0.05 tolerance.")

    # -------------------------------------------------------------------------
    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What is Post-Training Quantization (PTQ) in TensorFlow Lite, and how does INT8 dynamic range quantization work?
A1: Post-Training Quantization (PTQ) converts weights and/or activations from 32-bit floating point (`float32`, 4 bytes)
    to 8-bit integers (`int8`, 1 byte) after training is complete, without requiring retraining.
    - Mechanics of Dynamic Range Quantization:
      1. Storage: Weight tensors are converted from float32 to int8 using an affine mapping:
         q = round(r / Scale) + ZeroPoint.
         This immediately shrinks model file size by 75% on disk.
      2. Execution: During inference on CPU, weights are dynamically un-quantized to float32 at runtime,
         or executed directly using integer SIMD arithmetic instructions (e.g. ARM NEON or Intel AVX-VNNI),
         accelerating inference by 2x to 3x while fitting into fast L1/L2 processor caches.

Q2: What is the architectural difference between a standard TensorFlow `SavedModel` and a TensorFlow Lite (`.tflite`) file?
A2: - `SavedModel`: A hierarchical directory format containing `saved_model.pb` (a protocol buffer defining the complete
      computation graph) and a `variables/` directory containing weight checkpoints. It contains full training metadata,
      asset files, and signatures, and requires the complete ~500MB TensorFlow C++ runtime to execute.
    - `.tflite`: A single, self-contained binary file encoded using Google's **FlatBuffers** serialization library.
      Key differences:
      1. Zero Deserialization Overhead: FlatBuffers can be mapped directly into memory (`mmap`) and read in-place
         without unpacking or copying memory.
      2. Micro Runtime: Executes via `tflite_runtime`, a lightweight C++ engine requiring less than 1MB of memory,
         making it ideal for embedded drone hardware (STM32 microcontrollers, ESP32, and Raspberry Pi zero).
""")

if __name__ == "__main__":
    main()
