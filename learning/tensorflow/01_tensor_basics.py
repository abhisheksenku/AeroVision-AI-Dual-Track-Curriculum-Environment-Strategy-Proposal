"""
=============================================================================
AeroVision AI - TensorFlow Learning Track
Module 01: Tensor Basics, Operations, and Computational Mechanics
=============================================================================

Learning Objectives:
1. Understand Tensors as multi-dimensional geometric arrays (Scalars, Vectors, Matrices, N-D Tensors).
2. Master TensorFlow tensor creation: tf.constant, tf.Variable, dtypes, and shapes.
3. Understand NumPy interoperability (zero-copy memory sharing and .numpy()).
4. Master Tensor Arithmetic: Element-wise ops, Broadcasting rules, and Matrix Multiplication (tf.matmul).
5. Understand Reduction Operations (reduce_sum, reduce_mean) used in loss function implementations.
6. Introduction to Automatic Differentiation via tf.GradientTape (contrast with PyTorch .backward()).
7. Connect tensor representations to Drone Video frames: Shape [Batch, Height, Width, Channels].
"""

import os
import sys

# Suppress verbose TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np

def section_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    section_header("1. TENSOR CREATION & RANKS")
    # Theory: A tensor is a generalization of vectors and matrices to potentially higher dimensions.
    # Rank 0 = Scalar (Magnitude only, e.g. single loss value)
    # Rank 1 = Vector (1D array, e.g. class probabilities)
    # Rank 2 = Matrix (2D grid, e.g. single grayscale frame or weight matrix)
    # Rank 3 = 3D Tensor (e.g. RGB image [Height, Width, Channels])
    # Rank 4 = 4D Tensor (e.g. Drone Video Batch [Batch_Size, Height, Width, Channels])

    # Rank 0: Scalar
    drone_altitude = tf.constant(125.5, dtype=tf.float32)
    print(f"Rank 0 (Scalar) - Value: {drone_altitude.numpy():.1f}m | Shape: {drone_altitude.shape} | Rank: {tf.rank(drone_altitude)}")

    # Rank 1: Vector (e.g. bounding box: [x1, y1, x2, y2])
    bbox = tf.constant([120.0, 85.0, 310.0, 240.0], dtype=tf.float32)
    print(f"Rank 1 (Vector) - BBox: {bbox.numpy()} | Shape: {bbox.shape} | Rank: {tf.rank(bbox)}")

    # Rank 2: Matrix (e.g. 3x3 camera intrinsic calibration matrix K)
    K = tf.constant([
        [1000.0,    0.0, 960.0],
        [   0.0, 1000.0, 540.0],
        [   0.0,    0.0,   1.0]
    ], dtype=tf.float32)
    print(f"Rank 2 (Matrix) - Camera Intrinsics:\n{K.numpy()}\nShape: {K.shape}")

    # Rank 4: Drone Video Batch Representation [Batch=2, Height=4, Width=4, Channels=3]
    video_batch = tf.zeros(shape=(2, 4, 4, 3), dtype=tf.float32)
    print(f"Rank 4 (Video Batch) - Shape: {video_batch.shape} | Total Elements: {tf.size(video_batch).numpy()}")

    section_header("2. CONSTANTS VS VARIABLES")
    # tf.constant: Immutable tensor. Values cannot change after creation.
    # tf.Variable: Mutable state tensor. Used for trainable model weights and biases (W, b).
    w = tf.Variable(initial_value=tf.random.normal(shape=(3, 1), mean=0.0, stddev=0.1), name="weight")
    print("Trainable Weight Variable W (Initial):\n", w.numpy())

    # In-place parameter update (e.g. Gradient Descent step: W = W - lr * grad)
    learning_rate = 0.01
    gradient_w = tf.constant([[0.5], [-0.2], [0.1]], dtype=tf.float32)
    w.assign_sub(learning_rate * gradient_w) # assign_sub performs in-place subtraction: w = w - delta
    print("Trainable Weight Variable W (After 1 update step):\n", w.numpy())

    section_header("3. NUMPY INTEROPERABILITY")
    # TensorFlow tensors convert seamlessly to/from NumPy arrays
    np_array = np.array([10, 20, 30], dtype=np.float32)
    tf_from_np = tf.convert_to_tensor(np_array)
    print(f"Converted from NumPy: Type={type(tf_from_np).__name__}, Value={tf_from_np.numpy()}")

    section_header("4. BROADCASTING & ELEMENT-WISE ARITHMETIC")
    # Broadcasting Intuition:
    # When operating on two tensors of different shapes, TensorFlow automatically stretches
    # dimensions of size 1 to match the larger tensor without copying memory.
    # Rule: Two dimensions are compatible when:
    #   1. They are equal, OR
    #   2. One of them is 1.

    # Example: Normalize a batch of drone bounding box coordinates by image resolution
    # Bounding boxes: shape (3, 4) -> 3 boxes with [x1, y1, x2, y2]
    boxes = tf.constant([
        [100.0, 200.0, 300.0, 400.0],
        [ 50.0,  80.0, 150.0, 220.0],
        [400.0, 100.0, 600.0, 350.0]
    ], dtype=tf.float32)

    # Resolution vector: shape (1, 4) -> [Width, Height, Width, Height]
    img_dims = tf.constant([[1920.0, 1080.0, 1920.0, 1080.0]], dtype=tf.float32)

    # Broadcasted division: (3, 4) / (1, 4) -> (3, 4)
    normalized_boxes = boxes / img_dims
    print("Unnormalized Boxes (pixels):\n", boxes.numpy())
    print("\nNormalized Boxes [0.0 to 1.0] (via Broadcasting):\n", np.round(normalized_boxes.numpy(), 4))

    section_header("5. MATRIX MULTIPLICATION (tf.matmul) VS ELEMENT-WISE (tf.multiply)")
    # Mathematics:
    # Element-wise: C_ij = A_ij * B_ij (Shapes must match or broadcast)
    # Matmul: C_ij = sum_k (A_ik * B_kj) (Inner dimensions must match: (M, K) x (K, N) -> (M, N))

    A = tf.constant([[1.0, 2.0], [3.0, 4.0]]) # Shape (2, 2)
    B = tf.constant([[2.0, 0.0], [1.0, 3.0]]) # Shape (2, 2)

    elem_mult = tf.multiply(A, B) # Or simply A * B
    matrix_mult = tf.matmul(A, B)  # Or A @ B

    print("Matrix A:\n", A.numpy())
    print("Matrix B:\n", B.numpy())
    print("\nElement-wise Product (A * B):\n", elem_mult.numpy())
    print("\nMatrix Multiplication (A @ B):\n", matrix_mult.numpy())

    section_header("6. REDUCTION OPERATIONS: LOSS COMPUTATION")
    # Reductions collapse one or more dimensions (e.g. calculating batch Mean Squared Error)
    y_true = tf.constant([1.0, 0.0, 1.0, 1.0], dtype=tf.float32)
    y_pred = tf.constant([0.9, 0.2, 0.8, 0.95], dtype=tf.float32)

    # Step 1: Squared error per sample
    squared_errors = tf.square(y_true - y_pred)
    # Step 2: Mean over batch
    mse_loss = tf.reduce_mean(squared_errors)
    print(f"Ground Truth:      {y_true.numpy()}")
    print(f"Predictions:       {y_pred.numpy()}")
    print(f"Squared Errors:    {squared_errors.numpy()}")
    print(f"Mean Squared Error:{mse_loss.numpy():.6f}")

    section_header("7. AUTOMATIC DIFFERENTIATION: tf.GradientTape")
    # Theory: TensorFlow uses `tf.GradientTape` as a context manager that records
    # forward-pass operations onto a tape for reverse-mode automatic differentiation.
    # Contrast with PyTorch: PyTorch builds a graph on tensors with requires_grad=True,
    # then executes loss.backward(). TensorFlow explicitly queries the tape.

    # Example: Compute gradient of loss = w^2 + 3*w + 5 with respect to w at w = 2.0
    # Analytic derivative: d(loss)/dw = 2*w + 3 -> At w = 2.0, d(loss)/dw = 7.0
    w_param = tf.Variable(2.0)

    with tf.GradientTape() as tape:
        loss = w_param ** 2 + 3 * w_param + 5

    # Compute d(loss) / d(w_param)
    grad = tape.gradient(loss, w_param)
    print(f"Function: loss = w^2 + 3*w + 5 at w = {w_param.numpy()}")
    print(f"Computed Gradient d(loss)/dw via GradientTape: {grad.numpy():.1f} (Expected: 7.0)")

    section_header("EXERCISE FOR YOU")
    print("""
Your Mission:
1. Open this file and locate Section 7 (GradientTape).
2. Define a multi-variable function:
     loss = 2 * (x ** 2) + 3 * (y ** 2) - 4 * x * y
   where x and y are tf.Variable(3.0) and tf.Variable(2.0).
3. Compute the partial derivatives:
     d(loss)/dx  and  d(loss)/dy
4. Verify by hand:
     d(loss)/dx = 4*x - 4*y = 4(3) - 4(2) = 12 - 8 = 4.0
     d(loss)/dy = 6*y - 4*x = 6(2) - 4(3) = 12 - 12 = 0.0
5. Print both gradients to confirm your code matches the analytical result.
""")

if __name__ == "__main__":
    main()
