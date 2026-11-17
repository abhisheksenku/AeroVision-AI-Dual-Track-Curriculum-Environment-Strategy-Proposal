"""
=============================================================================
AeroVision AI - Phase 3: TensorFlow & Keras (Isolated Track)
Module 04: Model Training, Callbacks & Learning Dynamics
=============================================================================

Domain Problem:
Autonomous Drone Battery Degradation & Thermal Event Predictor:
Training a model to predict drone cell thermal degradation states from telemetry:
- Class 0: 'NORMAL_CELL'
- Class 1: 'THERMAL_THROTTLED'
- Class 2: 'OVERHEATING_CRITICAL'

Learning Objectives:
1. `model.compile()`: Loss functions, optimizers (AdamW), and evaluation metrics.
2. `model.fit()`: Epochs, batch size, validation split, and verbose logging.
3. Keras Callbacks:
   - `EarlyStopping`: Halting training when validation loss stops improving and restoring best weights.
   - `ModelCheckpoint`: Persisting the best model weights to disk automatically.
   - `ReduceLROnPlateau`: Dynamically decaying learning rate upon encountering plateaus.
4. Parsing training telemetry from `history.history`: loss curves and validation trajectories.
5. Evaluating test generalization with `model.evaluate()`.
6. Connection to AeroVision AI: Automated edge model training workflows.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC TELEMETRY DATA GENERATION
# =============================================================================
def generate_battery_thermal_data(n_samples=1200):
    np.random.seed(42)
    samples_per_class = n_samples // 3

    # Class 0: NORMAL (Current: 10-25A, Temp: 30-45C, Volt: 3.8-4.2V, Imp: 10-18 mOhm)
    c0 = np.random.normal([18.0, 38.0, 4.0, 14.0], [3.0, 3.0, 0.08, 1.5], (samples_per_class, 4))
    y0 = np.zeros(samples_per_class, dtype=int)

    # Class 1: THERMAL_THROTTLED (Current: 30-55A, Temp: 55-70C, Volt: 3.5-3.8V, Imp: 20-30 mOhm)
    c1 = np.random.normal([42.0, 62.0, 3.65, 25.0], [4.0, 4.0, 0.08, 2.0], (samples_per_class, 4))
    y1 = np.ones(samples_per_class, dtype=int)

    # Class 2: OVERHEATING_CRITICAL (Current: 55-90A, Temp: 75-100C, Volt: 3.1-3.4V, Imp: 35-55 mOhm)
    c2 = np.random.normal([70.0, 85.0, 3.25, 45.0], [6.0, 5.0, 0.06, 4.0], (samples_per_class, 4))
    y2 = np.full(samples_per_class, 2, dtype=int)

    X = np.vstack([c0, c1, c2]).astype(np.float32)
    y = np.concatenate([y0, y1, y2])

    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

# =============================================================================
# 2. MODEL ARCHITECTURE & COMPILATION
# =============================================================================
def build_thermal_classifier():
    model = keras.Sequential([
        layers.Input(shape=(4,), name="battery_telemetry"),
        layers.BatchNormalization(name="input_norm"),
        layers.Dense(32, activation="swish", name="dense_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.Dense(16, activation="swish", name="dense_2"),
        layers.Dense(3, activation="softmax", name="thermal_status")
    ], name="Drone_Thermal_Classifier")

    # Compile with AdamW optimizer and Sparse Categorical Crossentropy
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# =============================================================================
# 3. TRAINING WITH ADVANCED CALLBACKS
# =============================================================================
def main():
    section_header("1. SYNTHETIC TELEMETRY & DATA PARTITIONING")
    X, y = generate_battery_thermal_data(n_samples=1200)
    
    # 70% Train, 15% Validation, 15% Test
    n_train = int(0.70 * len(X))
    n_val   = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val     = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test   = X[n_train+n_val:], y[n_train+n_val:]

    print(f"Total Telemetry Records: {len(X)}")
    print(f"Train Set: {len(X_train)} | Val Set: {len(X_val)} | Test Set: {len(X_test)}")

    # -------------------------------------------------------------------------
    section_header("2. MODEL COMPILATION & CALLBACK SETUP")
    model = build_thermal_classifier()
    model.summary()

    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_weights_path = os.path.join(checkpoint_dir, "best_thermal_model.weights.h5")

    # Callbacks configuration
    my_callbacks = [
        # Early Stopping: Halts when val_loss plateaus for 4 epochs
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Model Checkpoint: Saves weights only when val_loss achieves a new minimum
        callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ),
        # Reduce LR On Plateau: Cuts learning rate in half when progress stalls
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # -------------------------------------------------------------------------
    section_header("3. TRAINING EXECUTION (MODEL.FIT)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=my_callbacks,
        verbose=1
    )

    # -------------------------------------------------------------------------
    section_header("4. EVALUATION & CONVERGENCE PROFILE")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    print(f"\nTraining Telemetry History:")
    print(f"  Total Epochs Completed:    {len(history.history['loss'])}")
    print(f"  Initial Training Loss:     {history.history['loss'][0]:.4f}")
    print(f"  Final Training Loss:       {history.history['loss'][-1]:.4f}")
    print(f"  Best Validation Loss:      {min(history.history['val_loss']):.4f}")

    assert test_acc > 0.95, "Model failed to achieve required accuracy!"
    print("--> Training Verification PASSED: Model achieved >95% accuracy on unseen test data.")

    # -------------------------------------------------------------------------
    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why should `restore_best_weights=True` always be set when using Keras `EarlyStopping`?
A1: If `restore_best_weights=False` (the default in some versions), when training terminates due to patience expiration
    (e.g., patience=5), the model in memory retains the weights from the FINAL epoch (epoch N), which by definition
    is 5 epochs WORSE than the best observed validation loss.
    Setting `restore_best_weights=True` ensures that upon stopping, Keras automatically rewinds model weights
    back to the exact historical checkpoint that achieved the global minimum validation loss, preventing test degradation.

Q2: What is the operational difference between `CategoricalCrossentropy` and `SparseCategoricalCrossentropy` in Keras?
A2: Both compute the exact same mathematical loss (- sum y_c * log p_c), but differ in target tensor format:
    - `CategoricalCrossentropy`: Expects one-hot encoded targets: shape (Batch, Num_Classes), e.g. [0, 1, 0].
      Flaw: Highly memory inefficient for large class counts (e.g. 10,000 classes requires allocating 10,000 floats per sample).
    - `SparseCategoricalCrossentropy`: Expects integer class labels: shape (Batch,), e.g. class index 1.
      Benefit: Minimal memory footprint, zero one-hot encoding overhead, and faster batch collation.
""")

if __name__ == "__main__":
    main()
