"""
=============================================================================
AeroVision AI - Phase 1: Machine Learning Foundations
Module 01: Linear Regression & Gradient Descent from Scratch
=============================================================================

Domain Problem:
Predicting Drone Battery Consumption Rate (Watts) from Telemetry:
- Feature 1: Flight Speed (m/s)
- Feature 2: Payload Weight (kg)
- Feature 3: Headwind Velocity (m/s)

Learning Objectives:
1. Understand the Linear Hypothesis: y_hat = w^T * x + b
2. Derive the Mean Squared Error (MSE) Loss Function.
3. Derive the analytical Gradients: d(MSE)/dw and d(MSE)/db.
4. Implement Vectorized Batch Gradient Descent from scratch in pure NumPy.
5. Prevent Data Leakage using StandardScaler (fit on train, transform on test).
6. Compare custom NumPy implementation against Scikit-learn LinearRegression and Ridge (L2).
7. Evaluation metrics: Mean Squared Error (MSE), Root MSE (RMSE), and R^2 Score.
8. Connection to AeroVision AI: How Bounding Box Regression in YOLO is a generalized linear head.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Set seed for reproducible synthetic telemetry
np.random.seed(42)

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC DRONE TELEMETRY DATASET GENERATION
# =============================================================================
def generate_drone_telemetry(n_samples=500):
    """
    Simulates real drone flight records:
    - Speed: 5 to 25 m/s
    - Payload: 0.5 to 5.0 kg
    - Headwind: 0 to 15 m/s
    True underlying physics:
    Power (Watts) = 120 (base hover) + 8.5*Speed + 24.0*Payload + 6.2*Headwind + Gaussian noise
    """
    speed = np.random.uniform(5.0, 25.0, size=(n_samples, 1))
    payload = np.random.uniform(0.5, 5.0, size=(n_samples, 1))
    headwind = np.random.uniform(0.0, 15.0, size=(n_samples, 1))

    # True parameters: w1=8.5, w2=24.0, w3=6.2, b=120.0
    noise = np.random.normal(loc=0.0, scale=8.0, size=(n_samples, 1))
    power_watts = 120.0 + (8.5 * speed) + (24.0 * payload) + (6.2 * headwind) + noise

    X = np.hstack([speed, payload, headwind])
    y = power_watts
    feature_names = ["Speed_mps", "Payload_kg", "Headwind_mps"]
    return X, y, feature_names

# =============================================================================
# 2. LINEAR REGRESSION FROM SCRATCH (NUMPY)
# =============================================================================
class ScratchLinearRegression:
    """
    Vectorized Linear Regression using Batch Gradient Descent.

    Mathematical Formulation:
    - Hypothesis:
        Y_hat = X * W + b
        Where X is (N, D), W is (D, 1), b is scalar.
    - Loss (Mean Squared Error):
        L = (1 / N) * sum((Y_hat - Y)^2) = (1 / N) * ||X * W + b - Y||^2
    - Gradients:
        dL/dW = (2 / N) * X^T * (Y_hat - Y)    -> Shape: (D, 1)
        dL/db = (2 / N) * sum(Y_hat - Y)       -> Shape: Scalar
    - Update Rule:
        W = W - alpha * (dL/dW)
        b = b - alpha * (dL/db)
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights with small random Gaussian values, bias to 0.0
        self.W = np.zeros((n_features, 1))
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward Pass: Compute predictions
            y_pred = np.dot(X, self.W) + self.b  # Shape: (N, 1)

            # Compute Loss (MSE)
            error = y_pred - y
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

            # Backward Pass: Analytical Gradients
            grad_W = (2.0 / n_samples) * np.dot(X.T, error)  # Shape: (D, 1)
            grad_b = (2.0 / n_samples) * np.sum(error)       # Shape: Scalar

            # Parameter Optimization Step
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"  [Epoch {epoch+1:4d}/{self.epochs}] Training MSE: {mse:.4f}")

    def predict(self, X):
        return np.dot(X, self.W) + self.b

# =============================================================================
# 3. EXECUTION, COMPARISON & VERIFICATION
# =============================================================================
def main():
    section_header("1. DATASET & PREPROCESSING (AVOIDING DATA LEAKAGE)")
    X, y, feature_names = generate_drone_telemetry(n_samples=600)
    print(f"Generated {X.shape[0]} drone telemetry samples with 3 features: {feature_names}")

    # Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Standardize Features: Zero-mean, unit-variance
    # CRITICAL: Fit scaler strictly on X_train to prevent test data leakage!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features normalized: mean ~ 0.0, std ~ 1.0")

    section_header("2. TRAINING LINEAR REGRESSION FROM SCRATCH (NUMPY)")
    scratch_model = ScratchLinearRegression(learning_rate=0.05, epochs=1000)
    scratch_model.fit(X_train_scaled, y_train)

    # Evaluate on Unseen Test Data
    y_pred_scratch = scratch_model.predict(X_test_scaled)
    scratch_mse = mean_squared_error(y_test, y_pred_scratch)
    scratch_r2 = r2_score(y_test, y_pred_scratch)

    print("\nScratch Model Results on Test Set:")
    print(f"  Test MSE:  {scratch_mse:.4f} W^2")
    print(f"  Test RMSE: {np.sqrt(scratch_mse):.4f} Watts")
    print(f"  Test R^2:  {scratch_r2:.4f} (Explains {scratch_r2*100:.2f}% of variance)")
    print(f"  Learned Weights (Normalized Space): {scratch_model.W.flatten().round(4)}")
    print(f"  Learned Bias (Intercept): {scratch_model.b:.4f}")

    section_header("3. BENCHMARK COMPARISON AGAINST SCIKIT-LEARN")
    # A. Ordinary Least Squares (Normal Equation closed-form solution)
    sk_lr = LinearRegression()
    sk_lr.fit(X_train_scaled, y_train)
    y_pred_sk = sk_lr.predict(X_test_scaled)
    sk_mse = mean_squared_error(y_test, y_pred_sk)
    sk_r2 = r2_score(y_test, y_pred_sk)

    # B. Ridge Regression (L2 Regularization: penalizes large weights)
    sk_ridge = Ridge(alpha=1.0)
    sk_ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = sk_ridge.predict(X_test_scaled)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    ridge_r2 = r2_score(y_test, y_pred_ridge)

    print("Model Comparison Summary:")
    print(f"  1. Scratch NumPy GD   -> Test MSE: {scratch_mse:.4f} | R^2: {scratch_r2:.4f}")
    print(f"  2. Scikit-Learn OLS   -> Test MSE: {sk_mse:.4f} | R^2: {sk_r2:.4f}")
    print(f"  3. Scikit-Learn Ridge -> Test MSE: {ridge_mse:.4f} | R^2: {ridge_r2:.4f}")

    # Verify that scratch NumPy matches analytical closed-form within 0.01%
    assert np.isclose(scratch_mse, sk_mse, rtol=0.01), "Scratch model did not converge to OLS minimum!"
    print("\nVerification Passed: Scratch Gradient Descent matched Scikit-learn OLS!")

    section_header("4. CONNECTION TO AEROVISION AI & DEEP LEARNING")
    print("""
How this connects to YOLO11 & Drone Computer Vision:
1. An Artificial Neural Network (Phase 2) is a stack of linear regression units
   interspersed with non-linear activation functions.
2. The final bounding box head in YOLO11 is a linear regression head that takes
   deep feature vectors and predicts 4 continuous values: [dx, dy, dw, dh].
3. The gradient update rule derived here:
     W = W - alpha * (dL/dW)
   is the exact same foundation used by AdamW and SGD to train billions of
   parameters across convolutional vision backbones.
""")

    section_header("EXERCISE FOR YOU (CODE MODIFICATION)")
    print("""
Your Mission:
1. Open this file and add L1 (Lasso) Regularization to ScratchLinearRegression:
     Loss = MSE + lambda * sum(|W|)
     dL/dW = (2/N)*X^T*(Y_hat - Y) + lambda * sign(W)
2. Add a `lambda_l1` hyperparameter (e.g. 0.5) to `__init__`.
3. Re-train and observe how L1 drives smaller, irrelevant feature weights to exactly 0.0.
4. Explain why L1 regularization performs automatic feature selection.
""")

    section_header("AI ENGINEER INTERVIEW QUESTIONS")
    print("""
Q1: "What is the difference between Batch Gradient Descent, Stochastic Gradient Descent (SGD),
     and Mini-Batch Gradient Descent? Which one do we use in Deep Learning and why?"
Answer:
  - Batch GD uses all N samples per step: Smooth convergence, but memory-intensive and slow on big data.
  - Stochastic GD uses 1 sample per step: Fast and breaks local minima, but gradients are noisy.
  - Mini-Batch GD (e.g. batch size 16, 32, 64) is the industry standard: It balances SIMD/GPU vectorization
    throughput with gradient variance that helps escape saddle points.

Q2: "What is the Normal Equation closed-form solution W = (X^T * X)^(-1) * X^T * Y, and why do
     we use Gradient Descent instead of the Normal Equation in Deep Learning?"
Answer:
  - The Normal Equation computes W in one algebraic step. However, computing (X^T * X)^(-1) has
    matrix inversion computational complexity of O(D^3) where D is feature dimension.
  - In Deep Learning, D is millions of parameters, and models are non-linear (no closed-form exists).
    Gradient Descent scales efficiently to arbitrary parameters and non-linear computational graphs.
""")

if __name__ == "__main__":
    main()
