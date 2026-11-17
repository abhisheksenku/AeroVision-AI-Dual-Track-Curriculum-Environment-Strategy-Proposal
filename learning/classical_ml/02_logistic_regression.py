"""
=============================================================================
AeroVision AI - Phase 1: Machine Learning Foundations
Module 02: Logistic Regression, Cross-Entropy & Decision Boundaries
=============================================================================

Domain Problem:
Drone Mission Flight Safety Classification:
Predict whether an autonomous inspection flight should PROCEED (Class 0)
or ABORT due to safety risk (Class 1) based on telemetry features:
- Feature 1: Wind Gust Speed (m/s)
- Feature 2: Motor Vibration RMS (g-force)
- Feature 3: Battery Temperature Differential (deg C)

Learning Objectives:
1. Understand why Linear Regression fails for probability estimation (unbounded outputs).
2. The Sigmoid Activation Function: Mapping real numbers (-inf, +inf) to (0.0, 1.0).
3. The Log-Odds (Logit) formulation: ln(p / (1-p)) = w^T * x + b.
4. Derive the Binary Cross-Entropy (BCE) Loss Function.
5. Why MSE is non-convex when paired with Sigmoid (local minima traps).
6. Implement Vectorized Logistic Regression from scratch in pure NumPy.
7. Decision Threshold Tuning: Trading off Precision vs Recall for safety-critical missions.
8. Evaluation Metrics: Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC.
9. Connection to AeroVision AI: How the YOLO11 detection head performs multi-label binary classification.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve
)

np.random.seed(42)

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC FLIGHT SAFETY DATASET GENERATION
# =============================================================================
def generate_drone_safety_data(n_samples=700):
    """
    Simulates real flight sensor conditions:
    - Gust Speed: 2 to 22 m/s
    - Vibration RMS: 0.1 to 1.8 g
    - Temp Delta: 0.5 to 12.0 deg C

    True risk probability:
    z = -4.5 + (0.35 * gust) + (2.8 * vibration) + (0.42 * temp_delta)
    p = 1 / (1 + exp(-z))
    y = 1 if p > 0.5 else 0 (with minor environmental stochasticity)
    """
    gust = np.random.uniform(2.0, 22.0, size=(n_samples, 1))
    vibration = np.random.uniform(0.1, 1.8, size=(n_samples, 1))
    temp_delta = np.random.uniform(0.5, 12.0, size=(n_samples, 1))

    z = -4.5 + (0.35 * gust) + (2.8 * vibration) + (0.42 * temp_delta)
    prob = 1.0 / (1.0 + np.exp(-z))

    # Binary label: 1 = Abort / High Risk, 0 = Safe Flight
    y = (prob > 0.50).astype(np.float64)

    # Inject realistic 3% sensor anomaly noise
    noise_idx = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    y[noise_idx] = 1.0 - y[noise_idx]

    X = np.hstack([gust, vibration, temp_delta])
    feature_names = ["GustSpeed_mps", "Vibration_g", "TempDelta_C"]
    return X, y, feature_names

# =============================================================================
# 2. LOGISTIC REGRESSION FROM SCRATCH (NUMPY)
# =============================================================================
class ScratchLogisticRegression:
    """
    Vectorized Logistic Regression using Gradient Descent on Binary Cross-Entropy.

    Mathematical Formulation:
    - Linear Logit:
        z = X * W + b                           (Shape: N x 1)
    - Sigmoid Probability Activation:
        p_hat = sigma(z) = 1 / (1 + exp(-z))    (Shape: N x 1)
    - Loss (Binary Cross-Entropy):
        L = - (1 / N) * sum( y * ln(p_hat) + (1 - y) * ln(1 - p_hat) )
    - Analytical Gradient:
        dL/dW = (1 / N) * X^T * (p_hat - y)     (Shape: D x 1)
        dL/db = (1 / N) * sum(p_hat - y)        (Shape: Scalar)
    - Gradient Descent Update:
        W = W - alpha * (dL/dW)
        b = b - alpha * (dL/db)
    """
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z):
        # Clip z to prevent numerical overflow in exp(-z)
        z_clipped = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward Pass: Compute probability predictions
            z = np.dot(X, self.W) + self.b
            p_hat = self._sigmoid(z)

            # Compute Binary Cross-Entropy Loss (with eps to prevent log(0))
            eps = 1e-15
            p_hat_clamped = np.clip(p_hat, eps, 1.0 - eps)
            bce_loss = -np.mean(y * np.log(p_hat_clamped) + (1.0 - y) * np.log(1.0 - p_hat_clamped))
            self.loss_history.append(bce_loss)

            # Backward Pass: Compute gradient
            error = p_hat - y
            grad_W = (1.0 / n_samples) * np.dot(X.T, error)
            grad_b = (1.0 / n_samples) * np.sum(error)

            # Optimization Step
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"  [Epoch {epoch+1:4d}/{self.epochs}] BCE Loss: {bce_loss:.5f}")

    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# =============================================================================
# 3. EXECUTION, EVALUATION & INTERVIEW QUESTIONS
# =============================================================================
def main():
    section_header("1. DATASET & HYGIENE (ZERO DATA LEAKAGE)")
    X, y, feature_names = generate_drone_safety_data(n_samples=700)
    print(f"Total flights: {X.shape[0]} | Features: {feature_names}")
    print(f"Class balance: Safe (0) = {np.sum(y == 0)} | Abort Risk (1) = {np.sum(y == 1)}")

    # Split: 80% Train, 20% Test (Stratified to maintain class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale strictly on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    section_header("2. TRAINING SCRATCH LOGISTIC REGRESSION (NUMPY)")
    model_scratch = ScratchLogisticRegression(learning_rate=0.2, epochs=1000)
    model_scratch.fit(X_train_scaled, y_train)

    y_pred_scratch = model_scratch.predict(X_test_scaled, threshold=0.5)
    y_prob_scratch = model_scratch.predict_proba(X_test_scaled)

    section_header("3. BENCHMARK COMPARISON AGAINST SCIKIT-LEARN")
    model_sk = LogisticRegression(C=1.0) # Inverse regularization strength
    model_sk.fit(X_train_scaled, y_train.ravel())
    y_pred_sk = model_sk.predict(X_test_scaled)
    y_prob_sk = model_sk.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    scratch_auc = roc_auc_score(y_test, y_prob_scratch)
    sk_auc = roc_auc_score(y_test, y_prob_sk)

    print("Model Performance on Test Set:")
    print(f"  Scratch NumPy Model -> ROC-AUC: {scratch_auc:.4f}")
    print(f"  Scikit-Learn Model  -> ROC-AUC: {sk_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred_scratch)
    print(f"\nConfusion Matrix (Scratch Model, threshold=0.50):")
    print(f"  TN={cm[0,0]:3d}  |  FP={cm[0,1]:3d} (False Alarms)")
    print(f"  FN={cm[1,0]:3d}  |  TP={cm[1,1]:3d} (Detected Risks)")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_scratch, target_names=["Safe (0)", "Abort (1)"]))

    section_header("4. DRONE SAFETY THRESHOLD TUNING (PRECISION VS RECALL)")
    print("""
In autonomous drone flight, what is worse?
- False Positive (FP): Grounding a safe drone (cost: delayed inspection).
- False Negative (FN): Letting an unsafe drone take off (cost: catastrophic crash / $50,000 drone lost).

Therefore, for drone safety, we MUST MAXIMIZE RECALL (minimize False Negatives).
We achieve this by LOWERING the decision threshold from 0.50 to 0.30:
""")
    for t in [0.70, 0.50, 0.30, 0.15]:
        y_p = model_scratch.predict(X_test_scaled, threshold=t)
        cm_t = confusion_matrix(y_test, y_p)
        rec = cm_t[1,1] / (cm_t[1,1] + cm_t[1,0])
        prec = cm_t[1,1] / (cm_t[1,1] + cm_t[0,1]) if (cm_t[1,1] + cm_t[0,1]) > 0 else 0
        print(f"  Threshold={t:.2f} -> Recall={rec*100:.1f}% (FN={cm_t[1,0]:2d}) | Precision={prec*100:.1f}% (FP={cm_t[0,1]:2d})")

    section_header("5. CONNECTION TO AEROVISION AI & YOLO11")
    print("""
How Logistic Regression powers Modern YOLO:
1. In multi-class object detection (YOLOv8 to YOLO11), the classification head
   does NOT use Softmax across all classes.
2. Instead, it uses independent LOGISTIC SIGMOID units for EVERY class:
     P(class_c | x) = sigmoid(w_c^T * f + b_c)
3. This is called Multi-Label Classification: an aerial object can be both
   'vehicle' and 'truck' without mutual exclusion.
4. When you set `conf_threshold=0.35` in AeroVision AI, you are literally tuning
   the Sigmoid decision threshold of this exact logistic equation!
""")

    section_header("EXERCISE FOR YOU (CODE MODIFICATION)")
    print("""
Your Mission:
1. In `ScratchLogisticRegression`, add L2 Regularization (Weight Decay):
     Loss = BCE + (lambda_l2 / 2) * sum(W^2)
     dL/dW = (1/N)*X^T*(p_hat - y) + lambda_l2 * W
2. Run training with `lambda_l2 = 0.1` and compare the magnitude of the learned
   weights ||W|| with and without regularization.
3. Explain why L2 weight decay prevents extreme saturated probabilities near 0.0 and 1.0.
""")

    section_header("AI ENGINEER INTERVIEW QUESTIONS")
    print("""
Q1: "Why do we use Binary Cross-Entropy loss instead of Mean Squared Error (MSE)
     when training Logistic Regression with a Sigmoid activation?"
Answer:
  - If you use MSE with Sigmoid: L = (1/N) * sum((sigmoid(z) - y)^2).
  - The derivative contains the term sigma'(z) = sigma(z) * (1 - sigma(z)).
  - When the model makes a confident wrong prediction (e.g. z = +10 when y = 0),
    sigma(10) ~ 1.0, so sigma'(10) ~ 0.0. The gradient vanishes! The model gets
    'stuck' and learns extremely slowly.
  - BCE cancels out the sigmoid derivative mathematically: dL/dz = (p_hat - y).
    The gradient is directly proportional to error, guaranteeing convex optimization.

Q2: "What is the difference between ROC-AUC and PR-AUC, and which is better for
     imbalanced drone safety datasets (e.g. 99% safe flights, 1% crashes)?"
Answer:
  - ROC-AUC plots True Positive Rate vs False Positive Rate. Because True Negatives (TN)
    is massive in imbalanced datasets, FPR stays tiny, producing an artificially high
    ROC-AUC (>0.98) even if the model predicts many false alarms.
  - PR-AUC (Precision-Recall AUC) completely excludes True Negatives from its formulation.
    It focuses strictly on the minority positive class, providing an honest evaluation of
    how well the model catches rare hazards without flooding operators with false alarms.
""")

if __name__ == "__main__":
    main()
