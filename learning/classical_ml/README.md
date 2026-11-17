# AeroVision AI – Classical Machine Learning Foundations

This directory (`learning/classical_ml/`) contains the complete mathematical and computational implementation of foundational Machine Learning algorithms, built from first principles in pure NumPy and verified against reference implementations.

Every module addresses a specific drone engineering problem in the **AeroVision AI** platform.

---

## Module Index & Curriculum Mapping

| Module | Script | Drone Domain Problem | Core Mathematical & Algorithmic Concepts | Status |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `01_linear_regression.py` | Predicting Drone Battery Power Consumption (Watts) | Vectorized Batch Gradient Descent, MSE loss, analytical gradients $\nabla_{\mathbf{w}}$, StandardScaler without data leakage, Scikit-learn OLS & Ridge ($L_2$). | ✅ Verified |
| **02** | `02_logistic_regression.py` | Flight Mission Abort Classification (Proceed vs Abort) | Sigmoid activation $\sigma(z)$, Log-Odds, Binary Cross-Entropy (BCE) loss, Decision Threshold Tuning (Precision vs Recall trade-off), ROC-AUC. | ✅ Verified |
| **03** | `03_decision_trees_random_forest.py` | Multi-Class Flight Mode & Thermal Health Diagnostics | Gini Impurity, Shannon Entropy, Information Gain, recursive tree building, Bootstrap Aggregation (Bagging), Feature Bagging ($\sqrt{D}$), Out-Of-Bag (OOB) error, Mean Decrease in Impurity (MDI). | ✅ Verified |
| **04** | `04_kmeans_clustering.py` | Spatial Waypoint Clustering & Vehicle Trajectory Flow | Unsupervised learning, Lloyd's algorithm, K-Means++ smart seeding, Within-Cluster Sum of Squares (Inertia), Elbow Method, Silhouette Analysis, historical YOLO anchor generation via IoU-clustering. | ✅ Verified |

---

## Mathematical Bridges: From Classical ML to Deep Learning & YOLO11

```
Classical ML Concept                   Deep Learning / Computer Vision Counterpart in AeroVision AI
--------------------                   ------------------------------------------------------------
w^T * x + b (Linear Hypothesis)   -->  Fully Connected Layer (nn.Linear) & 1x1 Convolution
MSE Loss                          -->  Bounding Box Regression Loss (Smooth L1 / Huber / CIoU)
Sigmoid Activation & BCE Loss     -->  Multi-label Classification Head in YOLO11
Decision Boundary Threshold (p)   -->  YOLO Confidence Threshold (conf_thresh = 0.25)
Information Gain / MDI            -->  Feature Attribution, Grad-CAM, Attention Maps in ViTs
K-Means Clustering                -->  Trajectory Lane Grouping (Module 5) & Pre-YOLOv8 Anchor Boxes
Bagging / Ensembles               -->  Test-Time Augmentation (TTA) & Multi-Model Ensembling (WBF)
```

---

## Execution Instructions

All scripts in this directory run within the primary CPU virtual environment (`.venv`):

```powershell
# 1. Activate the primary virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run Module 01: Linear Regression & Gradient Descent
python learning/classical_ml/01_linear_regression.py

# 3. Run Module 02: Logistic Regression & Decision Thresholds
python learning/classical_ml/02_logistic_regression.py

# 4. Run Module 03: Decision Trees & Random Forest Ensemble
python learning/classical_ml/03_decision_trees_random_forest.py

# 5. Run Module 04: K-Means Spatial Trajectory & Waypoint Clustering
python learning/classical_ml/04_kmeans_clustering.py
```

---

## Key Engineering Insights & Interview Cheat Sheet

### 1. Data Leakage Prevention
- **Rule**: Never call `fit_transform()` on the whole dataset before splitting.
- **Why**: Fitting a scaler on all data leaks the mean $\mu$ and standard deviation $\sigma$ of the test set into the training process, causing over-optimistic validation metrics that collapse in production.
- **Implementation**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled  = scaler.transform(X_test)  # Strictly using train mu and sigma!
  ```

### 2. Threshold Tuning in Safety-Critical Aerial Systems
- Default classification threshold is $0.5$.
- In drone safety (predicting `Flight Risk = 1` vs `Safe = 0`), a **False Negative** (predicting safe when the drone crashes) is catastrophic. A **False Positive** (aborting a flight that would have been safe) is only an operational delay.
- Lowering the decision threshold to $0.35$ increases **Recall** on the positive class, catching high-risk missions at the expense of minor precision reduction.

### 3. Tree Depth & Bias-Variance Decomposition
- **Unconstrained Tree**: High Variance, Zero Bias. Memorizes noise; achieves 100% training accuracy but drops significantly on test data.
- **Random Forest**: Reduces variance without increasing bias through two decorrelation mechanisms:
  1. *Row Bagging (Bootstrap)*: Averages predictions over $B$ trees, reducing variance by $\frac{1}{B}$.
  2. *Feature Bagging*: Evaluates only $\sqrt{D}$ features per split, preventing strong features from dominating every tree and lowering tree correlation $\rho$.

### 4. Convergence of Lloyd's K-Means Algorithm
- Monotonically decreases Within-Cluster Sum of Squares (WCSS).
- Expectation step minimizes distance given centroids; Maximization step minimizes distance given assignments.
- K-Means++ replaces uniform initialization with probabilistic sampling proportional to squared distance $D(x)^2$, guaranteeing an $\mathcal{O}(\log K)$ competitive bound against the optimal clustering.
