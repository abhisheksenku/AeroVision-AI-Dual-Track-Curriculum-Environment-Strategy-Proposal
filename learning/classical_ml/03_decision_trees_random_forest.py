"""
=============================================================================
AeroVision AI - Phase 1: Machine Learning Foundations
Module 03: Decision Trees, Random Forests & Ensemble Learning
=============================================================================

Domain Problem:
Autonomous Drone Flight Mode & Structural Health Classification:
Predict the drone's operational state from high-frequency telemetry sensors:
- Class 0: 'OPTIMAL_CRUISE'       - Normal flight, balanced parameters.
- Class 1: 'WIND_COMPENSATION'    - High external gust, increased throttle demand.
- Class 2: 'ROTOR_THERMAL_STRESS' - High motor vibration & elevated temperature.
- Class 3: 'EMERGENCY_RTH'        - Critical voltage drop or sensor anomaly.

Telemetry Features:
- Feature 0: Airspeed (m/s)
- Feature 1: Motor Vibration RMS (g)
- Feature 2: Battery Cell Voltage (Volts, range 3.0V - 4.2V)
- Feature 3: ESC / Motor Temperature (°C)
- Feature 4: Wind Gust Speed (m/s)

Learning Objectives:
1. Understand non-linear feature partitioning via orthogonal hyperplanes.
2. Formulate Gini Impurity and Shannon Entropy / Information Gain mathematically.
3. Implement a Decision Tree Node and recursive Split Finder from scratch in pure NumPy.
4. Diagnose Tree Overfitting: Why unconstrained depth produces zero training error but high test variance.
5. Tree Pruning & Regularization: max_depth, min_samples_split.
6. The Ensemble Principle: Bootstrap Aggregation (Bagging) to reduce model variance.
7. Random Forest Mechanics: Feature bagging (subset of sqrt(D) features per split) to decorrelate trees.
8. Feature Importance: Calculate Mean Decrease in Impurity (MDI) from scratch across all forest nodes.
9. Out-of-Bag (OOB) Error: Unbiased generalization estimation without a separate validation split.
10. Evaluation Metrics: Confusion Matrix, Multi-Class Precision, Recall, and F1-Score from scratch.
11. Connection to AeroVision AI: Why deep neural networks replaced tree ensembles for raw image pixels,
    while tree ensembles still dominate tabular flight telemetry and mission safety audits.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

np.random.seed(42)

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC TELEMETRY DATASET GENERATION
# =============================================================================
def generate_drone_flight_dataset(n_samples=1000):
    """
    Generates realistic, physically grounded multi-sensor drone telemetry.
    Four operational classes generated via physical parameter distributions.
    """
    samples_per_class = n_samples // 4
    
    # Class 0: OPTIMAL_CRUISE (Airspeed 12-18, Vib 0.1-0.4, Volt 3.8-4.2, Temp 35-50, Gust 1-5)
    c0_speed = np.random.normal(15.0, 2.0, (samples_per_class, 1))
    c0_vib   = np.random.normal(0.25, 0.05, (samples_per_class, 1))
    c0_volt  = np.random.normal(4.0, 0.08, (samples_per_class, 1))
    c0_temp  = np.random.normal(42.0, 3.0, (samples_per_class, 1))
    c0_gust  = np.random.normal(3.0, 1.0, (samples_per_class, 1))
    y0 = np.zeros((samples_per_class, 1), dtype=int)
    
    # Class 1: WIND_COMPENSATION (Airspeed 18-26, Vib 0.3-0.7, Volt 3.6-3.9, Temp 50-65, Gust 12-22)
    c1_speed = np.random.normal(21.0, 2.5, (samples_per_class, 1))
    c1_vib   = np.random.normal(0.5, 0.1, (samples_per_class, 1))
    c1_volt  = np.random.normal(3.75, 0.08, (samples_per_class, 1))
    c1_temp  = np.normal if hasattr(np, 'normal') else np.random.normal(56.0, 4.0, (samples_per_class, 1))
    c1_gust  = np.random.normal(16.0, 2.5, (samples_per_class, 1))
    y1 = np.ones((samples_per_class, 1), dtype=int)
    
    # Class 2: ROTOR_THERMAL_STRESS (Airspeed 8-16, Vib 0.8-1.8, Volt 3.5-3.8, Temp 75-95, Gust 4-10)
    c2_speed = np.random.normal(12.0, 2.0, (samples_per_class, 1))
    c2_vib   = np.random.normal(1.2, 0.2, (samples_per_class, 1))
    c2_volt  = np.random.normal(3.65, 0.08, (samples_per_class, 1))
    c2_temp  = np.random.normal(82.0, 5.0, (samples_per_class, 1))
    c2_gust  = np.random.normal(6.0, 1.5, (samples_per_class, 1))
    y2 = np.full((samples_per_class, 1), 2, dtype=int)
    
    # Class 3: EMERGENCY_RTH (Airspeed 4-12, Vib 0.2-1.5, Volt 3.0-3.3, Temp 40-70, Gust 2-15)
    c3_speed = np.random.normal(8.0, 2.0, (samples_per_class, 1))
    c3_vib   = np.random.normal(0.6, 0.3, (samples_per_class, 1))
    c3_volt  = np.random.normal(3.18, 0.06, (samples_per_class, 1))
    c3_temp  = np.random.normal(52.0, 8.0, (samples_per_class, 1))
    c3_gust  = np.random.normal(7.0, 3.0, (samples_per_class, 1))
    y3 = np.full((samples_per_class, 1), 3, dtype=int)
    
    X0 = np.hstack([c0_speed, c0_vib, c0_volt, c0_temp, c0_gust])
    X1 = np.hstack([c1_speed, c1_vib, c1_volt, c1_temp, c1_gust])
    X2 = np.hstack([c2_speed, c2_vib, c2_volt, c2_temp, c2_gust])
    X3 = np.hstack([c3_speed, c3_vib, c3_volt, c3_temp, c3_gust])
    
    X = np.vstack([X0, X1, X2, X3])
    y = np.vstack([y0, y1, y2, y3]).flatten()
    
    feature_names = ["Airspeed_mps", "Vibration_RMS_g", "Cell_Voltage_V", "Motor_Temp_C", "Wind_Gust_mps"]
    class_names = ["OPTIMAL_CRUISE", "WIND_COMPENSATION", "ROTOR_THERMAL_STRESS", "EMERGENCY_RTH"]
    
    return X, y, feature_names, class_names

# =============================================================================
# 2. MATHEMATICAL FOUNDATION: IMPURITY METRICS FROM SCRATCH
# =============================================================================
def calculate_gini(y):
    """
    Computes Gini Impurity:
        Gini(D) = 1 - sum_{c=1}^C (p_c)^2
    Measures the probability that a randomly chosen element from dataset D
    would be incorrectly labeled if randomly assigned according to class distribution.
    - Pure node (all single class): Gini = 0.0
    - Equally mixed 4 classes: Gini = 1 - 4 * (0.25)^2 = 0.75
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def calculate_entropy(y):
    """
    Computes Shannon Entropy:
        H(D) = - sum_{c=1}^C p_c * log2(p_c)
    Measures the expected information content (uncertainty) in bits.
    - Pure node: H(D) = 0.0 bits
    - Equally mixed 4 classes: - 4 * (0.25 * log2(0.25)) = 2.0 bits
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_information_gain(y_parent, y_left, y_right, criterion="gini"):
    """
    Information Gain:
        IG = Impurity(Parent) - [ (|D_left| / |D|) * Impurity(Left) + (|D_right| / |D|) * Impurity(Right) ]
    Measures the reduction in uncertainty/impurity achieved by splitting on a feature threshold.
    """
    impurity_fn = calculate_gini if criterion == "gini" else calculate_entropy
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0.0
    
    parent_imp = impurity_fn(y_parent)
    weighted_children_imp = (n_left / n) * impurity_fn(y_left) + (n_right / n) * impurity_fn(y_right)
    return parent_imp - weighted_children_imp

# =============================================================================
# 3. DECISION TREE CLASSIFIER IMPLEMENTATION FROM SCRATCH
# =============================================================================
class TreeNode:
    """Represents a single internal split node or terminal leaf in a Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, impurity=0.0, n_samples=0, gain=0.0):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split: X[:, feature] <= threshold
        self.left = left                # Left child TreeNode
        self.right = right              # Right child TreeNode
        self.value = value              # Predicted class index if this node is a leaf
        self.impurity = impurity        # Impurity at this node
        self.n_samples = n_samples      # Number of samples reaching this node
        self.gain = gain                # Impurity reduction achieved by this split
        
    def is_leaf(self):
        return self.value is not None

class ScratchDecisionTree:
    """
    Binary Decision Tree Classifier from first principles using pure NumPy.
    Supports Gini and Entropy impurity, configurable max_depth, min_samples_split,
    and optional feature subsampling (max_features) for Random Forest integration.
    """
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini", max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features  # None -> evaluate all features; int -> subsample
        self.root = None
        self.feature_importances_raw = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        self.feature_importances_raw = np.zeros(n_features)
        self.root = self._build_tree(X, y, depth=0)
        return self
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        impurity_fn = calculate_gini if self.criterion == "gini" else calculate_entropy
        current_impurity = impurity_fn(y)
        
        # Stopping criteria: pure node, max depth reached, or insufficient samples
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_class(y)
            return TreeNode(value=leaf_value, impurity=current_impurity, n_samples=n_samples)
            
        # Feature subsampling (Feature Bagging for Random Forests)
        if self.max_features is not None and self.max_features < n_features:
            features_to_check = np.random.choice(n_features, size=self.max_features, replace=False)
        else:
            features_to_check = np.arange(n_features)
            
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        
        for feature_idx in features_to_check:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            # Subsample candidate split points for efficiency on continuous telemetry
            if len(thresholds) > 20:
                thresholds = np.percentile(thresholds, np.linspace(5, 95, 20))
                
            for thresh in thresholds:
                left_idx = X_column <= thresh
                right_idx = ~left_idx
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                    
                gain = calculate_information_gain(
                    y, y[left_idx], y[right_idx], criterion=self.criterion
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = thresh
                    
        # If no split produces positive gain, convert to leaf
        if best_gain <= 0.0 or best_feature is None:
            return TreeNode(value=self._most_common_class(y), impurity=current_impurity, n_samples=n_samples)
            
        # Accumulate feature importance contribution: N_node * Information_Gain
        self.feature_importances_raw[best_feature] += (n_samples * best_gain)
        
        # Recursively construct child subtrees
        left_mask = X[:, best_feature] <= best_threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            impurity=current_impurity,
            n_samples=n_samples,
            gain=best_gain
        )
        
    def _most_common_class(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# =============================================================================
# 4. RANDOM FOREST CLASSIFIER FROM SCRATCH (BAGGING + FEATURE BAGGING + OOB)
# =============================================================================
class ScratchRandomForest:
    """
    Random Forest Classifier from scratch in NumPy:
    1. Bootstrap Aggregation (Bagging): Trains B trees on B bootstrap sets (sampled with replacement).
    2. Feature Bagging: At each node, evaluates sqrt(D) features to decorrelate trees.
    3. Out-Of-Bag (OOB) Evaluation: Measures generalization on unsampled points (~36.8% per tree).
    4. Mean Decrease in Impurity (MDI): Aggregates feature importances across all ensemble trees.
    """
    def __init__(self, n_estimators=25, max_depth=5, min_samples_split=2, criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.trees = []
        self.in_bag_masks = []
        self.feature_importances_ = None
        self.oob_score_ = None
        
    def fit(self, X, y):
        self.trees = []
        self.in_bag_masks = []
        n_samples, n_features = X.shape
        max_features = int(np.ceil(np.sqrt(n_features)))  # Standard Random Forest feature subset size
        total_importances = np.zeros(n_features)
        
        # Array to store OOB predictions: oob_predictions[sample_idx, class_idx]
        n_classes = len(np.unique(y))
        oob_vote_counts = np.zeros((n_samples, n_classes))
        
        for _ in range(self.n_estimators):
            # 1. Bootstrap sample (with replacement)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            # 2. Build tree with feature bagging
            tree = ScratchDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Accumulate feature importances
            total_importances += tree.feature_importances_raw
            
            # Record OOB votes for samples not present in this bootstrap
            if np.any(oob_mask):
                oob_preds = tree.predict(X[oob_mask])
                for idx, pred in zip(np.where(oob_mask)[0], oob_preds):
                    oob_vote_counts[idx, pred] += 1
                    
        # Normalize feature importances across all trees
        sum_imp = np.sum(total_importances)
        self.feature_importances_ = total_importances / sum_imp if sum_imp > 0 else total_importances
        
        # Compute Out-of-Bag (OOB) Accuracy Score
        evaluated_oob = np.sum(oob_vote_counts, axis=1) > 0
        if np.any(evaluated_oob):
            oob_final_preds = np.argmax(oob_vote_counts[evaluated_oob], axis=1)
            self.oob_score_ = np.mean(oob_final_preds == y[evaluated_oob])
        else:
            self.oob_score_ = 0.0
            
        return self
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_preds[i] = vals[np.argmax(counts)]
        return final_preds

# =============================================================================
# 5. EXECUTION, EVALUATION & BENCHMARKING
# =============================================================================
def main():
    section_header("1. SYNTHETIC TELEMETRY & DATA HYGIENE")
    X, y, feature_names, class_names = generate_drone_flight_dataset(n_samples=1000)
    print(f"Generated {X.shape[0]} flight records with {X.shape[1]} sensor features:")
    for i, name in enumerate(feature_names):
        print(f"  Feature {i}: {name:<18} (Min: {X[:, i].min():.2f}, Max: {X[:, i].max():.2f})")
        
    # Split train and test (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    section_header("2. MATHEMATICAL VERIFICATION: IMPURITY CALCULATIONS")
    pure_sample = np.array([0, 0, 0, 0, 0])
    mixed_binary = np.array([0, 0, 1, 1])
    equal_quad = np.array([0, 1, 2, 3])
    
    print("Impurity Metrics on Control Datasets:")
    print(f"  Pure (all class 0):       Gini = {calculate_gini(pure_sample):.4f} | Entropy = {calculate_entropy(pure_sample):.4f} bits")
    print(f"  50/50 Binary Split:       Gini = {calculate_gini(mixed_binary):.4f} | Entropy = {calculate_entropy(mixed_binary):.4f} bits")
    print(f"  25/25/25/25 Quad Split:   Gini = {calculate_gini(equal_quad):.4f} | Entropy = {calculate_entropy(equal_quad):.4f} bits")
    print(f"  Full Training Target:     Gini = {calculate_gini(y_train):.4f} | Entropy = {calculate_entropy(y_train):.4f} bits")

    # -------------------------------------------------------------------------
    section_header("3. TRAINING SCRATCH DECISION TREE (DEPTH=4)")
    scratch_tree = ScratchDecisionTree(max_depth=4, min_samples_split=5, criterion="gini")
    scratch_tree.fit(X_train, y_train)
    y_pred_scratch = scratch_tree.predict(X_test)
    scratch_acc = accuracy_score(y_test, y_pred_scratch)
    print(f"Scratch Decision Tree Test Accuracy: {scratch_acc * 100:.2f}%")
    print(f"Top Root Split Feature: {feature_names[scratch_tree.root.feature]} at threshold <= {scratch_tree.root.threshold:.3f}")
    print(f"Root Impurity Reduction (Gain): {scratch_tree.root.gain:.4f}")

    # -------------------------------------------------------------------------
    section_header("4. TRAINING SCRATCH RANDOM FOREST (25 TREES, DEPTH=4)")
    scratch_rf = ScratchRandomForest(n_estimators=25, max_depth=4, criterion="gini")
    scratch_rf.fit(X_train, y_train)
    y_pred_scratch_rf = scratch_rf.predict(X_test)
    scratch_rf_acc = accuracy_score(y_test, y_pred_scratch_rf)
    print(f"Scratch Random Forest Test Accuracy: {scratch_rf_acc * 100:.2f}%")
    print(f"Out-of-Bag (OOB) Generalization Score: {scratch_rf.oob_score_ * 100:.2f}%")
    
    print("\nDetailed Scratch Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_scratch_rf, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_scratch_rf)
    print(f"{'':>25} " + " ".join([f"{name[:6]:>7}" for name in class_names]))
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:>7d}" for val in row])
        print(f"{class_names[i]:>25}: {row_str}")

    # -------------------------------------------------------------------------
    section_header("5. FEATURE IMPORTANCE ANALYSIS (MEAN DECREASE IN IMPURITY)")
    importances = scratch_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Relative Feature Importance Computed Across All Forest Trees:")
    for rank, idx in enumerate(sorted_idx, 1):
        bar = "█" * int(importances[idx] * 40)
        print(f"  Rank {rank}: {feature_names[idx]:<18} | MDI: {importances[idx]:.4f} | {bar}")

    # -------------------------------------------------------------------------
    section_header("6. BIAS-VARIANCE EXPERIMENT: TREE DEPTH VS OVERFITTING")
    print(f"{'Max Depth':<10} | {'Train Accuracy':<16} | {'Test Accuracy':<16} | {'Diagnosis'}")
    print("-" * 65)
    for depth in [1, 2, 4, 7, 12]:
        dt = ScratchDecisionTree(max_depth=depth, criterion="gini")
        dt.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, dt.predict(X_train))
        te_acc = accuracy_score(y_test, dt.predict(X_test))
        if tr_acc < 0.85:
            diag = "High Bias (Underfit)"
        elif tr_acc - te_acc > 0.05:
            diag = "High Variance (Overfit)"
        else:
            diag = "Optimal Generalization"
        print(f"{depth:<10} | {tr_acc * 100:>12.2f}%    | {te_acc * 100:>12.2f}%   | {diag}")

    # -------------------------------------------------------------------------
    section_header("7. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why does a single deep Decision Tree have high variance, and how does a Random Forest fix this mathematically?
A1: An unconstrained decision tree splits until every leaf is pure (Gini=0), memorizing noise and outlier points (High Variance, Zero Bias).
    Random Forest fixes this through Bootstrap Aggregation (Bagging) and Feature Bagging:
    Var(Ensemble) = rho * sigma^2 + ((1 - rho) / B) * sigma^2
    Where B is the number of trees and rho is the pairwise correlation between trees.
    Bagging averages B trees, driving the second term ((1-rho)/B) * sigma^2 towards 0 as B increases.
    Feature subsampling (evaluating only sqrt(D) candidate features at each split) forces trees to pick different split features,
    which drives down the tree correlation rho, dramatically shrinking the total ensemble variance without increasing bias.

Q2: Why do modern computer vision systems (like AeroVision AI) use CNNs / ViTs instead of Random Forests for raw pixel inputs?
A2: 1. Locality & Spatial Invariance: A 2D convolution preserves spatial neighborhoods (adjacent pixels form edges, edges form parts).
       A decision tree treats every pixel coordinate as an independent orthogonal feature, completely losing 2D spatial coherence.
    2. Representation Learning: Deep CNNs automatically learn hierarchical feature representations (low-level edges to high-level vehicles).
       Decision trees cannot construct new feature combinations; they can only draw axis-aligned splits on raw input values.
    3. Dimensionality & Curse of Dimensionality: A standard 640x640x3 drone frame contains 1,228,800 numbers. Fitting decision trees
       on a 1.2M-dimensional space causes severe sparsity and high variance, whereas CNN weight sharing (e.g., 3x3 kernels) requires
       exponentially fewer parameters and achieves translation equivariance.
""")

if __name__ == "__main__":
    main()
