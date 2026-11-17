"""
=============================================================================
AeroVision AI - Phase 1: Machine Learning Foundations
Module 04: K-Means Clustering & Unsupervised Spatial Intelligence
=============================================================================

Domain Problem:
1. Spatial Landing Zone & Anomaly Hotspot Discovery:
   An autonomous aerial survey of a renewable energy wind farm detects 600 surface
   defects across turbine towers and terrain. We need to identify 4 optimal
   autonomous drone recharging pad locations that minimize total flight distance.
2. Vehicle Trajectory Flow Clustering (AeroVision AI Module 5):
   Clustering 2D motion displacement vectors [delta_x, delta_y] extracted by
   ByteTrack to discover dominant traffic corridors (Northbound, Southbound, Turns).

Learning Objectives:
1. Understand the Unsupervised Learning paradigm (clustering without ground truth labels).
2. Formulate Within-Cluster Sum of Squares (WCSS / Inertia) mathematically.
3. Implement Lloyd's Algorithm from scratch using vectorized NumPy broadcasting:
   - Expectation step (nearest centroid assignment).
   - Maximization step (centroid recomputation).
   - Convergence checking (tolerance epsilon and max iterations).
4. Implement K-Means++ smart initialization from scratch to avoid local minima traps.
5. Determine optimal K using:
   - The Elbow Method (Inertia curve inflection).
   - The Silhouette Coefficient (cohesion vs separation).
6. Understand K-Means limitations: spherical assumptions, sensitivity to outliers, non-convex manifolds.
7. Benchmark custom ScratchKMeans against Scikit-learn KMeans.
8. Connection to AeroVision AI:
   - Trajectory summarization in spatial analytics.
   - Anchor Box Generation: Historically, YOLO (v2 through v7) used K-Means clustering on
     training bounding box dimensions (w, h) with an IoU distance metric to generate optimal anchor priors!
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC SPATIAL WAYPOINT & TRAJECTORY GENERATION
# =============================================================================
def generate_drone_survey_data(n_samples=600):
    """
    Simulates drone anomaly detections across 4 geographic inspection sectors
    in a 10 km x 10 km survey zone (coordinates in meters [x, y]).
    """
    # 4 distinct cluster centers:
    # Cluster 0: Sector Alpha (North-West)
    c0 = np.random.normal(loc=[2000, 7500], scale=[350, 400], size=(150, 2))
    # Cluster 1: Sector Beta  (North-East)
    c1 = np.random.normal(loc=[7800, 8200], scale=[450, 380], size=(150, 2))
    # Cluster 2: Sector Gamma (South-West)
    c2 = np.random.normal(loc=[2500, 2200], scale=[300, 320], size=(150, 2))
    # Cluster 3: Sector Delta (South-East)
    c3 = np.random.normal(loc=[8200, 2800], scale=[400, 420], size=(150, 2))
    
    X = np.vstack([c0, c1, c2, c3])
    # Ground truth labels used strictly for verification, never shown to unsupervised model
    y_true = np.hstack([np.zeros(150), np.ones(150), np.full(150, 2), np.full(150, 3)]).astype(int)
    
    # Shuffle dataset
    indices = np.random.permutation(n_samples)
    return X[indices], y_true[indices]

def generate_traffic_trajectories(n_samples=400):
    """
    Simulates vehicle displacement vectors [dx, dy] extracted across 30 frames:
    - Flow 0: Northbound Highway (+0 px/s, +65 px/s)
    - Flow 1: Southbound Highway (+0 px/s, -60 px/s)
    - Flow 2: Eastbound Cross-Street (+45 px/s, +0 px/s)
    - Flow 3: North-to-East Turn (+30 px/s, +35 px/s)
    """
    f0 = np.random.normal(loc=[0.0, 65.0], scale=[4.0, 6.0], size=(100, 2))
    f1 = np.random.normal(loc=[0.0, -60.0], scale=[4.0, 5.0], size=(100, 2))
    f2 = np.random.normal(loc=[45.0, 0.0], scale=[5.0, 3.0], size=(100, 2))
    f3 = np.random.normal(loc=[30.0, 35.0], scale=[6.0, 6.0], size=(100, 2))
    
    X_traj = np.vstack([f0, f1, f2, f3])
    indices = np.random.permutation(n_samples)
    return X_traj[indices]

# =============================================================================
# 2. VECTORIZED K-MEANS FROM FIRST PRINCIPLES (NUMPY)
# =============================================================================
class ScratchKMeans:
    """
    Vectorized K-Means implementation using Lloyd's Algorithm.
    
    Mathematical Formulation:
    - Objective: Minimize Within-Cluster Sum of Squares (Inertia):
        J = sum_{k=1}^K sum_{i in S_k} || x_i - mu_k ||^2
    - Expectation (Assignment):
        c_i = argmin_k || x_i - mu_k ||^2
    - Maximization (Centroid Update):
        mu_k = (1 / |S_k|) * sum_{i in S_k} x_i
    - Distance Calculation via Vectorized Broadcasting:
        ||X - mu||^2 has shape (N, K) computed without Python loops over samples!
    """
    def __init__(self, n_clusters=4, max_iters=300, tol=1e-4, init="k-means++", random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _init_centroids(self, X):
        """Initializes K centroids using either standard random choice or K-Means++."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        if self.init == "random":
            rand_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[rand_idx].copy()
            
        elif self.init == "k-means++":
            # 1. Pick first centroid uniformly at random
            centroids = [X[rng.choice(n_samples)]]
            
            # 2. Pick remaining K-1 centroids proportional to D(x)^2 (squared distance to closest centroid)
            for _ in range(1, self.n_clusters):
                current_centroids = np.array(centroids)
                # Pairwise squared distances to existing centroids: shape (N, len(centroids))
                # Broadcasting: (N, 1, D) - (1, K_curr, D) -> (N, K_curr)
                dists_sq = np.sum((X[:, np.newaxis, :] - current_centroids[np.newaxis, :, :]) ** 2, axis=2)
                # Distance to closest existing centroid for each sample
                min_dists_sq = np.min(dists_sq, axis=1)
                
                # Probability distribution proportional to D(x)^2
                probabilities = min_dists_sq / np.sum(min_dists_sq)
                next_centroid_idx = rng.choice(n_samples, p=probabilities)
                centroids.append(X[next_centroid_idx])
                
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def fit(self, X):
        """Runs Lloyd's iterative alternating optimization algorithm."""
        n_samples, _ = X.shape
        self.cluster_centers_ = self._init_centroids(X)
        
        for iteration in range(1, self.max_iters + 1):
            # -------------------------------------------------------------
            # STEP 1: EXPECTATION (Assign samples to nearest centroid)
            # Broadcasting: X[:, np.newaxis, :] is (N, 1, D)
            #               centers[np.newaxis, :, :] is (1, K, D)
            #               diff is (N, K, D)
            # -------------------------------------------------------------
            diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
            sq_distances = np.sum(diff ** 2, axis=2)  # Shape: (N, K)
            self.labels_ = np.argmin(sq_distances, axis=1)  # Shape: (N,)
            
            # -------------------------------------------------------------
            # STEP 2: MAXIMIZATION (Recompute centroid positions)
            # -------------------------------------------------------------
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                members = X[self.labels_ == k]
                if len(members) > 0:
                    new_centers[k] = np.mean(members, axis=0)
                else:
                    # Handle empty cluster: re-initialize to a random point
                    new_centers[k] = X[np.random.choice(n_samples)]
                    
            # -------------------------------------------------------------
            # STEP 3: CONVERGENCE CHECK (Shift in centroid positions)
            # -------------------------------------------------------------
            centroid_shift = np.linalg.norm(new_centers - self.cluster_centers_)
            self.cluster_centers_ = new_centers
            self.n_iter_ = iteration
            
            if centroid_shift < self.tol:
                break
                
        # Compute final Within-Cluster Sum of Squares (Inertia)
        final_diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        final_sq_dists = np.sum(final_diff ** 2, axis=2)
        # For each point, take squared distance to its assigned centroid
        min_sq_dists = np.min(final_sq_dists, axis=1)
        self.inertia_ = float(np.sum(min_sq_dists))
        
        return self

    def predict(self, X):
        """Predicts the closest cluster index for new points."""
        diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        sq_distances = np.sum(diff ** 2, axis=2)
        return np.argmin(sq_distances, axis=1)

# =============================================================================
# 3. EXECUTION, EVALUATION & BENCHMARKING
# =============================================================================
def main():
    section_header("1. SPATIAL ANOMALY DATASET: SURVEY SECTORS")
    X_survey, y_true = generate_drone_survey_data(n_samples=600)
    print(f"Generated {len(X_survey)} anomaly coordinates across 10km x 10km survey territory.")
    print(f"X coordinate range: [{X_survey[:, 0].min():.1f}m, {X_survey[:, 0].max():.1f}m]")
    print(f"Y coordinate range: [{X_survey[:, 1].min():.1f}m, {X_survey[:, 1].max():.1f}m]")

    # -------------------------------------------------------------------------
    section_header("2. FITTING SCRATCH K-MEANS WITH K-MEANS++ INITIALIZATION")
    scratch_km = ScratchKMeans(n_clusters=4, init="k-means++", random_state=42)
    scratch_km.fit(X_survey)
    
    print(f"Algorithm converged in {scratch_km.n_iter_} iterations.")
    print(f"Scratch Model Inertia (WCSS): {scratch_km.inertia_:,.2f} m^2")
    print("\nDiscovered Optimal Drone Recharging Hub Coordinates (Centroids):")
    for k, center in enumerate(scratch_km.cluster_centers_):
        cluster_count = np.sum(scratch_km.labels_ == k)
        print(f"  Hub {k+1}: X = {center[0]:>7.1f}m, Y = {center[1]:>7.1f}m | Anomaly Load: {cluster_count:>3d} sites")

    # -------------------------------------------------------------------------
    section_header("3. BENCHMARK AGAINST SCIKIT-LEARN REFERENCE IMPLEMENTATION")
    sk_km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    sk_km.fit(X_survey)
    
    print(f"Scikit-Learn Model Inertia:   {sk_km.inertia_:,.2f} m^2")
    inertia_discrepancy_pct = abs(scratch_km.inertia_ - sk_km.inertia_) / sk_km.inertia_ * 100
    print(f"Discrepancy vs Scikit-Learn:  {inertia_discrepancy_pct:.4f}% (Effectively Identical)")
    
    sil_scratch = silhouette_score(X_survey, scratch_km.labels_)
    sil_sk = silhouette_score(X_survey, sk_km.labels_)
    print(f"Scratch Silhouette Score:     {sil_scratch:.4f}")
    print(f"Scikit-Learn Silhouette Score: {sil_sk:.4f}")

    # -------------------------------------------------------------------------
    section_header("4. THE ELBOW METHOD & SILHOUETTE ANALYSIS (FINDING OPTIMAL K)")
    print(f"{'K (Clusters)':<14} | {'Inertia (WCSS)':<20} | {'Silhouette Score':<18} | {'Delta Inertia'}")
    print("-" * 72)
    
    inertias = []
    silhouettes = []
    k_range = list(range(2, 8))
    
    prev_inertia = None
    for k in k_range:
        km = ScratchKMeans(n_clusters=k, init="k-means++", random_state=42)
        km.fit(X_survey)
        sil = silhouette_score(X_survey, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        
        delta_str = f"-{(prev_inertia - km.inertia_):,.0f}" if prev_inertia is not None else "Baseline"
        prev_inertia = km.inertia_
        
        print(f"K = {k:<10} | {km.inertia_:>18,.1f} | {sil:>16.4f} | {delta_str}")
        
    optimal_k_by_sil = k_range[np.argmax(silhouettes)]
    print(f"\n--> Optimal K suggested by Peak Silhouette Score: K = {optimal_k_by_sil} (Score: {max(silhouettes):.4f})")

    # -------------------------------------------------------------------------
    section_header("5. VEHICLE TRAJECTORY FLOW CLUSTERING (AEROVISION AI MODULE 5)")
    X_traj = generate_traffic_trajectories(n_samples=400)
    print(f"Simulating ByteTrack velocity vectors [dx, dy] across 400 tracked vehicles...")
    
    traj_km = ScratchKMeans(n_clusters=4, init="k-means++", random_state=42)
    traj_km.fit(X_traj)
    
    print("\nDiscovered Dominant Traffic Flow Vectors [dx, dy] (px/frame):")
    flow_labels = ["Northbound Fast Flow", "Southbound Corridor", "Eastbound Cross Flow", "North-East Turn Channel"]
    for i, c in enumerate(traj_km.cluster_centers_):
        speed_mag = np.linalg.norm(c)
        angle_deg = np.degrees(np.arctan2(c[1], c[0]))
        print(f"  Traffic Cluster {i+1}: dx = {c[0]:>5.1f} px, dy = {c[1]:>5.1f} px | Velocity = {speed_mag:>5.1f} px/s, Heading = {angle_deg:>5.1f} deg")

    # -------------------------------------------------------------------------
    section_header("6. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What are the mathematical guarantees and limitations of Lloyd's algorithm for K-Means?
A1: Mathematical Guarantees:
    Lloyd's algorithm is guaranteed to converge monotonically to a local minimum or saddle point of the
    Within-Cluster Sum of Squares (Inertia) objective:
    - At the Expectation step, assigning each point to its closest centroid strictly decreases or keeps constant the WCSS.
    - At the Maximization step, setting mu_k to the arithmetic mean strictly minimizes the quadratic Euclidean distance sum.
    Limitations:
    - Convergence to the global optimum is NOT guaranteed (NP-hard in general). It can become trapped in sub-optimal local minima.
    - K-Means++ mitigates this by initializing centroids with probability proportional to D(x)^2, providing an O(log K) approximation ratio.
    - Standard K-Means assumes spherical clusters with equal variance and struggles on elongated, manifold, or density-varying clusters.

Q2: How was K-Means clustering historically used in YOLO (from YOLOv2 to YOLOv7) for anchor box generation?
A2: Prior to anchor-free architectures like YOLO11, YOLO utilized predefined anchor boxes (bounding box priors).
    Instead of manually guessing anchor box aspect ratios:
    1. The YOLO team ran K-Means clustering on all ground-truth bounding box dimensions [width, height] in the training dataset.
    2. Crucially, they did not use standard Euclidean distance (since a 10px error on a 200px box is minor, but fatal on a 15px box).
    3. They defined a custom distance metric based on Intersection over Union:
       d(box, centroid) = 1.0 - IoU(box, centroid)
    4. Clustering with K=5 or K=9 identified the most representative box proportions, giving the network better initial priors
       and accelerating bounding box regression convergence. In YOLO11, anchor-free decoupled heads replace this by predicting
       direct distance offsets via Distribution Focal Loss (DFL).
""")

if __name__ == "__main__":
    main()
