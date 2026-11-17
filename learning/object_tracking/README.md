# Phase 6: Object Tracking with ByteTrack

This module covers Multi-Object Tracking (MOT), state estimation with Kalman filters, bipartite Hungarian matching, ByteTrack's two-stage association, and tracking evaluation metrics (MOTA, IDF1).

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_kalman_filter_fundamentals.py` | Kalman Filter State Estimation | State transition ($x_k = F x_{k-1} + w$), measurement mapping ($z = H x + v$), covariance extrapolation ($P = F P F^T + Q$), optimal Kalman Gain ($K$), and 2D velocity estimation. |
| `02_hungarian_matching_and_data_association.py` | Hungarian Bipartite Matching | Cost matrix ($1 - \text{IoU}$), Kuhn-Munkres algorithm ($\mathcal{O}(V^3)$), partitioning into Matched Pairs, Unmatched Tracks (misses/exits), and Unmatched Detections (newborn tracks). |
| `03_bytetrack_algorithm_deep_dive.py` | ByteTrack Two-Stage Association | Why SORT fails on occluded low-score detections; Stage 1 matching ($D_{\text{high}}, conf \ge 0.6$); Stage 2 matching ($D_{\text{low}}, 0.1 \le conf < 0.6$); track lifecycle (`Tracked`, `Lost`, `Removed`). |
| `04_tracking_metrics_mota_idf1.py` | Tracking Metrics (MOTA, IDF1) | False Negatives (FN), False Positives (FP), ID Switches (IDSW); Multiple Object Tracking Accuracy ($\text{MOTA} = 1 - \frac{FN + FP + IDSW}{GT}$); Identification F1 ($\text{IDF1}$). |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/object_tracking/01_kalman_filter_fundamentals.py
python learning/object_tracking/02_hungarian_matching_and_data_association.py
python learning/object_tracking/03_bytetrack_algorithm_deep_dive.py
python learning/object_tracking/04_tracking_metrics_mota_idf1.py
```

---

## Computer Vision Tracking Engineer Interview Questions

1. **Why does ByteTrack outperform DeepSORT without needing a deep Re-Identification (ReID) neural network?**
   *Answer*: DeepSORT extracts a 128-dimensional appearance embedding for every bounding box using a secondary CNN, which introduces high computational latency ($\approx 10-20\text{ ms}$ overhead) and frequently fails in aerial drone imagery where vehicles of the same color appear visually indistinguishable from above. Furthermore, DeepSORT discards low-confidence detections ($conf < 0.6$). ByteTrack leverages the simple but powerful observation that low-score detections are often true objects under temporary occlusion or motion blur. By matching unmatched tracks with low-confidence detections in a second stage using pure spatial IoU, ByteTrack achieves higher IDF1 and 30+ FPS throughput without any ReID compute overhead.

2. **What causes an ID Switch (IDSW) in aerial drone tracking?**
   *Answer*: ID switches occur when two objects pass close to each other (e.g. crossing pedestrians or overtaking vehicles), or when an object is temporarily occluded by trees/infrastructure. If the tracker fails to associate the re-emerging detection with the existing track, the old track times out and a new track ID is spawned. In drones, rapid camera yaw rotation or gimbal vibration also induces sudden frame-to-frame bounding box displacement that exceeds the Kalman filter's search gate, triggering false ID switches unless camera motion compensation (CMC) or low-threshold association is used.

3. **What is the difference between MOTA and IDF1?**
   *Answer*: $\text{MOTA}$ is heavily biased towards frame-level object detection accuracy ($FN$ and $FP$), meaning a detector with high detection precision can achieve high MOTA even if it switches track IDs frequently. $\text{IDF1}$ (Identification F1-score) specifically measures identity persistence over time by finding the optimal mapping between ground-truth trajectories and predicted trajectories, making IDF1 the gold standard metric for assessing whether persistent object IDs are maintained.
