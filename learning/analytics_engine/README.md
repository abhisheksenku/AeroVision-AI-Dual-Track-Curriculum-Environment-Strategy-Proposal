# Phase 7: Drone Analytics Engine

This module covers trajectory analytics, time-series moving averages, spatial density heatmaps via Gaussian Kernel Density Estimation (KDE), rule-based event detection (Point-in-Polygon), and statistical summary export.

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_time_series_trajectory_analysis.py` | Time-Series & Trajectories | Exponential Moving Average ($S_t = \alpha Y_t + (1-\alpha)S_{t-1}$), compass heading angles ($\theta = \arctan2(\Delta x, -\Delta y)$), velocity profiling, dwell-time & stoppage detection. |
| `02_spatial_density_and_heatmaps.py` | Spatial Density & Heatmaps | 2D spatial discretization, Gaussian Kernel Density Estimation ($G = \exp(-\frac{dx^2+dy^2}{2\sigma^2})$), pseudo-color mapping (`COLORMAP_JET`), alpha-blended congestion heatmaps. |
| `03_rule_based_event_detection.py` | Point-in-Polygon & Security Rules | Ray-Casting algorithm (Jordan curve theorem), speed violation gating, restricted hazard zone incursions, severity tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`). |
| `04_statistical_summaries_and_profiling.py` | Statistical Summary & JSON Export | Percentiles ($P_{95}$), IQR, estimated hourly flow rate, structured mission JSON serialization for FastAPI and LangGraph RAG ingestion. |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/analytics_engine/01_time_series_trajectory_analysis.py
python learning/analytics_engine/02_spatial_density_and_heatmaps.py
python learning/analytics_engine/03_rule_based_event_detection.py
python learning/analytics_engine/04_statistical_summaries_and_profiling.py
```

---

## Analytics & ML Systems Engineer Interview Questions

1. **How do you efficiently check whether an aerial vehicle or pedestrian has entered a restricted polygon zone?**
   *Answer*: We use the **Ray-Casting Algorithm** (based on the Jordan Curve Theorem). A horizontal ray is projected from the test point $(x_0, y_0)$ to positive infinity. We compute edge-ray intersections with all $N$ polygon segments. If the count of intersections is odd, the point is strictly inside the polygon; if even, it is outside. For static inspection zones with thousands of moving objects, we first apply an axis-aligned bounding box (AABB) broad-phase test ($\mathcal{O}(1)$), only evaluating ray-casting when the point lies within the zone's spatial bounds.

2. **Why is the 95th percentile ($P_{95}$) preferred over the mean for traffic speed monitoring?**
   *Answer*: The sample mean is heavily vulnerable to extreme skewness and sensor noise (e.g., detector jitter on stopped vehicles or temporary tracking leaps). $P_{95}$ reveals the speed threshold below which 95% of traffic flows, effectively highlighting true persistent over-speeders while ignoring single-frame velocity spikes and stationary vehicle tails.

3. **How does spatial Kernel Density Estimation (KDE) differ from simple grid binning?**
   *Answer*: Grid binning discretizes space into rigid square bins (e.g. $32 \times 32$ px), suffering from boundary artifacts where two observations situated 1 pixel apart across a cell border are counted in separate bins. Gaussian KDE splats a continuous 2D Gaussian probability density function over neighboring pixels, providing smooth, differentiable heatmaps that accurately reflect true spatial congestion without grid boundary distortion.
