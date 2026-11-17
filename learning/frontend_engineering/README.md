# Phase 9: Frontend & Data Visualization

This module covers frontend architecture for real-time computer vision systems: React 19, Vite, TypeScript, Tailwind CSS, Recharts data visualization, and asynchronous job polling.

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_react_polling_and_state_machine.py` | State Machines & Polling | Asynchronous job state machines (`IDLE` $\to$ `UPLOADING` $\to$ `PROCESSING` $\to$ `COMPLETED`), exponential backoff polling ($\Delta t = \min(\Delta t_{\max}, \Delta t_0 \cdot \gamma^k)$), preventing client-side network congestion. |
| `02_recharts_data_transformations.py` | Recharts Visualizations | Transforming backend relational JSON into multi-bar and donut chart array payloads, semantic Tailwind color token mappings, speed distribution metrics. |
| `03_video_canvas_coordinate_mapping.py` | Responsive Canvas Mapping | Solving the video-to-canvas coordinate mismatch, CSS `object-fit: contain` scaling equations ($\text{scale} = \min(W_c/W_v, H_c/H_v)$), pillarbox/letterbox bar offset calculations. |
| `04_frontend_api_contract_verifier.py` | Contract Verification | Automated integration test verifying exact synchronization between FastAPI response schemas and TypeScript client interfaces (`SystemHealth`, `JobResponse`, `MissionAnalytics`). |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/frontend_engineering/01_react_polling_and_state_machine.py
python learning/frontend_engineering/02_recharts_data_transformations.py
python learning/frontend_engineering/03_video_canvas_coordinate_mapping.py
python learning/frontend_engineering/04_frontend_api_contract_verifier.py
```

---

## Frontend & Fullstack AI Engineer Interview Questions

1. **How do you synchronize an HTML5 canvas overlay with an HTML5 `<video>` element during dynamic window resizing?**
   *Answer*: When a video element uses CSS `object-fit: contain`, its rendered display aspect ratio is preserved by adding horizontal or vertical letterbox padding. Directly querying the bounding box coordinates on the canvas will misalign the detections if the display aspect ratio differs from native resolution. We compute a uniform scale factor $\text{scale} = \min(W_{\text{container}} / W_{\text{native}}, H_{\text{container}} / H_{\text{native}})$, and offset bars $\Delta x = (W_{\text{container}} - W_{\text{native}} \cdot \text{scale})/2$. All bounding boxes and trajectory paths are multiplied by `scale` and shifted by $(\Delta x, \Delta y)$ within a `ResizeObserver` callback loop.

2. **Why is polling with exponential backoff preferred over short polling or raw WebSockets for heavy batch video analysis?**
   *Answer*: Video analysis takes between 10 seconds and several minutes. Fixed short polling (e.g. every 500ms) generates hundreds of redundant HTTP requests, placing needless load on reverse proxies and database connections. WebSockets require maintaining persistent open TCP connections and stateful server gateways, which complicates horizontal autoscaling across serverless or containerized backends. Exponential backoff polling ($\Delta t = \min(\Delta t_{\max}, \Delta t_0 \cdot \gamma^k)$) polls aggressively at the start for fast feedback and gracefully backs off during long batch compute, remaining completely stateless and cache-friendly.
