# MLOps, Model Optimization & Production Deployment

Welcome to the **AeroVision AI MLOps & Production Engineering Curriculum**. This guide details the principles, tooling, and architectures required to transition deep learning models from laboratory research into hardened, scalable, edge-and-cloud containerized production systems.

---

## 1. Model Serialization & Acceleration Runtimes

| Feature | PyTorch Eager (Standard) | TorchScript (JIT) | ONNX Runtime (CPU / OpenVINO) | TensorRT (GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Paradigm** | Python-interpreted dynamic graphs | C++ runtime via JIT compilation | Graph-optimized cross-platform runtime | Native NVIDIA kernel-level optimization |
| **Hardware Targets** | Python environments | Cross-platform C++ embeddable | CPU, Intel iGPU, ARM, CUDA, CoreML | Strictly NVIDIA GPUs |
| **Optimization Pass**| Basic PyTorch caching | Operator fusion, dead code elimination | Constant folding, layer fusion, memory reuse | FP16/INT8 kernel tuning, tensor core alignment |
| **Observed Latency**  | ~5.52 ms | ~4.10 ms | ~1.74 ms (3.17x Speedup) | < 0.80 ms |
| **AeroVision Role**   | Active training & research | Intermediate export intermediate | Production CPU inference backbone | Enterprise edge deployment |

---

## 2. Integer Quantization Mechanics (PTQ vs QAT)

To deploy models onto edge drone platforms with thermal and power constraints (e.g. Intel Core 5, Raspberry Pi, Jetson), models must be quantized from 32-bit floating point ($\text{FP32}$) to 8-bit integers ($\text{INT8}$):

### 2.1 Post-Training Dynamic Quantization (PTQ)
Linear weights are quantized to 8-bit integers offline:
$$q = \text{clip}\left(\left\lfloor \frac{w}{S} \right\rceil + Z, -128, 127\right), \quad S = \frac{\max(w) - \min(w)}{255}$$

During inference:
- Activations remain dynamic FP32 until matrix multiplication.
- Weight matrix multiplications execute using SIMD integer vector registers (AVX2/AVX-512 VNNI).
- **Benefits**: Zero retraining required; cuts model disk and RAM footprint by **74.8%** with negligible accuracy degradation ($\Delta < 0.002$).

### 2.2 Quantization-Aware Training (QAT)
Fake-quantization nodes are inserted during the forward pass:
- Gradients flow through the straight-through estimator (STE).
- The network learns to adjust weights to compensate for rounding errors.
- **Benefits**: Preserves $> 99.5\%$ of mAP even on ultra-low-bit architectures.

---

## 3. Multi-Stage Docker Architecture

Production containerization enforces least-privilege security and minimal image size:

```
┌──────────────────────────────────────┐
│ Stage 1: Build & Wheel Compilation  │
│ FROM python:3.11-slim AS builder     │
│ (build-essential, pip wheel cache)   │
└──────────────────┬───────────────────┘
                   │ COPY --from=builder /root/.local
                   ▼
┌──────────────────────────────────────┐
│ Stage 2: Final Minimal Runtime Image │
│ FROM python:3.11-slim AS runtime     │
│ (libglib, libgl1, curl, non-root)    │
│ USER appuser (UID 1000)              │
│ HEALTHCHECK /api/v1/health           │
└──────────────────────────────────────┘
```

- **Reduced Attack Surface**: Compilers (`gcc`, `g++`, `make`) are discarded in Stage 1 and never ship to production containers.
- **Layer Caching**: `COPY requirements.txt` precedes source code, ensuring Docker caches heavy dependencies when code changes.

---

## 4. Covariate Drift & Population Stability Index (PSI)

Drone camera feeds frequently experience statistical drift due to lighting changes, sun glare, seasonal traffic variations, or route modifications:

$$\text{PSI} = \sum_{i=1}^B \left( P_i - Q_i \right) \cdot \ln\left( \frac{P_i}{Q_i} \right)$$

Where $Q_i$ is baseline training proportion and $P_i$ is observed production batch proportion across quantile bins $i \in \{1, \dots, B\}$:

- $\text{PSI} < 0.10$: **Nominal** (No drift; model calibration certified).
- $0.10 \le \text{PSI} \le 0.25$: **Moderate Drift** (Alert logged in telemetry dashboard).
- $\text{PSI} > 0.25$: **Critical Drift** (Automated trigger for dataset collection & retraining).

---

## 5. MLOps Interview Questions & Answers

### Q1: Why convert PyTorch models to ONNX before deploying to edge drone hardware?
**Answer**:
PyTorch eager execution incurs substantial Python runtime interpreter overhead, GIL contention, and dynamic dispatch latency. Exporting to ONNX produces a standardized static computational graph that runtime engines (ONNX Runtime, OpenVINO) optimize through:
1. **Constant Folding**: Pre-calculating static weights and offsets at compilation time.
2. **Layer Fusion**: Merging `Conv2D + BatchNorm + SiLU` into a single vectorized kernel call, avoiding repeated memory round-trips.
3. **Hardware Tuning**: Leveraging Intel AVX2/AVX-512 hardware vector instructions natively without Python dependencies.

### Q2: How do you handle non-root user permissions when containers must write to mounted volumes?
**Answer**:
1. Create a designated user and group with explicit UID/GID in the Dockerfile:
   `RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser`
2. Pre-create volume directories and grant ownership during container build:
   `RUN mkdir -p /app/data && chown -R appuser:appgroup /app/data`
3. In `docker-compose.yml`, mount host directories with matching permissions or run init containers that align POSIX ACLs.

### Q3: What is the difference between Data Drift, Concept Drift, and Covariate Shift?
**Answer**:
- **Covariate Shift / Data Drift**: The distribution of inputs shifts ($P(X_{\text{train}}) \neq P(X_{\text{prod}})$) while the underlying ground truth relationship remains constant ($P(Y|X)$ is unchanged). Example: Drone flies over wet asphalt instead of dry asphalt.
- **Concept Drift**: The statistical relationship between inputs and outputs changes ($P(Y|X_{\text{train}}) \neq P(Y|X_{\text{prod}})$). Example: Facility speed limits are reduced from 50 km/h to 30 km/h; vehicles moving at 40 km/h are now infractions.
