# AeroVision AI – Intelligent Drone Video Analytics & Autonomous AI Agent Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.15%2BCPU-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg)](https://tailwindcss.com/)

AeroVision AI is a portfolio-grade drone video intelligence platform designed to ingest aerial video feeds, detect and track small/dense objects using Ultralytics YOLO11 and ByteTrack, compute movement analytics and zone statistics, and empower an autonomous LangGraph AI Agent to answer questions and produce evidence-grounded aerial inspection reports.

---

## 1. System Architecture

```text
Drone Video Feed
       │
       ▼
OpenCV Sequential Frame Extractor (Memory-Safe Streaming)
       │
       ▼
Ultralytics YOLO11 Object Detector (CPU Optimized)
       │
       ▼
ByteTrack Multi-Object Tracker (Lapx Linear Assignment)
       │
       ▼
Movement & Zone Analytics Engine (Trajectories & Class Counts)
       │
       ▼
FastAPI Backend (Async Video Worker + REST APIs)
       │
 ┌─────┴─────────────────────────────────┐
 │                                       │
 ▼                                       ▼
PostgreSQL / SQLite Database     LangGraph AI Agent
                                         │
                                         ▼
                                 Qdrant Vector DB (RAG)
                                         │
                                         ▼
                                 Google Gemini API
                                         │
                                         ▼
                              Evidence-Grounded Reports
```

---

## 2. Hardware Compatibility Profile

AeroVision AI is engineered to run reliably on resource-conscious edge and development machines:
- **Target OS**: Windows 11 / Linux (tested on Intel Core 5 120U with integrated graphics).
- **GPU Requirement**: **None (No CUDA required)**. Uses lightweight CPU wheels and quantized architectures.
- **Memory Consumption**: Low-memory generator pipeline processes frames sequentially without loading multi-gigabyte video files into RAM.

---

## 3. Project Structure

```text
AeroVision-AI/
├── .env.example                       # Environment variables template
├── .gitignore                         # Version control exclusions
├── DEVELOPMENT_PROGRESS.md            # Living milestone progress log
├── README.md                          # Project overview and setup
├── requirements.txt                   # Frozen Python dependencies
├── .venv/                             # Isolated Python 3.11 virtual environment
│
├── backend/                           # FastAPI backend
│   ├── app/
│   │   ├── api/v1/                    # REST endpoints (videos, analytics, agent)
│   │   ├── core/                      # Settings, logging, security
│   │   ├── db/                        # Database schemas & sessions
│   │   ├── cv/                        # OpenCV, YOLO11, ByteTrack modules
│   │   ├── agent/                     # LangGraph workflow, tools, RAG
│   │   └── workers/                   # Background video processing tasks
│   └── tests/                         # Pytest test suite
│
├── data/                              # Data directories (gitignored)
│   ├── samples/                       # Test drone clips (VisDrone benchmark)
│   ├── uploads/                       # User-uploaded raw videos
│   ├── processed/                     # Annotated output videos & snapshots
│   └── reference_docs/                # RAG safety & flight documentation
│
├── models/
│   └── weights/                       # Pretrained model weights (e.g., yolo11n.pt)
│
└── frontend/                          # React 19 + TypeScript + Vite Dashboard
    ├── src/
    │   ├── components/                # UI widgets & video player
    │   ├── pages/                     # Application views
    │   └── services/                  # Backend API clients
    ├── tailwind.config.js
    └── vite.config.ts
```

---

## 4. Quickstart Setup Guide

### Prerequisites
- Python 3.11.x
- Node.js 20.x LTS + npm
- Docker & Docker Compose (Optional for containerized run)
- Git

### Option A: Run via Docker Compose (Recommended)
```bash
docker compose up --build
```
- Frontend Dashboard: [http://localhost](http://localhost)
- Backend REST API: [http://localhost/api/v1/health](http://localhost/api/v1/health)
- API Documentation (Swagger): [http://localhost/api/docs](http://localhost/api/docs)

### Option B: Local Development Setup

#### 1. Clone & Set Up Environment
```powershell
# Clone the repository
git clone https://github.com/abhisheksenku/AeroVision-AI-Dual-Track-Curriculum-Environment-Strategy-Proposal.git
cd AeroVision-AI-Dual-Track-Curriculum-Environment-Strategy-Proposal

# Create virtual environment
python -m venv .venv

# Activate virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1
```

#### 2. Install Python Dependencies
```powershell
# Install PyTorch CPU
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Computer Vision Stack & Platform Dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### 3. Run Backend Server
```powershell
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Run Frontend Dashboard
```powershell
cd frontend
npm install
npm run build
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

#### 5. Run Verification & Test Suites
```powershell
.\.venv\Scripts\python.exe backend/tests/test_rag_pipeline.py
.\.venv\Scripts\python.exe backend/tests/test_agent_graph.py
.\.venv\Scripts\python.exe backend/tests/test_mlops_pipeline.py
```

---

## 5. Implementation Roadmap Status (14 / 14 Phases Complete)

- [x] **Phase 0: Development Environment & Engineering Foundations** (Dual `.venv` / `.venv_tf`, Smart App Control shims)
- [x] **Phase 1: Mathematics & Classical ML Foundations** (Vectors, Gradients, Linear/Logistic, Trees, K-Means)
- [x] **Phase 2: Deep Learning with PyTorch** (Autograd, Custom CNNs, ResNet50, YOLO11 detection backbone)
- [x] **Phase 3: TensorFlow & Keras Edge Track** (Functional MIMO, Callbacks, ResNet Transfer Learning, TFLite INT8 PTQ)
- [x] **Phase 4: Computer Vision Foundations** (OpenCV streaming, optical flow, homography, perspective warping)
- [x] **Phase 5: Modern Object Detection with YOLO11** (Single-stage inference, NMS, Precision-Recall benchmark)
- [x] **Phase 6: Multi-Object Tracking with ByteTrack** (Kalman Filter, Hungarian Algorithm, lapx, track smoothing)
- [x] **Phase 7: Drone Analytics Engine** (Ray-casting polygon geofencing, tripwires, loitering detection, time-to-violation)
- [x] **Phase 8: Production Backend Engineering** (FastAPI, SQLite persistence, async background video worker, REST API)
- [x] **Phase 9: Production Frontend Dashboard** (React 19, TypeScript, TailwindCSS, Lucide, Recharts telemetry)
- [x] **Phase 10: LLM & Generative AI Foundations** (Gemini 2.5 Flash, structured incident reporting, hallucination verification)
- [x] **Phase 11: Regulatory RAG Pipeline** (Qdrant Vector DB, FAA Part 107 / OSHA 1910 / NIST, cosine similarity search)
- [x] **Phase 12: Autonomous AI Agents with LangGraph** (StateGraph, ReAct supervisor multi-agent architecture, self-reflection loop)
- [x] **Phase 13: MLOps, Containerization & Production Deployment** (ONNX Runtime 3.17x CPU speedup, dynamic INT8 quantization, PSI drift engine, multi-stage Docker, Nginx, CI/CD)

For comprehensive engineering logs, benchmarking data, and architectural notes, see [DEVELOPMENT_PROGRESS.md](file:///D:/Projects/AeroVision-AI/DEVELOPMENT_PROGRESS.md) and [AEROVISION_PROJECT_LEARNING_ROADMAP.md](file:///D:/Projects/AeroVision-AI/AEROVISION_PROJECT_LEARNING_ROADMAP.md).
