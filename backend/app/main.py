"""
=============================================================================
AeroVision AI - FastAPI Production Application Root
=============================================================================
Main web service integrating Computer Vision pipelines, SQLite persistence,
asynchronous worker dispatch, and RESTful analytics endpoints.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from backend.app.db.database import engine, Base
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.jobs import router as jobs_router
from backend.app.api.v1.analytics import router as analytics_router

# 1. Initialize Database Tables on Startup
Base.metadata.create_all(bind=engine)

# 2. Instantiate FastAPI Application
app = FastAPI(
    title="AeroVision AI - Intelligent Drone Surveillance & Analytics Platform",
    description="Production REST API providing high-throughput drone video tracking, object detection, and compliance auditing.",
    version="1.0.0"
)

# 3. Configure Cross-Origin Resource Sharing (CORS) for React Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Mount Static Directories for Video Playback
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
app.mount("/media/uploads", StaticFiles(directory="data/uploads"), name="uploads")
app.mount("/media/processed", StaticFiles(directory="data/processed"), name="processed")

# 5. Register Versioned API Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")

@app.get("/")
def get_root():
    return {
        "platform": "AeroVision AI",
        "status": "OPERATIONAL",
        "version": "1.0.0",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
