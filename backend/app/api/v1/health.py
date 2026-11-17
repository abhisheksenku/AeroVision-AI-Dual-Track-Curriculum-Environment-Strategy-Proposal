"""
=============================================================================
AeroVision AI - Healthcheck REST API Route
=============================================================================
System readiness and operational diagnostics endpoint.
"""

import os
import psutil
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.db.database import get_db

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", status_code=status.HTTP_200_OK)
def get_system_health(db: Session = Depends(get_db)):
    """System health check verifying DB connectivity, CPU, RAM, and model weights."""
    # Verify DB
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # Verify Model Weights
    weights_path = "models/weights/yolo11n.pt"
    model_ok = os.path.exists(weights_path)

    # Hardware Metrics
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)

    return {
        "status": "HEALTHY" if (db_ok and model_ok) else "DEGRADED",
        "database_connected": db_ok,
        "model_weights_available": model_ok,
        "system_metrics": {
            "cpu_usage_percent": cpu_percent,
            "ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_free_gb": round(mem.available / (1024**3), 2)
        }
    }
