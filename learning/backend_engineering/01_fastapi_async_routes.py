"""
=============================================================================
AeroVision AI - Backend Engineering Foundations: Module 01
=============================================================================
FastAPI Asynchronous Routing, Pydantic Data Contracts & Error Handling

Learning Objectives:
1. Master FastAPI asynchronous request lifecycles (async def vs def).
2. Validate incoming HTTP payloads using Pydantic v2 BaseModels.
3. Construct RESTful HTTP status codes (200, 201, 202, 400, 404, 422).
4. Structure modular application routing using APIRouter.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.testclient import TestClient

# 1. Pydantic Request & Response Data Contracts
class MissionCreateRequest(BaseModel):
    mission_id: str = Field(..., min_length=3, max_length=50, description="Unique drone flight mission ID")
    flight_altitude_m: float = Field(..., gt=0.0, le=500.0, description="Operating altitude in meters")
    target_classes: List[str] = Field(default_factory=lambda: ["car", "truck", "person"])

class MissionResponse(BaseModel):
    mission_id: str
    flight_altitude_m: float
    status: str
    target_classes: List[str]

# 2. Modular APIRouter
router = APIRouter(prefix="/api/v1/missions", tags=["missions"])
_db_store = {}

@router.post("/", response_model=MissionResponse, status_code=status.HTTP_201_CREATED)
async def create_mission(payload: MissionCreateRequest):
    if payload.mission_id in _db_store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mission '{payload.mission_id}' already exists."
        )
    record = {
        "mission_id": payload.mission_id,
        "flight_altitude_m": payload.flight_altitude_m,
        "status": "CONFIGURED",
        "target_classes": payload.target_classes
    }
    _db_store[payload.mission_id] = record
    return record

@router.get("/{mission_id}", response_model=MissionResponse)
async def get_mission(mission_id: str):
    if mission_id not in _db_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found."
        )
    return _db_store[mission_id]

# 3. FastAPI App Initialization
app = FastAPI(title="AeroVision AI - Learning API")
app.include_router(router)

def demonstrate_fastapi_contracts():
    print("=" * 70)
    print("STEP 1: FASTAPI REQUEST VALIDATION & HTTP STATUS CODES")
    print("=" * 70)

    client = TestClient(app)

    # Test 1: Successful creation (201 Created)
    valid_payload = {
        "mission_id": "UAV_ALPHA_01",
        "flight_altitude_m": 45.0,
        "target_classes": ["car", "bus", "person"]
    }
    res = client.post("/api/v1/missions/", json=valid_payload)
    print(f"POST /api/v1/missions/ -> Status {res.status_code}")
    print(f"Response Body: {res.json()}")
    assert res.status_code == 201
    assert res.json()["status"] == "CONFIGURED"

    # Test 2: Validation Error on negative altitude (422 Unprocessable Entity)
    invalid_payload = {
        "mission_id": "UAV_FAIL",
        "flight_altitude_m": -10.0 # Invalid: gt=0.0
    }
    res_err = client.post("/api/v1/missions/", json=invalid_payload)
    print(f"\nPOST Invalid Payload -> Status {res_err.status_code} (Pydantic caught invalid altitude)")
    assert res_err.status_code == 422

    # Test 3: Duplicate error (400 Bad Request)
    res_dup = client.post("/api/v1/missions/", json=valid_payload)
    print(f"POST Duplicate Mission -> Status {res_dup.status_code}")
    assert res_dup.status_code == 400

    # Test 4: Retrieve existing (200 OK)
    res_get = client.get("/api/v1/missions/UAV_ALPHA_01")
    print(f"GET Existing Mission -> Status {res_get.status_code}")
    assert res_get.status_code == 200

    # Test 5: Not Found (404 Not Found)
    res_404 = client.get("/api/v1/missions/NON_EXISTENT")
    print(f"GET Non-existent Mission -> Status {res_404.status_code}")
    assert res_404.status_code == 404

    print("\n[SUCCESS] FastAPI request lifecycle and Pydantic validation contracts verified.")
    print()

def main():
    demonstrate_fastapi_contracts()
    print("=" * 70)
    print("PHASE 8 - MODULE 01: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
