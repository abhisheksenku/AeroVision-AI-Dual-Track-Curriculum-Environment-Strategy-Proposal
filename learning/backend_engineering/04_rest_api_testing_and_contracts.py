"""
=============================================================================
AeroVision AI - Backend Engineering Foundations: Module 04
=============================================================================
API Testing, Multipart File Uploads, and Contract Verification

Learning Objectives:
1. Test multipart/form-data video upload endpoints via TestClient.
2. Verify HTTP headers, Content-Type, and CORS preflight options.
3. Validate API contract fidelity against Pydantic response models.
4. Test edge cases: empty files, invalid extensions, and out-of-range parameters.
"""

import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

app = FastAPI(title="Upload Contract Demo")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadSuccessResponse(BaseModel):
    filename: str
    file_size_bytes: int
    content_type: str
    mission_name: str

@app.post("/api/v1/videos/upload", response_model=UploadSuccessResponse, status_code=status.HTTP_201_CREATED)
async def upload_drone_video(
    mission_name: str = Form(...),
    video_file: UploadFile = File(...)
):
    # Validate extension
    allowed_exts = [".mp4", ".avi", ".mov", ".mkv"]
    filename = video_file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Allowed: {allowed_exts}"
        )

    # Read bytes to verify content
    content = await video_file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    return {
        "filename": filename,
        "file_size_bytes": len(content),
        "content_type": video_file.content_type or "application/octet-stream",
        "mission_name": mission_name
    }

def demonstrate_api_testing():
    print("=" * 70)
    print("STEP 1: MULTIPART UPLOAD TESTING & VALIDATION RULES")
    print("=" * 70)

    client = TestClient(app)

    # 1. Valid video upload (simulating 5KB dummy MP4 bytes)
    fake_mp4_bytes = b"\x00\x00\x00\x18ftypmp42" + b"A" * 5000
    files = {"video_file": ("flight_mission.mp4", io.BytesIO(fake_mp4_bytes), "video/mp4")}
    data = {"mission_name": "SURVEILLANCE_RUN_01"}

    res = client.post("/api/v1/videos/upload", data=data, files=files)
    print(f"POST /api/v1/videos/upload (Valid MP4) -> Status {res.status_code}")
    print(f"Response Body: {res.json()}")

    assert res.status_code == 201
    assert res.json()["filename"] == "flight_mission.mp4"
    assert res.json()["file_size_bytes"] == len(fake_mp4_bytes)

    # 2. Invalid Extension (.txt)
    bad_files = {"video_file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")}
    res_bad_ext = client.post("/api/v1/videos/upload", data=data, files=bad_files)
    print(f"\nPOST Invalid Extension (.txt) -> Status {res_bad_ext.status_code}")
    print(f"Error Detail: {res_bad_ext.json()['detail']}")
    assert res_bad_ext.status_code == 400

    # 3. Empty File
    empty_files = {"video_file": ("empty.mp4", io.BytesIO(b""), "video/mp4")}
    res_empty = client.post("/api/v1/videos/upload", data=data, files=empty_files)
    print(f"\nPOST Empty File (0 bytes) -> Status {res_empty.status_code}")
    assert res_empty.status_code == 400

    # 4. CORS Preflight Check
    cors_res = client.options(
        "/api/v1/videos/upload",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST"
        }
    )
    print(f"\nCORS Preflight (OPTIONS) -> Status {cors_res.status_code}")
    assert cors_res.status_code == 200
    assert "access-control-allow-origin" in cors_res.headers

    print("\n[SUCCESS] Multipart uploads, input validation, and CORS contracts verified.")
    print()

def main():
    demonstrate_api_testing()
    print("=" * 70)
    print("PHASE 8 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
