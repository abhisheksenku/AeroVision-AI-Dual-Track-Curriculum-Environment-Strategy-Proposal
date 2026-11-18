"""
=============================================================================
AeroVision AI - Module 03: Docker Multi-Stage Containerization Patterns
=============================================================================
Teaches production containerization patterns for AI applications:
1. Multi-stage build architecture separating compilation from runtime.
2. Layer caching optimization for fast CI/CD rebuilds.
3. Least-privilege security hardening (non-root execution).
4. Environment variable injection, health checks, and port exposure.
=============================================================================
"""

import re
from typing import Dict, List, Any


# =============================================================================
# 1. DEFINING PRODUCTION DOCKERFILE TEMPLATES
# =============================================================================
SAMPLE_BACKEND_DOCKERFILE = """
# -----------------------------------------------------------------------------
# Stage 1: Build & Dependency Wheel Cache
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install OS build dependencies for compiling native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Final Minimal Runtime Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install libglib2.0 and libgl1 for OpenCV headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libglib2.0-0 \\
    libgl1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Security: Create unprivileged non-root user
RUN groupadd -g 1000 appgroup && \\
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Copy application source code with correct ownership
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup data/regulations /app/data/regulations
COPY --chown=appuser:appgroup models/weights /app/models/weights

# Create media volume mount directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/qdrant_db && \\
    chown -R appuser:appgroup /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""


# =============================================================================
# 2. DOCKERFILE STATIC AUDIT ENGINE
# =============================================================================
class DockerfileAuditor:
    """
    Audits Dockerfile against CIS Docker Benchmarks and production MLOps standards.
    """
    def __init__(self, content: str):
        self.content = content
        self.lines = [line.strip() for line in content.split("\n") if line.strip()]

    def audit(self) -> Dict[str, Any]:
        results = {}

        # 1. Multi-Stage Check
        from_stages = re.findall(r"^FROM\s+(\S+)(?:\s+AS\s+(\w+))?", self.content, re.MULTILINE | re.IGNORECASE)
        results["stages_count"] = len(from_stages)
        results["is_multistage"] = len(from_stages) > 1

        # 2. Security: Non-Root Execution
        has_user = any(line.upper().startswith("USER") for line in self.lines)
        results["has_non_root_user"] = has_user

        # 3. Health Check
        has_healthcheck = any("HEALTHCHECK" in line.upper() for line in self.lines)
        results["has_healthcheck"] = has_healthcheck

        # 4. Layer Cache Hygiene
        has_no_cache = "--no-cache-dir" in self.content
        results["has_no_cache_dir"] = has_no_cache

        # 5. Base Image Pinned
        base_images = [stage[0] for stage in from_stages]
        results["base_images"] = base_images
        results["all_pinned"] = all("latest" not in img for img in base_images)

        return results


# =============================================================================
# 3. VERIFICATION EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: PARSING MULTI-STAGE DOCKER ARCHITECTURE")
    print("=" * 70)
    auditor = DockerfileAuditor(SAMPLE_BACKEND_DOCKERFILE)
    audit = auditor.audit()

    print(f"Total Build Stages Found  : {audit['stages_count']}")
    print(f"Base Images Referenced    : {audit['base_images']}")
    print(f"Multi-Stage Verified      : {audit['is_multistage']}")
    print(f"Non-Root User Hardening   : {audit['has_non_root_user']}")
    print(f"Healthcheck Probes Config : {audit['has_healthcheck']}")
    print(f"pip Cache Suppression     : {audit['has_no_cache_dir']}")
    print(f"Pinned Tag Hygiene (Anti-Latest): {audit['all_pinned']}")

    assert audit["is_multistage"] is True
    assert audit["has_non_root_user"] is True
    assert audit["has_healthcheck"] is True
    assert audit["all_pinned"] is True

    print("\n" + "=" * 70)
    print("STEP 2: VALIDATING PRODUCTION CONTAINER ENVIRONMENT CONTRACTS")
    print("=" * 70)
    env_vars = re.findall(r"ENV\s+(\w+)=([^\n]+)", SAMPLE_BACKEND_DOCKERFILE)
    print("Injected Production Environment Variables:")
    for k, v in env_vars:
        print(f"  * {k:20s} = {v}")

    assert any(k == "PYTHONUNBUFFERED" for k, _ in env_vars)
    assert any(k == "PORT" for k, _ in env_vars)

    print("\n[SUCCESS] Module 03 Docker Containerization Patterns verified successfully!")


if __name__ == "__main__":
    main()
