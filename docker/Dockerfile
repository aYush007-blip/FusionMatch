# Multi-stage lightweight CPU-optimized Dockerfile for FusionMatch Serving
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for OpenCV, Pillow, and health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
