# -------------------------------------------------------------
# Stage 1: Build Dependencies & Cache Model Weights
# -------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download SigLIP model into local image cache
RUN python -c "from transformers import AutoProcessor, AutoModel; \
    AutoProcessor.from_pretrained('google/siglip-base-patch16-224'); \
    AutoModel.from_pretrained('google/siglip-base-patch16-224')"

# -------------------------------------------------------------
# Stage 2: Final Runtime Image
# -------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub

# Create non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Copy dependencies and pre-cached models from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /root/.cache /home/appuser/.cache

# Copy application source code
COPY --chown=appuser:appuser src /app/src

USER appuser
EXPOSE 8000

# Healthcheck for orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]