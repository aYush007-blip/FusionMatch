"""FastAPI Production Serving Application for FusionMatch Deduplication Engine."""

from typing import Optional
from pathlib import Path
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .schemas import (
    CheckRequest,
    CheckResponse,
    BatchCheckRequest,
    BatchCheckResponse,
    HealthResponse,
)
from .inference import FusionMatchInferenceEngine
from .logging_config import setup_logger

engine: Optional[FusionMatchInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes logging, loads ONNX runtime session, FAISS index, and calibrated thresholds."""
    global engine
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logger(log_level)
    logger.info("Initializing FusionMatch Serving Application...")

    onnx_path = os.getenv("ONNX_PATH", "artifacts/onnx/fusion_match_int8.onnx")
    if not Path(onnx_path).exists():
        # Fallback to fp32 onnx if int8 not yet created
        onnx_path = os.getenv("ONNX_FP32_PATH", "artifacts/onnx/fusion_match_fp32.onnx")

    index_path = os.getenv("INDEX_PATH", "artifacts/index/index.faiss")
    id_map_path = os.getenv("ID_MAP_PATH", "artifacts/index/id_map.json")
    thresholds_path = os.getenv("THRESHOLDS_PATH", "artifacts/index/thresholds.json")

    # If artifacts directory is missing during development/testing, initialize fallback test engine
    if not (Path(onnx_path).exists() and Path(index_path).exists() and Path(id_map_path).exists()):
        logger.warning("Production artifacts not found on disk. Initializing auto-generated test engine...")
        _ensure_test_artifacts(onnx_path, index_path, id_map_path, thresholds_path)

    try:
        engine = FusionMatchInferenceEngine(
            onnx_path=onnx_path,
            index_path=index_path,
            id_map_path=id_map_path,
            thresholds_path=thresholds_path if Path(thresholds_path).exists() else None,
        )
        logger.info(f"FusionMatch engine loaded successfully ({engine.index.ntotal} vectors indexed).")
    except Exception as e:
        logger.exception(f"Failed to initialize inference engine: {e}")
        raise RuntimeError(f"Engine startup failed: {e}")

    yield
    logger.info("Shutting down FusionMatch serving application.")


def _ensure_test_artifacts(onnx_path: str, index_path: str, id_map_path: str, thresholds_path: str) -> None:
    """Generates minimal valid test artifacts if running before full pipeline export."""
    import numpy as np
    import json
    from ..models.fusion_match_model import FusionMatchModel
    from ..export.to_onnx import export_to_onnx
    from ..indexing.build_index import IndexBuilder

    model = FusionMatchModel(use_mock=True, embed_dim=256)
    export_to_onnx(model, onnx_path, use_mock=True)

    n, d = 64, 256
    embs = np.random.randn(n, d).astype(np.float32)
    skus = [f"B0{i:08d}" for i in range(n)]

    builder = IndexBuilder(embed_dim=d, nlist=4, m=8, nbits=4)
    builder.build(embs, skus, save_dir=Path(index_path).parent)

    with open(thresholds_path, "w") as f:
        json.dump({"__default__": 0.70, "SHOES": 0.75}, f, indent=2)


app = FastAPI(
    title="FusionMatch Deduplication API",
    description="Cross-Modal & Multi-View Product Deduplication Engine with ONNX Runtime & FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Returns service health status and current index footprint."""
    if engine is None or engine.index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready",
        )
    return HealthResponse(
        status="ok",
        indexed_vectors=engine.index.ntotal,
        embed_dim=256,
        device="CPU",
    )


@app.post("/v1/check", response_model=CheckResponse, tags=["Deduplication"])
def check_duplicate(req: CheckRequest):
    """Processes a single product listing (image + title) for catalog duplication."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    try:
        return engine.check_single(
            image_base64=req.image_base64,
            title=req.title,
            category=req.category,
            top_k=req.top_k,
        )
    except Exception as e:
        logger.exception("Error processing single check request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/v1/check/batch", response_model=BatchCheckResponse, tags=["Deduplication"])
def check_duplicate_batch(req: BatchCheckRequest):
    """Processes a batch of candidate product listings with high-throughput vector retrieval."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    try:
        results = engine.check_batch(req.items, top_k=req.top_k)
        return BatchCheckResponse(results=results)
    except Exception as e:
        logger.exception("Error processing batch check request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )
