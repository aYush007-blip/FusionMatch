"""Pydantic Request & Response Schemas for FusionMatch Deduplication API."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """Retrieved candidate duplicate product match."""
    sku_id: str = Field(..., description="Matched catalog SKU identifier")
    similarity: float = Field(..., description="Cosine similarity score in range [0.0, 1.0]")


class CheckRequest(BaseModel):
    """Single product duplicate check request."""
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG product image")
    title: Optional[str] = Field(None, description="Listing title or product description text")
    category: Optional[str] = Field(None, description="Product category for selecting calibrated decision threshold")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of nearest neighbor candidates to return")


class CheckResponse(BaseModel):
    """Single product duplicate check response."""
    is_duplicate: bool = Field(..., description="True if top candidate similarity exceeds calibrated category threshold")
    threshold_used: float = Field(..., description="Calibrated decision threshold applied for this category")
    candidates: List[Candidate] = Field(default_factory=list, description="Top-K retrieved duplicate candidates")
    gate_weights: Dict[str, float] = Field(..., description="Softmax gate weights allocated to visual and textual modalities")


class BatchCheckItem(CheckRequest):
    """Single item within a batch check request."""
    item_id: str = Field(..., description="Unique caller-assigned item identifier")


class BatchCheckRequest(BaseModel):
    """Batch duplicate check request payload."""
    items: List[BatchCheckItem] = Field(..., min_length=1, max_length=100, description="List of items to check")
    top_k: int = Field(5, ge=1, le=50, description="Number of candidate matches per item")


class BatchCheckResult(CheckResponse):
    """Single check result within a batch check response."""
    item_id: str = Field(..., description="Caller-assigned item identifier")


class BatchCheckResponse(BaseModel):
    """Batch duplicate check response payload."""
    results: List[BatchCheckResult] = Field(..., description="List of check responses for each submitted item")


class HealthResponse(BaseModel):
    """Service health and index status response."""
    status: str = Field("ok", description="Service health indicator")
    indexed_vectors: int = Field(..., description="Total vectors active in FAISS index")
    embed_dim: int = Field(..., description="Embedding dimension")
    device: str = Field("CPU", description="Inference execution device")
