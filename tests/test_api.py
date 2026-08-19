"""Unit and Integration Tests for FastAPI Serving and ONNX Inference Engine."""

import pytest
import io
import base64
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path

from src.models.fusion_match_model import FusionMatchModel
from src.export.to_onnx import export_to_onnx
from src.export.quantize import quantize_dynamic_int8
from src.serving.main import app


@pytest.fixture(scope="module")
def sample_image_b64() -> str:
    """Generates a valid Base64-encoded test JPEG image."""
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture(scope="module")
def test_client():
    """Initializes FastAPI test client with lifespan context."""
    with TestClient(app) as client:
        yield client


def test_health_check(test_client):
    """Verify /health endpoint returns active status and vector count."""
    resp = test_client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_vectors"] > 0
    assert data["embed_dim"] == 256


def test_single_check_endpoint(test_client, sample_image_b64):
    """Verify /v1/check returns valid duplicate predictions and candidates."""
    payload = {
        "image_base64": sample_image_b64,
        "title": "AmazonBasics High-Speed HDMI Cable, 6 Feet, 1-Pack",
        "category": "ELECTRONIC_CABLE",
        "top_k": 3,
    }
    resp = test_client.post("/v1/check", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    data = resp.json()
    assert "is_duplicate" in data
    assert isinstance(data["is_duplicate"], bool)
    assert "threshold_used" in data
    assert 0.3 <= data["threshold_used"] <= 0.99
    assert "candidates" in data
    assert len(data["candidates"]) <= 3

    for candidate in data["candidates"]:
        assert "sku_id" in candidate
        assert 0.0 <= candidate["similarity"] <= 1.0

    assert "gate_weights" in data
    assert "visual" in data["gate_weights"]
    assert "textual" in data["gate_weights"]
    gate_sum = data["gate_weights"]["visual"] + data["gate_weights"]["textual"]
    assert np.isclose(gate_sum, 1.0, atol=1e-3)


def test_batch_check_endpoint(test_client, sample_image_b64):
    """Verify /v1/check/batch handles multiple items with consistent output."""
    items = [
        {
            "item_id": "item_1",
            "image_base64": sample_image_b64,
            "title": "Ergonomic Office Chair with Lumbar Support",
            "category": "CHAIR",
            "top_k": 2,
        },
        {
            "item_id": "item_2",
            "image_base64": sample_image_b64,
            "title": "Wireless Bluetooth Mouse Black",
            "category": "ELECTRONICS",
            "top_k": 2,
        },
    ]
    resp = test_client.post("/v1/check/batch", json={"items": items, "top_k": 2})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["item_id"] == "item_1"
    assert data["results"][1]["item_id"] == "item_2"


def test_missing_modality_graceful_degradation(test_client, sample_image_b64):
    """Verify service gracefully handles empty/missing text title without error."""
    payload = {
        "image_base64": sample_image_b64,
        "title": "",
        "top_k": 3,
    }
    resp = test_client.post("/v1/check", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_duplicate"] in [True, False]
    assert data["gate_weights"]["visual"] > 0.0


def test_onnx_export_and_quantization_roundtrip():
    """Verify ONNX export, INT8 quantization, and session execution."""
    model = FusionMatchModel(use_mock=True, embed_dim=256)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fp32_path = Path(tmp_dir) / "model_fp32.onnx"
        int8_path = Path(tmp_dir) / "model_int8.onnx"

        export_to_onnx(model, fp32_path, use_mock=True)
        assert fp32_path.exists()

        quantize_dynamic_int8(fp32_path, int8_path)
        assert int8_path.exists()
