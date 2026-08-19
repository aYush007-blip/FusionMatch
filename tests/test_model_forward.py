"""Unit tests for FusionMatch Model Architecture, Gated Fusion, and Quality Proxies."""

import pytest
import numpy as np
from PIL import Image
import torch
from src.models.fusion_match_model import FusionMatchModel
from src.models.quality_proxies import (
    image_quality_score,
    text_quality_score,
    compute_single_image_quality,
    compute_single_text_quality,
)
from src.models.gated_fusion import GatedFusion
from src.models.projection_head import ProjectionHead


def test_forward_shapes():
    """Verify that forward pass produces exact expected tensor output dimensions."""
    model = FusionMatchModel(use_mock=True, embed_dim=256)
    b = 4
    pixel_values = torch.randn(b, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (b, 32))
    attention_mask = torch.ones(b, 32, dtype=torch.long)
    q_v = torch.rand(b)
    q_t = torch.rand(b)

    emb, gates = model(pixel_values, input_ids, attention_mask, q_v, q_t)
    assert emb.shape == (b, 256), f"Expected shape (4, 256), got {emb.shape}"
    assert gates.shape == (b, 2), f"Expected shape (4, 2), got {gates.shape}"


def test_gate_softmax_normalization():
    """Verify that fusion gates satisfy convex combination constraint (g_v + g_t == 1.0)."""
    fusion = GatedFusion(vision_dim=768, text_dim=768, shared_dim=768)
    b = 8
    v_pool = torch.randn(b, 768)
    t_pool = torch.randn(b, 768)
    q_v = torch.rand(b)
    q_t = torch.rand(b)

    fused, gates = fusion(v_pool, t_pool, q_v, q_t)
    assert fused.shape == (b, 768)
    assert gates.shape == (b, 2)
    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones(b), atol=1e-5), "Gate weights must sum to 1.0"


def test_embedding_l2_unit_norm():
    """Verify that projection head produces strict unit-length vectors on hypersphere."""
    proj = ProjectionHead(in_dim=768, hidden_dim=512, out_dim=256)
    b = 16
    x = torch.randn(b, 768)
    emb = proj(x)
    assert emb.shape == (b, 256)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(b), atol=1e-4), "Embeddings must be L2 unit normalized"


def test_multi_view_pooling():
    """Verify multi-angle visual perspective aggregation across K views."""
    model = FusionMatchModel(use_mock=True, embed_dim=256)
    b, k = 3, 4  # 3 SKUs, 4 image angles each
    pixel_values = torch.randn(b, k, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (b, 24))
    attention_mask = torch.ones(b, 24, dtype=torch.long)

    emb, gates = model(pixel_values, input_ids, attention_mask)
    assert emb.shape == (b, 256)
    assert gates.shape == (b, 2)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(b), atol=1e-4)


def test_gradient_isolation_frozen_backbone():
    """Verify that frozen backbone parameters receive zero gradients while heads train."""
    model = FusionMatchModel(
        use_mock=True,
        freeze_vision=True,
        freeze_text=True,
        embed_dim=256,
    )
    b = 2
    pixel_values = torch.randn(b, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (b, 16))
    attention_mask = torch.ones(b, 16, dtype=torch.long)
    q_v = torch.tensor([0.8, 0.4])
    q_t = torch.tensor([0.9, 0.2])

    emb, gates = model(pixel_values, input_ids, attention_mask, q_v, q_t)
    target = torch.randn(b, 256)
    target = torch.nn.functional.normalize(target, dim=-1)
    loss = 1.0 - (emb * target).sum(dim=-1).mean()
    loss.backward()

    # Backbone parameters should have requires_grad=False and grad=None
    for name, p in model.encoder.named_parameters():
        assert not p.requires_grad, f"Parameter {name} in backbone should be frozen"
        assert p.grad is None, f"Parameter {name} should receive no gradient"

    # Fusion gate and projection head should have non-zero gradients
    assert model.fusion.gate_net[0].weight.grad is not None
    assert model.proj_head.net[0].weight.grad is not None


def test_quality_proxies():
    """Verify quality score bounds and sensitivity to blur / empty text."""
    # 1. Image quality proxy test
    sharp_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # High frequency noise gives high Laplacian variance
    sharp_score = compute_single_image_quality(sharp_img)
    assert 0.0 <= sharp_score <= 1.0

    # Flat image gives zero Laplacian variance
    flat_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    flat_score = compute_single_image_quality(flat_img)
    assert 0.0 <= flat_score <= 0.4
    assert sharp_score > flat_score

    # Batch tensor image scorer
    batch_img_tensor = torch.rand(4, 3, 224, 224)
    img_scores = image_quality_score(batch_img_tensor)
    assert img_scores.shape == (4,)
    assert (img_scores >= 0.0).all() and (img_scores <= 1.0).all()

    # 2. Text quality proxy test
    empty_score = compute_single_text_quality("")
    assert empty_score == 0.0

    long_text = "This is a detailed and high quality e-commerce product title with specifications."
    long_score = compute_single_text_quality(long_text)
    assert 0.0 < long_score <= 1.0
    assert long_score > empty_score

    batch_texts = ["Short title", "", "Long descriptive title with multiple attributes"]
    text_scores = text_quality_score(batch_texts)
    assert text_scores.shape == (3,)
    assert text_scores[1] == 0.0
    assert text_scores[2] > text_scores[0]


def test_param_budget_summary():
    """Verify parameter counting and component budget report."""
    model = FusionMatchModel(use_mock=True, freeze_vision=True, freeze_text=True)
    summary = model.get_param_budget_summary()
    assert "total_params" in summary
    assert "trainable_params" in summary
    assert summary["trainable_params"] > 0
    assert summary["trainable_params"] < summary["total_params"]
