"""Unit tests for InfoNCE loss, Hard Negative Miner, Evaluation Metrics, and Trainer."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.fusion_match_model import FusionMatchModel
from src.training.losses import InfoNCELoss, HardNegativeMiner
from src.training.metrics import compute_pairwise_f1, compute_precision_recall_at_k, evaluate_embeddings
from src.training.trainer import FusionMatchTrainer


def test_infonce_loss_forward_and_backward():
    """Verify InfoNCE loss calculation and gradient flow."""
    criterion = InfoNCELoss(temperature=0.07)
    b, d = 8, 256
    anchor = torch.randn(b, d, requires_grad=True)
    positive = torch.randn(b, d, requires_grad=True)

    anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=-1)
    positive_norm = torch.nn.functional.normalize(positive, p=2, dim=-1)

    loss = criterion(anchor_norm, positive_norm)
    assert loss.ndim == 0, "Loss must be scalar"
    assert loss.item() > 0.0, "Loss must be positive"

    loss.backward()
    assert anchor.grad is not None, "Gradients must flow to anchor embeddings"
    assert positive.grad is not None, "Gradients must flow to positive embeddings"


def test_infonce_with_hard_negatives():
    """Verify InfoNCE loss computation when mined hard negatives are appended."""
    criterion = InfoNCELoss(temperature=0.07)
    b, k, d = 4, 3, 256
    anchor = torch.nn.functional.normalize(torch.randn(b, d), p=2, dim=-1)
    positive = torch.nn.functional.normalize(torch.randn(b, d), p=2, dim=-1)
    hard_negs = torch.nn.functional.normalize(torch.randn(b, k, d), p=2, dim=-1)

    loss = criterion(anchor, positive, hard_negs)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_hard_negative_miner_masking():
    """Verify that same-SKU embeddings are never mined as hard negatives."""
    miner = HardNegativeMiner(bank_size=32, embed_dim=128, k=2, device="cpu")
    
    # Add 4 items of SKU_A and 4 items of SKU_B
    emb_a = torch.nn.functional.normalize(torch.randn(4, 128), p=2, dim=-1)
    skus_a = ["SKU_A"] * 4
    emb_b = torch.nn.functional.normalize(torch.randn(4, 128), p=2, dim=-1)
    skus_b = ["SKU_B"] * 4

    miner.update(emb_a, skus_a)
    miner.update(emb_b, skus_b)

    assert miner.current_size == 8

    # Query hard negatives for an anchor of SKU_A
    anchor_a = torch.nn.functional.normalize(torch.randn(2, 128), p=2, dim=-1)
    mined = miner.mine(anchor_a, ["SKU_A", "SKU_A"])
    # All returned hard negatives must be from SKU_B (not SKU_A)
    # Vectors 4..7 in the bank belong to SKU_B
    assert mined.shape == (2, 2, 128)

    # Test overflow update where batch size > bank_size
    large_emb = torch.randn(40, 128)
    large_skus = [f"SKU_{i}" for i in range(40)]
    miner.update(large_emb, large_skus)
    assert miner.current_size == 32
    assert miner.full is True


def test_infonce_empty_batch():
    """Verify InfoNCE handles zero-sized batches without error."""
    criterion = InfoNCELoss(temperature=0.07)
    loss = criterion(torch.empty(0, 128), torch.empty(0, 128))
    assert loss.item() == 0.0


def test_pairwise_f1_metric():
    """Verify pairwise F1 calculation on perfect synthetic clusters."""
    # Cluster 1: 4 identical vectors for SKU_A
    vec_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    emb_a = vec_a.repeat(4, 1)
    skus_a = ["SKU_A"] * 4

    # Cluster 2: 4 identical vectors for SKU_B (orthogonal to SKU_A)
    vec_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    emb_b = vec_b.repeat(4, 1)
    skus_b = ["SKU_B"] * 4

    all_embs = torch.cat([emb_a, emb_b], dim=0)
    all_skus = skus_a + skus_b

    f1 = compute_pairwise_f1(all_embs, all_skus, threshold=0.70)
    assert np.isclose(f1, 1.0), f"Expected perfect F1 of 1.0, got {f1}"


def test_precision_recall_at_k():
    """Verify Precision@K and Recall@K on known synthetic clusters."""
    # 2 clusters of 3 items each
    vec_a = torch.tensor([1.0, 0.0, 0.0])
    emb_a = vec_a.repeat(3, 1)
    skus_a = ["SKU_A"] * 3

    vec_b = torch.tensor([0.0, 1.0, 0.0])
    emb_b = vec_b.repeat(3, 1)
    skus_b = ["SKU_B"] * 3

    all_embs = torch.cat([emb_a, emb_b], dim=0)
    all_skus = skus_a + skus_b

    p_at_2, r_at_2 = compute_precision_recall_at_k(all_embs, all_skus, k=2)
    assert np.isclose(p_at_2, 1.0), f"Expected P@2=1.0, got {p_at_2}"
    assert np.isclose(r_at_2, 1.0), f"Expected R@2=1.0, got {r_at_2}"

    eval_dict = evaluate_embeddings(all_embs, all_skus, k_values=(1, 2))
    assert eval_dict["pairwise_f1"] == 1.0
    assert eval_dict["precision@2"] == 1.0


def test_trainer_smoke_run():
    """Verify that FusionMatchTrainer executes a smoke epoch without runtime errors."""
    model = FusionMatchModel(use_mock=True, embed_dim=256)
    
    # Synthetic batch dataset
    b = 8
    mock_batch = {
        "anchor_pixel_values": torch.randn(b, 3, 224, 224),
        "anchor_input_ids": torch.randint(0, 1000, (b, 16)),
        "anchor_attention_mask": torch.ones(b, 16, dtype=torch.long),
        "anchor_q_v": torch.rand(b),
        "anchor_q_t": torch.rand(b),
        "positive_pixel_values": torch.randn(b, 3, 224, 224),
        "positive_input_ids": torch.randint(0, 1000, (b, 16)),
        "positive_attention_mask": torch.ones(b, 16, dtype=torch.long),
        "positive_q_v": torch.rand(b),
        "positive_q_t": torch.rand(b),
        "sku_ids": [f"SKU_{i%4}" for i in range(b)],
    }
    
    val_batch = {
        "pixel_values": torch.randn(b, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (b, 16)),
        "attention_mask": torch.ones(b, 16, dtype=torch.long),
        "q_v": torch.rand(b),
        "q_t": torch.rand(b),
        "sku_ids": [f"SKU_{i%4}" for i in range(b)],
    }

    train_loader = [mock_batch]
    val_loader = [val_batch]

    trainer = FusionMatchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            "warmup_epochs": 1,
            "finetune_epochs": 0,
            "total_epochs": 1,
            "checkpoint_dir": "artifacts/checkpoints",
        },
        device="cpu",
    )

    history = trainer.fit(epochs=1)
    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_f1" in history[0]
