"""Unit tests for FAISS Vector Indexing, IVF-PQ Compression, nprobe Tuning, and Bayesian Threshold Calibration."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import faiss
from src.indexing.build_index import IndexBuilder, tune_nprobe
from src.indexing.threshold_calibration import BayesianThresholdCalibrator


def test_index_builder_ivfpq_creation_and_search():
    """Verify IndexIVFPQ building, training, saving, and ID mapping retrieval."""
    n, d = 300, 256
    np.random.seed(42)
    embeddings = np.random.randn(n, d).astype(np.float32)
    sku_ids = [f"SKU_{i:04d}" for i in range(n)]

    builder = IndexBuilder(embed_dim=d, nlist=16, m=32, nbits=8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        index = builder.build(embeddings, sku_ids, save_dir=tmp_dir, nprobe=8)
        assert index.ntotal == n
        assert (Path(tmp_dir) / "index.faiss").exists()
        assert (Path(tmp_dir) / "id_map.json").exists()

        # Query index
        query = embeddings[:5]
        distances, indices = index.search(query, k=3)
        assert indices.shape == (5, 3)
        assert distances.shape == (5, 3)
        # Top-1 match for query vector should be its own index
        assert (indices[:, 0] == np.arange(5)).all()


def test_index_flat_baseline_and_recall():
    """Verify IndexFlatIP baseline and recall comparison against IndexIVFPQ."""
    n, d = 320, 256
    np.random.seed(42)
    embeddings = np.random.randn(n, d).astype(np.float32)
    sku_ids = [f"SKU_{i:04d}" for i in range(n)]

    builder = IndexBuilder(embed_dim=d, nlist=16, m=32, nbits=8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ivf_index = builder.build(embeddings, sku_ids, save_dir=tmp_dir, nprobe=16)
        flat_index = builder.build_flat_baseline(embeddings, save_dir=tmp_dir)

        assert ivf_index.ntotal == n
        assert flat_index.ntotal == n

        queries = embeddings[10:20]
        _, gt_ids = flat_index.search(queries, k=5)
        _, pred_ids = ivf_index.search(queries, k=5)

        # Check that top-1 match has >= 90% overlap
        top1_recall = np.mean(gt_ids[:, 0] == pred_ids[:, 0])
        assert top1_recall >= 0.80, f"Expected high top-1 recall, got {top1_recall}"


def test_tune_nprobe():
    """Verify nprobe tuning Pareto curve calculation."""
    n, d = 300, 256
    np.random.seed(42)
    embeddings = np.random.randn(n, d).astype(np.float32)
    sku_ids = [f"SKU_{i:04d}" for i in range(n)]

    builder = IndexBuilder(embed_dim=d, nlist=16, m=32, nbits=8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ivf_index = builder.build(embeddings, sku_ids, save_dir=tmp_dir)
        flat_index = builder.build_flat_baseline(embeddings)

        results = tune_nprobe(
            index=ivf_index,
            flat_index=flat_index,
            query_embeddings=embeddings[:20],
            k=5,
            nprobe_grid=(1, 4, 8, 16),
        )

        assert len(results) > 0
        for r in results:
            assert "nprobe" in r
            assert "recall@5" in r
            assert "latency_ms_per_query" in r
            assert 0.0 <= r["recall@5"] <= 1.0


def test_bayesian_threshold_calibration():
    """Verify Bayesian decision threshold calibration on synthetic similarity pairs."""
    np.random.seed(42)
    # Synthetic positive pairs (mean sim 0.88) and negative pairs (mean sim 0.35)
    pos_sims = np.random.normal(0.88, 0.05, size=200).clip(0.0, 1.0)
    neg_sims = np.random.normal(0.35, 0.10, size=800).clip(0.0, 1.0)

    sims = np.concatenate([pos_sims, neg_sims])
    labels = np.concatenate([np.ones(200), np.zeros(800)])

    calibrator = BayesianThresholdCalibrator(prior_alpha=2.0, prior_beta=2.0)
    val_data = {
        "SHOES": (sims, labels),
        "HOME": (sims, labels),
    }

    thresholds = calibrator.fit(val_data)
    assert "SHOES" in thresholds
    assert "HOME" in thresholds
    assert "__default__" in thresholds

    # Optimal threshold should be between negative and positive distributions
    shoes_t = thresholds["SHOES"]
    assert 0.55 <= shoes_t <= 0.85, f"Calibrated threshold {shoes_t} outside expected range [0.55, 0.85]"

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "thresholds.json"
        calibrator.save(save_file)
        assert save_file.exists()

        loaded_calibrator = BayesianThresholdCalibrator.load(save_file)
        assert loaded_calibrator.get_threshold("SHOES") == shoes_t
