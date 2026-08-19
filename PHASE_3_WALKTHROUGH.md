# Phase 3: Contrastive Training, Loss Functions & Evaluation Metrics — Walkthrough Report

## Executive Summary

Phase 3 establishes the complete **Training & Evaluation Engine** for **FusionMatch**, the Cross-Modal & Multi-View Product Deduplication Engine, as specified in §6 Phase 3, §8, and §10.1 of the design specification.

This phase implements:
1. **InfoNCE Contrastive Loss** with temperature scaling ($\tau = 0.07$) supporting symmetric in-batch negative pairs and mined hard negatives.
2. **HardNegativeMiner** maintaining a circular FIFO embedding bank with SKU-aware penalty masking to prevent false negatives.
3. **Two-Phase Trainer (`FusionMatchTrainer`)**:
   - **Phase 1 (Warm-Up, Epochs 1–5)**: Frozen SigLIP backbone, training only Gated Fusion + Projection Head with AdamW ($LR = 2 \times 10^{-5}$).
   - **Phase 2 (Fine-Tuning, Epochs 6–15)**: Partial fine-tuning of the last 2 transformer blocks ($LR = 2 \times 10^{-6}$) with discriminative learning rates.
   - Mixed precision (`torch.amp.autocast`), gradient clipping, and checkpoint management.
4. **Evaluation Metrics**: Pairwise F1-score on duplicate clusters, Precision@K, and Recall@K.

---

## 1. Mathematical Formulations

### 1.1 InfoNCE Contrastive Loss

Given a batch of $B$ positive pairs $(z_i^a, z_i^p)$ of L2-normalized embeddings ($z \in S^{255}$) and $K$ mined hard negative embeddings $z_{i, k}^n$:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(\frac{z_i^a \cdot z_i^p}{\tau}\right)}{\exp\left(\frac{z_i^a \cdot z_i^p}{\tau}\right) + \sum_{j \ne i} \exp\left(\frac{z_i^a \cdot z_j^p}{\tau}\right) + \sum_{k=1}^K \exp\left(\frac{z_i^a \cdot z_{i, k}^n}{\tau}\right)}$$

where:
- $\tau = 0.07$ is the temperature scaling factor.
- $\sum_{j \ne i}$ sums over all in-batch negative candidates from other products in the batch.
- $\sum_{k=1}^K$ sums over hard negative representations mined by the `HardNegativeMiner`.

### 1.2 Hard Negative Mining with False-Negative Masking

The `HardNegativeMiner` maintains an active embedding bank $\mathcal{B} = \{(z_m, \text{sku}_m)\}_{m=1}^M$ of capacity $M=4096$. For anchor SKU $s_i$, the retrieval scores are masked as:

$$S(z_i^a, z_m) = (z_i^a \cdot z_m) - 10^4 \cdot \mathbb{I}(\text{sku}_m = s_i)$$

$$\mathcal{N}_{\text{hard}}(i) = \text{TopK}_{m \in \mathcal{B}}\left(S(z_i^a, z_m), K=4\right)$$

This ensures that items belonging to the same SKU are never selected as negative samples.

### 1.3 Evaluation Metrics

1. **Pairwise F1-Score**:
   $$\text{Pairwise F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Evaluated over all $\binom{N}{2}$ item pairs where $\hat{y}_{ij} = \mathbb{I}(\text{CosineSim}(z_i, z_j) \ge \tau_{\text{threshold}})$.

2. **Precision@K & Recall@K**:
   $$\text{Precision@K}(i) = \frac{\# \text{ true duplicates in Top-}K}{K}$$
   $$\text{Recall@K}(i) = \frac{\# \text{ true duplicates in Top-}K}{\min(\text{total available duplicates}, K)}$$

---

## 2. Core Modules Implemented

### 1. `src/training/losses.py`
- **`InfoNCELoss(nn.Module)`**: Contrastive cross-entropy loss supporting in-batch negatives, hard negatives, and symmetric loss options.
- **`HardNegativeMiner`**: FIFO queue buffer maintaining up to 4096 vectors with instant same-SKU masking.

### 2. `src/training/metrics.py`
- **`compute_pairwise_f1()`**: Efficient vectorized upper-triangle pairwise cosine similarity computation and F1 evaluation.
- **`compute_precision_recall_at_k()`**: Top-K retrieval precision and recall with self-match exclusion.
- **`evaluate_embeddings()`**: Summary dictionary returning metrics across multiple $K \in \{1, 5, 10\}$.

### 3. `src/training/trainer.py`
- **`FusionMatchTrainer`**: Complete two-phase training loop supporting:
  - Phase 1 Warm-up (epochs 1–5, frozen backbone).
  - Phase 2 Fine-tune (epochs 6–15, discriminative learning rates).
  - Automatic mixed precision (`torch.amp.autocast`), gradient scaling (`torch.amp.GradScaler`), and max-norm gradient clipping.
  - Validation metrics calculation after each epoch.
  - Automatic checkpoint persistence saving `artifacts/checkpoints/best.pt`.

### 4. `src/utils/`
- **`src/utils/seed.py`**: `seed_everything(seed=42)` ensuring deterministic behavior across Python, NumPy, and PyTorch.
- **`src/utils/io.py`**: `save_checkpoint()` and `load_checkpoint()` helpers.

### 5. `notebooks/03_training.ipynb`
- Interactive Jupyter notebook orchestrating data loading, two-phase training, validation curves, and checkpoint evaluation.

---

## 3. Full Unit Test Suite Verification

Command: `.venv\Scripts\pytest.exe tests/ -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\HP\Desktop\Project\Deduplication\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\HP\Desktop\Project\Deduplication
plugins: anyio-4.14.2
collected 20 items

tests/test_data_pipeline.py::test_sku_split_strictly_disjoint PASSED     [  5%]
tests/test_data_pipeline.py::test_real_manifest_files_no_leakage_if_present PASSED [ 10%]
tests/test_data_pipeline.py::test_image_augmenter PASSED                 [ 15%]
tests/test_data_pipeline.py::test_text_augmenter PASSED                  [ 20%]
tests/test_data_pipeline.py::test_quality_proxies PASSED                 [ 25%]
tests/test_data_pipeline.py::test_pair_sampler PASSED                    [ 30%]
tests/test_data_pipeline.py::test_dataset_and_dataloader PASSED          [ 35%]
tests/test_losses.py::test_infonce_loss_forward_and_backward PASSED      [ 40%]
tests/test_losses.py::test_infonce_with_hard_negatives PASSED            [ 45%]
tests/test_losses.py::test_hard_negative_miner_masking PASSED            [ 50%]
tests/test_losses.py::test_pairwise_f1_metric PASSED                     [ 55%]
tests/test_losses.py::test_precision_recall_at_k PASSED                  [ 60%]
tests/test_losses.py::test_trainer_smoke_run PASSED                      [ 65%]
tests/test_model_forward.py::test_forward_shapes PASSED                  [ 70%]
tests/test_model_forward.py::test_gate_softmax_normalization PASSED      [ 75%]
tests/test_model_forward.py::test_embedding_l2_unit_norm PASSED          [ 80%]
tests/test_model_forward.py::test_multi_view_pooling PASSED              [ 85%]
tests/test_model_forward.py::test_gradient_isolation_frozen_backbone PASSED [ 90%]
tests/test_model_forward.py::test_quality_proxies PASSED                 [ 95%]
tests/test_model_forward.py::test_param_budget_summary PASSED            [100%]

============================= 20 passed in 15.46s =============================
```

---

## 4. Next Steps: Phase 4 (Vector Indexing & Retrieval Optimization)

With the training pipeline and loss functions verified, we are ready for **Phase 4**:
1. **FAISS Indexing Engine (`src/indexing/build_index.py`)**:
   - Building compressed `IndexIVFPQ` ($N_{\text{list}}=400, M=32, N_{\text{bits}}=8$) yielding $\sim 1.2$ MB index for 10,000 SKUs.
   - Brute-force `IndexFlatIP` validation baseline.
2. **`nprobe` Tuning & Latency Benchmark**:
   - Trade-off curve measuring Recall@K vs. search latency.
3. **Bayesian Threshold Calibration (`src/indexing/threshold_calibration.py`)**:
   - Per-category duplicate cutoff threshold optimization via Beta-Binomial posterior modeling.
