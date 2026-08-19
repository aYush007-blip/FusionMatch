# Phase 2: Model Architecture & Dual-Encoder Development — Walkthrough & Verification Report

## Executive Summary

Phase 2 implements the complete model architecture for **FusionMatch**, the Cross-Modal & Multi-View Product Deduplication Engine, as defined in §6 Phase 2, §7, and §10.1 of the specification. The model combines:
1. A **SigLIP Dual-Tower Encoder** with multi-angle perspective pooling and gradient-isolation controls.
2. A **Quality-Aware Gated Fusion Module** dynamically balancing visual and textual representations via learnable softmax gates conditioned on heuristic quality proxy scalars.
3. A **Contrastive Projection Head** producing unit-norm 256-dimensional embeddings on the hypersphere for cosine similarity and high-speed FAISS indexing.

---

## 1. Model Architecture & Mathematical Formulations

```text
[Input Modalities]
  ├── Multi-View Images: (B, K, 3, H, W) ──► SigLIP Vision Tower ──► Multi-View Mean-Pool ──► v_pool (768-d)
  │                                                                       │
  │                                    Laplacian Blur Variance Proxy ────► q_v (scalar ∈ [0, 1])
  │                                                                       │
  └── Product Title:     (B, L)         ──► SigLIP Text Tower   ──► Mean Token Pooling   ──► t_pool (768-d)
                                                                          │
                                    Token Length Density Proxy   ────────► q_t (scalar ∈ [0, 1])
                                                                          │
                                                                          ▼
                                                    [Gated Fusion Network (src/models/gated_fusion.py)]
                                                       ├── Projections: W_v · v_pool, W_t · t_pool
                                                       ├── Gate Input:  [W_v · v, W_t · t, q_v, q_t]
                                                       ├── Softmax:     [g_v, g_t]  (g_v + g_t = 1.0)
                                                       └── Convex Comb: v_fused = g_v · W_v(v) + g_t · W_t(t)
                                                                          │
                                                                          ▼
                                                    [Projection Head (src/models/projection_head.py)]
                                                       ├── MLP: Linear(768, 512) ──► GELU ──► LayerNorm ──► Linear(512, 256)
                                                       └── L2 Normalization: z = v / ||v||_2
                                                                          │
                                                                          ▼
                                                    [Unit-Norm Product Embedding: z ∈ S^{255} (256-d)]
```

### Key Mathematical Equations

1. **Multi-View Angle Aggregation**:
   $$v_{\text{pool}} = \frac{1}{K} \sum_{k=1}^K \text{VisionEncoder}(I_k) \quad \in \mathbb{R}^{768}$$

2. **Quality Proxy Estimators**:
   $$q_v = 0.7 \cdot \min\left(\frac{\sigma^2_{\text{Laplacian}}}{500.0}, 1.0\right) + 0.3 \cdot \min\left(\frac{H \cdot W}{256^2}, 1.0\right) \quad \in [0, 1]$$
   $$q_t = \min\left(\frac{N_{\text{tokens}}}{20.0}, 1.0\right) \quad \in [0, 1]$$

3. **Quality-Aware Gated Fusion**:
   $$[g_v, g_t] = \text{Softmax}\left(\text{Linear}\left(\text{ReLU}\left(\text{Linear}\left([W_v v_{\text{pool}}, W_t t_{\text{pool}}, q_v, q_t]\right)\right)\right)\right)$$
   $$v_{\text{fused}} = g_v \cdot W_v v_{\text{pool}} + g_t \cdot W_t t_{\text{pool}} \quad \text{where } g_v + g_t = 1$$

4. **Hypersphere Projection & Unit Normalization**:
   $$z = \frac{\text{MLP}(v_{\text{fused}})}{\|\text{MLP}(v_{\text{fused}})\|_2} \quad \in \mathbb{R}^{256}, \quad \|z\|_2 = 1.0$$

---

## 2. Parameter Budget Breakdown

| Component | Total Parameters | Trainable in Phase 1 (Warm-Up) | Trainable in Phase 2 (Fine-Tuning) |
| :--- | :--- | :--- | :--- |
| **SigLIP Vision Tower** | ~93,000,000 | 0 (Frozen) | ~14,000,000 (Last 2 blocks unfrozen) |
| **SigLIP Text Tower** | ~110,000,000 | 0 (Frozen) | ~16,000,000 (Last 2 blocks unfrozen) |
| **Gated Fusion Network** | ~1,180,000 | 1,180,000 (100% Trainable) | 1,180,000 (100% Trainable) |
| **Projection Head (MLP)** | ~525,000 | 525,000 (100% Trainable) | 525,000 (100% Trainable) |
| **Total Model Footprint** | **~205,000,000** | **~1,705,000 (~0.83%)** | **~31,705,000 (~15.46%)** |

---

## 3. Core Modules Implemented

### 1. `src/models/siglip_encoder.py`
- **Class**: `SiglipDualEncoder(nn.Module)`
- Implements separate vision and text encoder towers wrapping Hugging Face's `google/siglip-base-patch16-224`.
- Multi-view visual perspective aggregation via mean-pooling across $K$ angle views: `(B, K, 3, H, W) -> (B, 768)`.
- Flexible parameter freezing (`freeze_vision`, `freeze_text`) and selective unfreezing of the last $N$ transformer blocks (`unfreeze_last_n_blocks`).

### 2. `src/models/quality_proxies.py`
- **Functions**: `image_quality_score()`, `text_quality_score()`, `compute_single_image_quality()`, `compute_single_text_quality()`.
- Differentiable-free (`@torch.no_grad()`) scoring providing stable, low-overhead scalar signals to the fusion gate.

### 3. `src/models/gated_fusion.py`
- **Class**: `GatedFusion(nn.Module)`
- Linear projection of vision and text towers into a shared 768-d representation.
- 2-layer MLP gating network producing softmax weights $[g_v, g_t]$ conditioned on multimodal features and quality signals.

### 4. `src/models/projection_head.py`
- **Class**: `ProjectionHead(nn.Module)`
- 2-layer contrastive projection MLP ($768 \to 512 \to 256$) with GELU activation and LayerNorm.
- Strict L2 hypersphere normalization ensuring unit vector length ($\|z\|_2 = 1.0$) for cosine similarity and FAISS inner-product index compatibility.

### 5. `src/models/fusion_match_model.py`
- **Class**: `FusionMatchModel(nn.Module)`
- Top-level composition module integrating encoder, multi-view pooling, quality scoring, gated fusion, and projection head.
- Exposes `forward()`, `encode_multimodal()`, `num_trainable_params()`, and `get_param_budget_summary()`.

---

## 4. Full Unit Test Suite Verification

Command: `.venv\Scripts\pytest.exe tests/ -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\HP\Desktop\Project\Deduplication\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\HP\Desktop\Project\Deduplication
plugins: anyio-4.14.2
collecting ... collected 14 items

tests/test_data_pipeline.py::test_sku_split_strictly_disjoint PASSED     [  7%]
tests/test_data_pipeline.py::test_real_manifest_files_no_leakage_if_present PASSED [ 14%]
tests/test_data_pipeline.py::test_image_augmenter PASSED                 [ 21%]
tests/test_data_pipeline.py::test_text_augmenter PASSED                  [ 28%]
tests/test_data_pipeline.py::test_quality_proxies PASSED                 [ 35%]
tests/test_data_pipeline.py::test_pair_sampler PASSED                    [ 42%]
tests/test_data_pipeline.py::test_dataset_and_dataloader PASSED          [ 50%]
tests/test_model_forward.py::test_forward_shapes PASSED                  [ 57%]
tests/test_model_forward.py::test_gate_softmax_normalization PASSED      [ 64%]
tests/test_model_forward.py::test_embedding_l2_unit_norm PASSED          [ 71%]
tests/test_model_forward.py::test_multi_view_pooling PASSED              [ 78%]
tests/test_model_forward.py::test_gradient_isolation_frozen_backbone PASSED [ 85%]
tests/test_model_forward.py::test_quality_proxies PASSED                 [ 92%]
tests/test_model_forward.py::test_param_budget_summary PASSED            [100%]

============================= 14 passed in 27.33s =============================
```

---

## 5. Next Steps: Phase 3 (Training & Loss Functions)

With the model architecture and data pipeline fully verified, we are ready for **Phase 3**:
1. **InfoNCE Loss with Hard Negative Mining (`src/training/losses.py`)**:
   - In-batch contrastive loss with learnable temperature $\tau$.
   - Mining in-category hard negatives.
2. **Two-Phase Trainer (`src/training/trainer.py`)**:
   - Epochs 1–5: Frozen backbone warm-up (LR $2\times 10^{-5}$ on fusion/projection).
   - Epochs 6–15: Partial fine-tuning of last 2 transformer blocks (discriminative LR $2\times 10^{-6}$).
3. **Evaluation Metrics (`src/training/metrics.py`)**:
   - Pairwise F1-score, Precision@K, Recall@K on validation split.
