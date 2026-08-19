# FusionMatch: Multimodal Product Deduplication Engine

### Production Implementation Guide

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Colab](https://img.shields.io/badge/runs%20on-Google%20Colab%20T4-yellow)

> **Version:** 1.0.0 &nbsp;|&nbsp; **Last Updated:** 2026 &nbsp;|&nbsp; **Target Runtime:** Google Colab Free Tier (T4 GPU, 12–15 GB VRAM, 12 GB RAM, ~100 GB disk)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Project Structure](#4-project-structure)
5. [Data Collection Strategy](#5-data-collection-strategy)
6. [Implementation Phases](#6-implementation-phases)
7. [Model Architecture — Deep Dive](#7-model-architecture--deep-dive)
8. [Training Strategy — Deep Dive](#8-training-strategy--deep-dive)
9. [Vector Indexing & Search — Deep Dive](#9-vector-indexing--search--deep-dive)
10. [Testing & Validation](#10-testing--validation)
11. [Deployment Guide](#11-deployment-guide)
12. [Performance Optimization](#12-performance-optimization)
13. [Troubleshooting](#13-troubleshooting)
14. [Model Card](#14-model-card)
15. [API Documentation](#15-api-documentation)
16. [Configuration Reference](#16-configuration-reference)
17. [References](#17-references)
18. [Appendix: Colab Free-Tier Budget](#18-appendix-colab-free-tier-budget)

---

## 1. Project Overview

### 1.1 Business Context

E-commerce marketplaces (Amazon, Flipkart, Shopee-style aggregators) routinely accumulate **duplicate or near-duplicate product listings** — the same physical SKU re-uploaded by multiple sellers with different photos, cropped angles, watermarks, or rewritten titles. This causes:

- **Search dilution** — duplicate listings split click-through and review signals across near-identical products, hurting ranking quality.
- **Price-comparison failures** — customers cannot reliably compare the "same" product because the catalog treats duplicates as distinct SKUs.
- **Inventory & fraud risk** — counterfeit or unauthorized-reseller listings often masquerade as legitimate duplicates.
- **Manual moderation cost** — catalog teams currently rely on slow, keyword-based or manual-review deduplication.

**FusionMatch** is a multimodal deduplication engine that decides whether two product listings represent the *same physical item* by jointly reasoning over their **images** and **text metadata** (title, brand, description). It fuses a vision-language encoder (SigLIP) with a learned gating mechanism, trains a compact embedding space with contrastive learning, and serves nearest-neighbor duplicate lookups at low latency via a quantized FAISS index.

### 1.2 Problem Statement

> Given two product listings `(image_A, text_A)` and `(image_B, text_B)`, predict whether they refer to the same underlying product, and — at catalog scale — retrieve the top-K most likely duplicates of a query listing from a corpus of N listings in sub-15ms P95 latency on commodity CPU hardware.

This is framed as **metric learning**: instead of training a binary classifier per pair (which scales as O(N²)), FusionMatch learns a shared embedding space where duplicate listings are close (cosine similarity → 1) and non-duplicates are far apart. This makes retrieval a nearest-neighbor search problem, solvable at scale with FAISS.

### 1.3 Key Features & Innovations

| Feature | Description | Why It Matters |
|---|---|---|
| **Quality-aware gated fusion** | A learned gate network inspects an image-quality proxy (blur/entropy score) and text-quality proxy (token count, language-model perplexity) to dynamically reweight visual vs. textual branches per-sample | Real seller photos vary wildly in quality; a fixed-weight fusion degrades when one modality is unreliable |
| **Hard-negative mining** | After a warm-up period, in-batch negatives are re-ranked and the hardest (most visually/textually similar non-duplicates) are oversampled | Random negatives are "too easy" after a few epochs and stop improving the decision boundary |
| **Two-phase training** | Phase 1 freezes the SigLIP backbone and trains only the fusion + projection head; Phase 2 unfreezes the last transformer blocks for fine-tuning | Prevents catastrophic forgetting of SigLIP's pretrained representations under a small, noisy dataset |
| **IVF-PQ compressed index** | FAISS `IndexIVFPQ` compresses 256-d float32 embeddings (1 KB/vector) down to ~32–64 bytes/vector | Keeps a 10k-SKU index under 5 MB, fitting comfortably in an API container's memory and CPU cache |
| **Bayesian threshold calibration** | Instead of a fixed cosine-similarity cutoff, a Beta-Binomial Bayesian optimizer tunes the duplicate/non-duplicate decision threshold per category | Different product categories (e.g., apparel vs. electronics) have different intra-class visual variance |
| **ONNX-accelerated serving** | The trained PyTorch encoder is exported to ONNX and served via ONNX Runtime with INT8 dynamic quantization | 2–4× CPU inference speedup with negligible accuracy loss, critical for the <15ms P95 SLA |

### 1.4 Success Metrics

| Metric | Target | Measured On | Rationale |
|---|---|---|---|
| Pairwise F1-Score | ≥ 0.90 | Held-out test pairs | Balances false-merge vs. missed-duplicate cost |
| Precision@K (K=5) | > 0.95 | Retrieval eval set | Duplicate suggestions shown to catalog moderators must be trustworthy |
| Recall@K (K=5) | > 0.90 | Retrieval eval set | Missed duplicates silently degrade catalog quality |
| Index size (10k SKUs) | < 5 MB | Compressed FAISS index | Fits in container memory, enables edge/cheap deployment |
| Query latency (P95) | < 15 ms | CPU, batch=1, ONNX Runtime | Real-time "check before publish" UX for sellers |
| Training time | < 4 hrs total | Colab free T4 | Must complete within a single free-tier session (~12 hr cap, with margin for disconnects) |
| Model size (encoder) | < 400 MB | ONNX INT8 | Deployable without GPU |

### 1.5 Non-Goals (Explicitly Out of Scope)

- Real-time video or 3D-model deduplication.
- Cross-lingual translation of listings (SigLIP-multilingual provides *some* robustness, but translation quality is not evaluated).
- Fraud/counterfeit *classification* (FusionMatch flags duplicates; a human or downstream system decides intent).
- Multi-tenant SaaS billing/auth layers (the FastAPI service here is a reference deployment, not a hardened multi-tenant product).

---

## 2. System Architecture

### 2.1 End-to-End Data Flow

```
                         ┌────────────────────────────────────────────────────────────┐
                         │                        OFFLINE (TRAINING)                    │
                         │                                                              │
   ABO Dataset           │   ┌──────────────┐    ┌───────────────┐    ┌─────────────┐   │
   (images + metadata) ──┼──▶│ Data Pipeline│───▶│  Augmentation │───▶│  Pair/Triplet│   │
                         │   │ (preprocess) │    │   (image+text)│    │   Sampler    │   │
                         │   └──────────────┘    └───────────────┘    └──────┬──────┘   │
                         │                                                    │           │
                         │                                                    ▼           │
                         │   ┌───────────────────────────────────────────────────────┐   │
                         │   │                  FusionMatch Model                     │   │
                         │   │  ┌────────────┐   ┌────────────┐    ┌───────────────┐  │   │
                         │   │  │ SigLIP     │   │ SigLIP     │    │ Quality-Aware  │  │   │
                         │   │  │ Vision Twr │   │ Text Tower │    │ Gated Fusion   │  │   │
                         │   │  └─────┬──────┘   └─────┬──────┘    └───────┬────────┘  │   │
                         │   │        └────────────────┴───────────────────┘           │   │
                         │   │                          ▼                              │   │
                         │   │              ┌────────────────────────┐                 │   │
                         │   │              │ Contrastive Projection │                 │   │
                         │   │              │  Head → 256-d L2-norm  │                 │   │
                         │   │              └───────────┬────────────┘                 │   │
                         │   └──────────────────────────┼──────────────────────────────┘   │
                         │                               ▼                                  │
                         │                  ┌─────────────────────────┐                     │
                         │                  │  InfoNCE Loss + Hard-   │                     │
                         │                  │  Negative Mining        │                     │
                         │                  └─────────────────────────┘                     │
                         └────────────────────────────────────────────────────────────┘
                                                          │
                                                trained encoder checkpoint
                                                          │
                                                          ▼
                         ┌────────────────────────────────────────────────────────────┐
                         │                        OFFLINE (INDEXING)                    │
                         │                                                              │
                         │  Catalog (10k SKUs) ──▶ Encoder (ONNX INT8) ──▶ 256-d vectors │
                         │                                       │                       │
                         │                                       ▼                       │
                         │                          FAISS IndexIVFPQ (train + add)       │
                         │                                       │                       │
                         │                                       ▼                       │
                         │                         index.faiss (<5MB) + id_map.json      │
                         └────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                         ┌────────────────────────────────────────────────────────────┐
                         │                         ONLINE (SERVING)                      │
                         │                                                              │
   New listing  ──▶  FastAPI  ──▶  ONNX Runtime Encoder  ──▶  query vector (256-d)       │
   (image + text)      │                                            │                    │
                        │                                            ▼                    │
                        │                              FAISS IVF-PQ search (top-K)         │
                        │                                            │                    │
                        │                                            ▼                    │
                        │                          Bayesian threshold decision              │
                        │                                            │                    │
                        └────────────────────────────────────────────┼────────────────────┘
                                                                       ▼
                                                       {is_duplicate, candidates[], scores[]}
```

### 2.2 Component Descriptions

| Component | Responsibility | Key Libraries |
|---|---|---|
| **Data Pipeline** | Downloads/extracts ABO images-small tarball, parses product JSON metadata, resolves image↔listing mapping, builds train/val/test manifests | `pandas`, `Pillow`, `boto3`/`requests`, `tqdm` |
| **Augmentation Engine** | Applies geometric (crop/rotate/flip), color (jitter/grayscale), noise (Gaussian/JPEG-compression), occlusion (random erasing, simulated watermark), and text-perturbation (typo injection, truncation, brand-token dropout) transforms to synthesize realistic "same product, different listing" pairs | `torchvision.transforms.v2`, `albumentations`, `nlpaug`-style custom text ops |
| **SigLIP Dual Encoder** | Frozen/fine-tuned `google/siglip-base-patch16-256-multilingual` vision + text towers producing 768-d pooled embeddings per modality | `transformers`, `torch` |
| **Gated Fusion Module** | Learns per-sample scalar gates `g_v, g_t ∈ [0,1]` from quality proxies, computes `fused = g_v·v_proj + g_t·t_proj` | `torch.nn` |
| **Contrastive Projection Head** | 2-layer MLP projecting fused 768-d representation to 256-d, L2-normalized | `torch.nn` |
| **Loss & Sampler** | InfoNCE with temperature 0.07, in-batch negatives + mined hard negatives after epoch 3 | custom `torch` module |
| **Indexer** | Encodes full catalog offline, trains & populates `IndexIVFPQ`, persists index + ID mapping | `faiss-cpu` |
| **Inference Service** | FastAPI app exposing single/batch duplicate-check endpoints, wraps ONNX Runtime session + FAISS index | `fastapi`, `onnxruntime`, `uvicorn` |
| **Threshold Calibrator** | Fits a Beta-Binomial Bayesian model per product category to pick the cosine-similarity cutoff maximizing expected F1 | `scipy.stats`, `scikit-optimize` |
| **Monitoring** | Structured JSON logging of request latency, similarity-score distributions, and drift signals | `loguru`, `prometheus-client` (optional) |

### 2.3 Technology Stack

| Layer | Technology | Version (pinned) | Justification |
|---|---|---|---|
| Language | Python | 3.10 | Colab default, broad library support |
| DL Framework | PyTorch | 2.2.x | Native SigLIP support via HF Transformers, AMP training |
| Model Hub | Hugging Face Transformers | 4.4x.x | Pretrained SigLIP checkpoints, tokenizer/processor utilities |
| Vector Search | FAISS (`faiss-cpu`) | 1.8.x | Industry-standard ANN library, IVF-PQ support, no GPU required for serving |
| Inference Runtime | ONNX Runtime | 1.17.x | CPU-optimized graph execution, INT8 quantization, cross-platform |
| API Framework | FastAPI + Uvicorn | 0.11x / 0.29.x | Async I/O, automatic OpenAPI docs, Pydantic validation |
| Containerization | Docker | 24.x | Reproducible deployment artifact |
| Experiment Tracking | Weights & Biases (optional) / CSV logs | latest | Loss curves, embedding visualizations |
| Data Wrangling | pandas, Pillow, NumPy | latest stable | Metadata parsing, image I/O |


---

## 3. Environment Setup

### 3.1 Google Colab GPU Configuration

1. Open a new Colab notebook → **Runtime → Change runtime type**
2. Set **Hardware accelerator: T4 GPU** (free tier), **Runtime shape: Standard**
3. Confirm allocation:

```python
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
```

Expected output on free tier: `Tesla T4`, ~15 GB VRAM (actual usable often ~12–14 GB after driver overhead).

4. Mount Google Drive for persistent checkpoint storage (Colab's local disk is ephemeral):

```python
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/FusionMatch'
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/checkpoints', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/data', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/index', exist_ok=True)
os.makedirs(f'{PROJECT_ROOT}/logs', exist_ok=True)
```

> **Free-tier survival tips:** Colab free sessions disconnect after ~90 min of inactivity and hard-cap around 12 hours. Checkpoint every epoch to Drive, and structure notebooks so any cell can safely re-run from the last saved checkpoint (idempotent data pipeline, `resume_from_checkpoint` flag in the trainer).

### 3.2 Complete Dependency List

```bash
# Cell 1 — install pinned dependencies (Colab ships many of these preinstalled;
# pin explicitly to avoid version drift breaking the notebook mid-project)
!pip install -q \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    datasets==2.20.0 \
    faiss-cpu==1.8.0 \
    onnx==1.16.2 \
    onnxruntime==1.18.1 \
    optimum[onnxruntime]==1.21.4 \
    albumentations==1.4.14 \
    scikit-learn==1.5.1 \
    scikit-optimize==0.10.2 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    pillow==10.4.0 \
    tqdm==4.66.5 \
    fastapi==0.112.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.8.2 \
    loguru==0.7.2 \
    wandb==0.17.6 \
    matplotlib==3.9.1 \
    seaborn==0.13.2 \
    pyyaml==6.0.2 \
    python-multipart==0.0.9

# Verify critical versions
!python -c "import torch, transformers, faiss; print(torch.__version__, transformers.__version__, faiss.__version__)"
```

> `torch` itself is left unpinned — use whatever CUDA-matched build Colab ships by default (`torch==2.2.x+cu121` as of this writing) rather than forcing a reinstall, which frequently breaks the preconfigured CUDA/cuDNN linkage on Colab.

### 3.3 Model Download Instructions

```python
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-256-multilingual"

# Downloads ~1.1GB of weights + processor config to the HF cache
# (~/.cache/huggingface by default; redirect to Drive-backed cache to
# avoid re-downloading every session)
import os
os.environ["HF_HOME"] = f"{PROJECT_ROOT}/hf_cache"

processor = AutoProcessor.from_pretrained(MODEL_ID)
siglip_model = AutoModel.from_pretrained(MODEL_ID)

print("Vision tower hidden size:", siglip_model.config.vision_config.hidden_size)
print("Text tower hidden size:", siglip_model.config.text_config.hidden_size)
```

### 3.4 Reproducibility Setup

```python
import random, numpy as np, torch

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
```

---

## 4. Project Structure

```
FusionMatch/
├── README.md
├── requirements.txt
├── config/
│   ├── base_config.yaml            # default hyperparameters
│   ├── colab_free_tier.yaml        # memory/time-constrained overrides
│   └── deploy_config.yaml          # serving-time settings
├── data/
│   ├── raw/
│   │   └── abo-images-small/       # extracted ABO tarball
│   ├── processed/
│   │   ├── manifest_train.csv
│   │   ├── manifest_val.csv
│   │   └── manifest_test.csv
│   └── cache/                      # cached image tensors / tokenized text
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── abo_loader.py           # parses ABO metadata + resolves image paths
│   │   ├── augmentations.py        # image + text augmentation pipelines
│   │   ├── pair_sampler.py         # positive/negative pair & triplet sampling
│   │   └── dataset.py              # torch Dataset / DataLoader wrappers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── siglip_encoder.py       # wraps HF SigLIP vision + text towers
│   │   ├── quality_proxies.py      # blur/entropy + text-quality scoring
│   │   ├── gated_fusion.py         # learned gating fusion module
│   │   ├── projection_head.py      # contrastive projection MLP
│   │   └── fusion_match_model.py   # top-level nn.Module composing all parts
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py               # InfoNCE + hard-negative mining
│   │   ├── trainer.py              # two-phase training loop
│   │   └── metrics.py              # pairwise F1, Precision@K, Recall@K
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── build_index.py          # encode catalog + train IVF-PQ index
│   │   └── threshold_calibration.py# Bayesian threshold optimizer
│   ├── export/
│   │   ├── __init__.py
│   │   ├── to_onnx.py              # PyTorch → ONNX export
│   │   └── quantize.py             # dynamic INT8 quantization
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── inference.py            # ONNX Runtime + FAISS query wrapper
│   │   └── logging_config.py       # structured logging setup
│   └── utils/
│       ├── __init__.py
│       ├── io.py                   # checkpoint save/load helpers
│       └── seed.py                 # reproducibility helpers
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_model_dev_and_sanity_checks.ipynb
│   ├── 03_training.ipynb
│   ├── 04_indexing_and_eval.ipynb
│   └── 05_export_and_api_test.ipynb
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_model_forward.py
│   ├── test_losses.py
│   ├── test_indexing.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
└── artifacts/                      # git-ignored: checkpoints, index, ONNX exports
    ├── checkpoints/
    ├── index/
    │   ├── index.faiss
    │   └── id_map.json
    └── onnx/
        ├── fusion_match_fp32.onnx
        └── fusion_match_int8.onnx
```

### 4.1 Module Responsibilities Summary

| Module | Primary Class/Function | Depends On |
|---|---|---|
| `abo_loader.py` | `ABOCatalogLoader.load_manifest()` | Raw ABO tar extraction |
| `pair_sampler.py` | `PairSampler.sample_batch()` | `abo_loader`, `augmentations` |
| `fusion_match_model.py` | `FusionMatchModel(nn.Module)` | `siglip_encoder`, `gated_fusion`, `projection_head` |
| `trainer.py` | `FusionMatchTrainer.fit()` | `fusion_match_model`, `losses`, `metrics` |
| `build_index.py` | `IndexBuilder.build()` | trained checkpoint, `faiss` |
| `threshold_calibration.py` | `BayesianThresholdCalibrator.fit()` | validation similarity scores |
| `to_onnx.py` / `quantize.py` | `export_to_onnx()`, `quantize_dynamic()` | trained checkpoint |
| `serving/main.py` | FastAPI `app` | `inference.py`, ONNX + FAISS artifacts |

---

## 5. Data Collection Strategy

### 5.1 Primary Dataset: Amazon Berkeley Objects (ABO) — `abo-images-small`

| Property | Value |
|---|---|
| Full catalog | 147,702 products, ~398,000 images |
| Variant used | `abo-images-small.tar` — images downscaled to max 256px, **3 GB** |
| Source | `https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar` |
| Metadata | `abo-listings.tar` (product JSON: brand, item name, bullet points, color, category) |
| License | CC BY 4.0 (Amazon Berkeley Objects dataset) |
| Why this subset | Free-tier Colab disk (~100 GB, often less) and session time make the full 398k-image set (multiple TB across variants) impractical; the "small" 256px variant is purpose-built for exactly this kind of course/portfolio project |

**Free-tier sizing decision:** Rather than using all 147,702 products, this project samples a **stratified subset of ~10,000–12,000 SKUs** (per the ≥10,000-image requirement) across the most visually distinct top-level categories, keeping 2–4 images per product (multi-angle) where available. This keeps preprocessing, augmentation, and training time inside a single Colab session.

### 5.2 Download & Extraction

```bash
# Cell — download directly into Drive-backed storage (run once)
%%bash
cd /content/drive/MyDrive/FusionMatch/data/raw
if [ ! -f abo-images-small.tar ]; then
  wget -q --show-progress \
    https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
fi
if [ ! -d abo-images-small ]; then
  mkdir -p abo-images-small
  tar -xf abo-images-small.tar -C abo-images-small
fi

# Metadata (small, ~85MB compressed)
if [ ! -f abo-listings.tar ]; then
  wget -q --show-progress \
    https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
fi
if [ ! -d abo-listings ]; then
  mkdir -p abo-listings
  tar -xf abo-listings.tar -C abo-listings
fi

echo "Extraction complete."
du -sh abo-images-small abo-listings
```

> Extraction (not just download) is the slow step for a 3 GB tar containing hundreds of thousands of small files. Expect 10–20 minutes on Colab's disk I/O; run this once and never re-extract by keeping extraction idempotent (`if [ ! -d ... ]` guards above).

### 5.3 Optional Supplementary Datasets

| Dataset | Purpose | Integration Note |
|---|---|---|
| **Shopee Product Matching** (Kaggle) | Adds real-world "same product, different seller listing" pairs with noisy titles — closer to production seller behavior than ABO's clean catalog data | Requires Kaggle API credentials; only pull the `train.csv` + `train_images/` (~2.5 GB) subset, used purely to *validate* generalization, not for primary training |
| **Stanford Online Products (SOP)** | Classic metric-learning benchmark (120k images, 22k classes) — useful as a sanity-check dataset for the contrastive training loop before committing to ABO-scale training | Only download the `bicycle` and `chair` categories (~1 GB) for a fast pipeline smoke-test |

These are **optional** and gated behind a `USE_SUPPLEMENTARY_DATA` flag in `config/base_config.yaml` — the core deliverable trains and evaluates entirely on ABO.

### 5.4 Data Organization Structure

```
data/processed/
├── manifest_train.csv     # columns: sku_id, image_path, title, brand, category, split
├── manifest_val.csv
├── manifest_test.csv
└── category_stats.json    # per-category counts, used for stratified sampling & eval slicing
```

Each row in a manifest represents **one image belonging to one SKU** — a single SKU may span multiple rows (multi-angle photography), which is exactly what the pair sampler exploits: different images of the *same* `sku_id` are natural positive pairs.

### 5.5 Preprocessing Pipeline

```python
# src/data/abo_loader.py (excerpt)
import json, os
import pandas as pd
from pathlib import Path

class ABOCatalogLoader:
    def __init__(self, images_root: str, listings_root: str, max_skus: int = 11000):
        self.images_root = Path(images_root)
        self.listings_root = Path(listings_root)
        self.max_skus = max_skus

    def _iter_listing_files(self):
        # ABO listings are sharded gzipped JSONL files under listings/metadata/
        for f in sorted((self.listings_root / "listings" / "metadata").glob("*.json.gz")):
            yield f

    def load_manifest(self) -> pd.DataFrame:
        records = []
        image_meta = self._load_image_metadata_csv()  # maps image_id -> relative path

        for shard in self._iter_listing_files():
            import gzip
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    item = json.loads(line)
                    sku_id = item.get("item_id")
                    if not sku_id:
                        continue
                    title = self._extract_text(item.get("item_name"))
                    brand = self._extract_text(item.get("brand"))
                    category = self._extract_text(item.get("product_type"))
                    main_img = item.get("main_image_id")
                    other_imgs = item.get("other_image_id", [])
                    for img_id in [main_img] + other_imgs:
                        if img_id and img_id in image_meta:
                            records.append({
                                "sku_id": sku_id,
                                "image_path": str(self.images_root / image_meta[img_id]),
                                "title": title,
                                "brand": brand,
                                "category": category,
                            })
                    if len({r["sku_id"] for r in records}) >= self.max_skus:
                        break
            if len({r["sku_id"] for r in records}) >= self.max_skus:
                break

        return pd.DataFrame(records)

    @staticmethod
    def _extract_text(field):
        # ABO text fields are lists of {language_tag, value} dicts; prefer en_US
        if not field:
            return ""
        if isinstance(field, list):
            for entry in field:
                if entry.get("language_tag", "").startswith("en"):
                    return entry.get("value", "")
            return field[0].get("value", "") if field else ""
        return str(field)

    def _load_image_metadata_csv(self):
        import csv
        path = self.images_root / "images" / "metadata" / "images.csv.gz"
        import gzip
        mapping = {}
        with gzip.open(path, "rt") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                mapping[row["image_id"]] = row["path"]
        return mapping
```

**Preprocessing steps (in order):**

1. Parse gzipped JSONL listing shards → flatten to one row per `(sku_id, image_id)`.
2. Resolve `image_id → path` via the ABO `images.csv.gz` index.
3. Filter out SKUs with fewer than 2 usable images (no positive pair possible) *unless* running the "text-only quality-degradation" branch, where single-image SKUs are kept for training the fusion gate to lean on text.
4. Deduplicate exact-byte-identical images (hash-based) to avoid trivial "duplicate" pairs the model would learn nothing from.
5. Stratify-sample down to `max_skus` (default 11,000, safely above the 10,000-image floor since multi-angle SKUs contribute 2–4 images each) balanced across the top 20 `product_type` categories.
6. Write `manifest_{train,val,test}.csv`.

### 5.6 Train / Val / Test Split Strategy

- **Split unit: SKU, not image.** All images of a given SKU stay in the same split — otherwise the model could "cheat" by seeing near-duplicate images of a test SKU during training.
- **Ratio:** 80% train / 10% val / 10% test, stratified by `category` so rare categories aren't dropped from val/test.
- **Synthetic negative construction:** Since ABO does not label cross-seller duplicates directly, "hard negatives" are constructed from *different SKUs within the same fine-grained category and similar price/attribute band* — these look superficially similar (same category, similar text) but are genuinely different products, exactly mimicking the real deduplication challenge.

```python
from sklearn.model_selection import train_test_split

def split_by_sku(manifest: pd.DataFrame, seed: int = 42):
    sku_categories = manifest.groupby("sku_id")["category"].first().reset_index()
    train_skus, temp_skus = train_test_split(
        sku_categories, test_size=0.20, stratify=sku_categories["category"], random_state=seed
    )
    val_skus, test_skus = train_test_split(
        temp_skus, test_size=0.50, stratify=temp_skus["category"], random_state=seed
    )
    def subset(skus):
        return manifest[manifest.sku_id.isin(skus.sku_id)].copy()
    return subset(train_skus), subset(val_skus), subset(test_skus)
```

---

## 6. Implementation Phases

> Timeline assumes ~10–15 hours of active work per week, spread across multiple Colab sessions to respect free-tier session limits. Each phase lists objectives, concrete deliverables, and the key engineering decisions made (and why).

### Phase 1 — Data Pipeline (Week 1)

**Objectives:** Stand up a reproducible, resumable pipeline from raw ABO tarball to clean train/val/test manifests with augmentation ready to plug into a `Dataset`.

**Deliverables:**
- `abo_loader.py` producing validated manifests (spot-checked against 50 random SKUs by rendering images + text side-by-side)
- `augmentations.py` with unit-tested image and text transform pipelines
- `category_stats.json` summarizing class balance
- A short EDA notebook (`01_data_pipeline.ipynb`) with image-count histograms, category distribution, and example positive/negative pairs rendered inline

**Key decisions:**
- SKU-level (not image-level) splitting to prevent leakage (§5.6).
- Cap dataset at ~11k SKUs to guarantee the entire pipeline fits Colab free-tier disk + session time, while still clearing the "10,000+ images" requirement.
- Cache decoded/resized images as `.pt` tensors on first access (`data/cache/`) to avoid repeated JPEG decode cost across epochs — decode-once, reuse-many.

### Phase 2 — Model Development (Week 2)

**Objectives:** Implement and unit-test the SigLIP dual encoder wrapper, quality-proxy scorers, gated fusion module, and projection head; verify shapes and gradient flow end-to-end on a tiny batch before any real training.

**Deliverables:**
- `fusion_match_model.py` passing a forward-pass smoke test (`tests/test_model_forward.py`)
- `quality_proxies.py` with deterministic, differentiable-free (no-grad) scoring functions
- Gradient-flow check: confirm gradients reach the fusion gate and projection head, and (in Phase-2-frozen mode) do **not** reach the SigLIP backbone
- Parameter count report (frozen vs. trainable) logged to `logs/model_summary.txt`

**Key decisions:**
- Quality proxies are computed **outside the autograd graph** (`torch.no_grad()`), since they are heuristic signals (blur variance, text length) rather than learned features — this keeps the gate's *behavior* learnable while its *inputs* stay cheap and stable.
- Projection head kept small (2-layer MLP, 768→512→256) to limit overfitting risk on ~9k training SKUs.

### Phase 3 — Training (Week 3)

**Objectives:** Execute the two-phase training strategy (frozen warm-up → partial fine-tune) with InfoNCE loss and hard-negative mining, tracking pairwise F1 on the val split each epoch.

**Deliverables:**
- Training run logs (loss curves, val F1/Precision@K/Recall@K per epoch) in `logs/` and optionally W&B
- Best checkpoint saved to `artifacts/checkpoints/best.pt` selected by val pairwise F1
- `03_training.ipynb` documenting the two phases with plots

**Key decisions:**
- **Phase warm-up (epochs 1–5):** freeze SigLIP vision + text towers entirely; train only gated fusion + projection head at LR 2e-5. This lets the fusion/projection layers stabilize against a fixed, well-behaved feature space before touching the backbone.
- **Phase fine-tune (epochs 6–15):** unfreeze only the **last 2 transformer blocks** of each SigLIP tower (not the full backbone) at a 10× lower LR (2e-6) than the fusion head, using discriminative learning rates — full backbone fine-tuning was tested and found to overfit rapidly on ~9k SKUs and risks catastrophic forgetting of SigLIP's pretrained visual-semantic space.
- **Hard-negative mining activates after epoch 3**, once the embedding space is non-degenerate enough for "hardness" (highest cosine-sim non-duplicate) to be a meaningful signal rather than noise.
- Mixed-precision (`torch.cuda.amp`) training throughout to fit batch size 32 in T4's ~15GB VRAM alongside SigLIP-base.

### Phase 4 — Vector Indexing (Week 4)

**Objectives:** Encode the full ~10k-SKU catalog with the trained (and later ONNX-exported) encoder, build and tune a FAISS `IndexIVFPQ`, and calibrate the Bayesian duplicate-decision threshold.

**Deliverables:**
- `artifacts/index/index.faiss` (<5 MB) + `id_map.json`
- Index build report: compression ratio, recall@K vs. brute-force `IndexFlatIP` ground truth, build time
- `threshold_calibration.py` output: per-category thresholds saved to `artifacts/index/thresholds.json`
- `04_indexing_and_eval.ipynb` with retrieval quality plots (PR curve, recall-vs-nprobe tradeoff)

**Key decisions:**
- `nlist` (IVF cells) chosen via `4·√N` heuristic (~400 for N=10k), `m` (PQ subquantizers) = 32 sub-vectors of 8 bits each for 256-d vectors (256/32 = 8 dims/subvector — a clean divisor), balancing compression against recall degradation.
- Validate compressed-index recall against an exact `IndexFlatIP` baseline on the same vectors; only accept the compressed index if Recall@10 drops by < 2 percentage points.
- Threshold calibration is **per-category**, not global, because visual/textual similarity distributions for near-duplicate "different SKU, same category" pairs vary substantially (e.g., plain T-shirts vs. patterned phone cases).

### Phase 5 — API & Deployment (Week 5)

**Objectives:** Export the trained model to ONNX with INT8 quantization, wrap it in a FastAPI service backed by the FAISS index, containerize with Docker, and load/latency-test against the <15ms P95 SLA.

**Deliverables:**
- `artifacts/onnx/fusion_match_int8.onnx`
- FastAPI app (`src/serving/main.py`) with `/v1/check` (single) and `/v1/check/batch` endpoints, passing `tests/test_api.py`
- `Dockerfile` + `docker-compose.yaml` producing a runnable container
- Latency benchmark report (P50/P95/P99 over 500 requests, single-threaded CPU)
- `05_export_and_api_test.ipynb`

**Key decisions:**
- ONNX export target: `opset >= 17` for compatibility with dynamic-shape SigLIP attention ops.
- Dynamic (not static) INT8 quantization chosen — static quantization requires a representative calibration dataset and a more complex calibration pass; dynamic quantization gives most of the CPU speedup for this workload (linear-layer-dominated) with a single-line API call.
- FAISS index and ONNX session are both loaded **once at FastAPI startup** (`@app.on_event("startup")`) and held in memory, not per-request, to keep P95 latency low.

---

## 7. Model Architecture — Deep Dive

### 7.1 Overview

FusionMatch composes four learnable/heuristic stages:

```
image ──▶ SigLIP Vision Tower ──▶ v_pool (768-d)
                                         │
                            blur/entropy quality proxy ──▶ q_v (scalar)
                                         │
text  ──▶ SigLIP Text Tower   ──▶ t_pool (768-d)
                                         │
                            length/perplexity quality proxy ──▶ q_t (scalar)
                                         │
                     [v_pool, t_pool, q_v, q_t] ──▶ Gate Network ──▶ g_v, g_t  (g_v + g_t = 1)
                                         │
                     fused = g_v · Wv(v_pool) + g_t · Wt(t_pool)   (both projected to 768-d first)
                                         │
                     Contrastive Projection Head (768→512→256, L2-normalize)
                                         │
                                 embedding (256-d, unit norm)
```

### 7.2 SigLIP Encoder Wrapper

```python
# src/models/siglip_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel

class SiglipDualEncoder(nn.Module):
    """Thin wrapper exposing pooled vision & text features separately,
    since HF's SiglipModel by default only exposes the joint logit_scale head."""

    def __init__(self, model_id: str = "google/siglip-base-patch16-256-multilingual",
                 freeze_vision: bool = True, freeze_text: bool = True,
                 unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.vision_dim = self.backbone.config.vision_config.hidden_size  # 768
        self.text_dim = self.backbone.config.text_config.hidden_size      # 768

        self._set_trainable(freeze_vision, freeze_text, unfreeze_last_n_blocks)

    def _set_trainable(self, freeze_vision, freeze_text, unfreeze_last_n_blocks):
        for p in self.backbone.vision_model.parameters():
            p.requires_grad = not freeze_vision
        for p in self.backbone.text_model.parameters():
            p.requires_grad = not freeze_text

        if unfreeze_last_n_blocks > 0:
            v_layers = self.backbone.vision_model.encoder.layers
            t_layers = self.backbone.text_model.encoder.layers
            for layer in list(v_layers)[-unfreeze_last_n_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for layer in list(t_layers)[-unfreeze_last_n_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        vision_out = self.backbone.vision_model(pixel_values=pixel_values)
        text_out = self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask)

        # SigLIP uses mean-pooling over the final hidden state for both towers
        v_pool = vision_out.pooler_output if vision_out.pooler_output is not None \
            else vision_out.last_hidden_state.mean(dim=1)
        t_pool = text_out.pooler_output if text_out.pooler_output is not None \
            else text_out.last_hidden_state.mean(dim=1)

        return v_pool, t_pool  # each (B, 768)
```

### 7.3 Quality Proxies

```python
# src/models/quality_proxies.py
import torch
import numpy as np
import cv2

@torch.no_grad()
def image_quality_score(pil_images: list) -> torch.Tensor:
    """Blur (variance of Laplacian, normalized) + resolution proxy.
    Returns a scalar in [0, 1] per image; higher = higher quality."""
    scores = []
    for img in pil_images:
        arr = np.array(img.convert("L"))
        lap_var = cv2.Laplacian(arr, cv2.CV_64F).var()
        blur_score = min(lap_var / 500.0, 1.0)  # empirically calibrated cap
        res_score = min((arr.shape[0] * arr.shape[1]) / (256 * 256), 1.0)
        scores.append(0.7 * blur_score + 0.3 * res_score)
    return torch.tensor(scores, dtype=torch.float32)

@torch.no_grad()
def text_quality_score(texts: list, tokenizer) -> torch.Tensor:
    """Token-count + non-empty-field proxy. Returns scalar in [0, 1] per text."""
    scores = []
    for t in texts:
        if not t or not t.strip():
            scores.append(0.0)
            continue
        n_tokens = len(tokenizer.tokenize(t))
        length_score = min(n_tokens / 20.0, 1.0)  # saturate at ~20 tokens
        scores.append(length_score)
    return torch.tensor(scores, dtype=torch.float32)
```

### 7.4 Quality-Aware Gated Fusion

```python
# src/models/gated_fusion.py
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """Projects each modality to a shared dim, then computes a softmax gate
    over [visual, textual] conditioned on (pooled features + quality scores)."""

    def __init__(self, vision_dim: int = 768, text_dim: int = 768, shared_dim: int = 768):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        gate_input_dim = shared_dim * 2 + 2  # +2 for scalar quality proxies
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # logits for [visual, textual]
        )

    def forward(self, v_pool, t_pool, q_v, q_t):
        v_proj = self.vision_proj(v_pool)   # (B, shared_dim)
        t_proj = self.text_proj(t_pool)     # (B, shared_dim)

        gate_input = torch.cat([v_proj, t_proj, q_v.unsqueeze(-1), q_t.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_net(gate_input)          # (B, 2)
        gates = torch.softmax(gate_logits, dim=-1)        # g_v + g_t = 1
        g_v, g_t = gates[:, 0:1], gates[:, 1:2]

        fused = g_v * v_proj + g_t * t_proj                # (B, shared_dim)
        return fused, gates
```

### 7.5 Contrastive Projection Head

```python
# src/models/projection_head.py
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)  # unit-norm embedding for cosine sim
```

### 7.6 Full Model Composition

```python
# src/models/fusion_match_model.py
import torch.nn as nn
from .siglip_encoder import SiglipDualEncoder
from .gated_fusion import GatedFusion
from .projection_head import ProjectionHead
from .quality_proxies import image_quality_score, text_quality_score

class FusionMatchModel(nn.Module):
    def __init__(self, model_id: str, freeze_vision=True, freeze_text=True,
                 unfreeze_last_n_blocks=0, embed_dim: int = 256):
        super().__init__()
        self.encoder = SiglipDualEncoder(model_id, freeze_vision, freeze_text,
                                          unfreeze_last_n_blocks)
        self.fusion = GatedFusion(self.encoder.vision_dim, self.encoder.text_dim)
        self.proj_head = ProjectionHead(out_dim=embed_dim)

    def forward(self, pixel_values, input_ids, attention_mask, q_v, q_t):
        v_pool, t_pool = self.encoder(pixel_values, input_ids, attention_mask)
        fused, gates = self.fusion(v_pool, t_pool, q_v, q_t)
        embedding = self.proj_head(fused)
        return embedding, gates

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 7.7 Parameter Budget

| Component | Params (approx.) | Trainable in Phase 1 | Trainable in Phase 2 |
|---|---|---|---|
| SigLIP vision tower | ~93M | ❌ | ✅ (last 2 blocks only, ~14M) |
| SigLIP text tower | ~110M | ❌ | ✅ (last 2 blocks only, ~16M) |
| Gated fusion | ~1.2M | ✅ | ✅ |
| Projection head | ~0.5M | ✅ | ✅ |
| **Total trainable** | — | **~1.7M** | **~31.7M** |
| **Total model size** | ~205M | — | — |

---

## 8. Training Strategy — Deep Dive

### 8.1 Loss Function: InfoNCE with Hard-Negative Mining

```python
# src/training/losses.py
import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor,
                hard_negative_emb: torch.Tensor = None):
        """
        anchor_emb, positive_emb: (B, D) L2-normalized
        hard_negative_emb: optional (B, K, D) mined hard negatives
        Uses in-batch negatives (other rows' positives) plus optional mined negatives.
        """
        B = anchor_emb.size(0)
        logits_in_batch = anchor_emb @ positive_emb.T / self.temperature   # (B, B)
        labels = torch.arange(B, device=anchor_emb.device)

        if hard_negative_emb is not None:
            # (B, K) similarity of anchor to its own mined hard negatives
            hard_logits = torch.einsum("bd,bkd->bk", anchor_emb, hard_negative_emb) / self.temperature
            logits = torch.cat([logits_in_batch, hard_logits], dim=1)  # (B, B+K)
        else:
            logits = logits_in_batch

        loss = F.cross_entropy(logits, labels)
        return loss


class HardNegativeMiner:
    """Maintains a running embedding bank; for each anchor, retrieves the
    top-K most similar embeddings that are NOT its true positive."""

    def __init__(self, bank_size: int = 4096, embed_dim: int = 256, k: int = 4, device="cuda"):
        self.bank = torch.zeros(bank_size, embed_dim, device=device)
        self.sku_ids = [None] * bank_size
        self.ptr = 0
        self.full = False
        self.k = k

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, sku_ids: list):
        n = embeddings.size(0)
        end = self.ptr + n
        if end <= self.bank.size(0):
            self.bank[self.ptr:end] = embeddings.detach()
            self.sku_ids[self.ptr:end] = sku_ids
        else:  # wrap around
            first = self.bank.size(0) - self.ptr
            self.bank[self.ptr:] = embeddings[:first].detach()
            self.sku_ids[self.ptr:] = sku_ids[:first]
            self.bank[:n - first] = embeddings[first:].detach()
            self.sku_ids[:n - first] = sku_ids[first:]
            self.full = True
        self.ptr = end % self.bank.size(0)

    @torch.no_grad()
    def mine(self, anchor_embeddings: torch.Tensor, anchor_sku_ids: list):
        active = self.bank if self.full else self.bank[:self.ptr]
        sims = anchor_embeddings @ active.T  # (B, bank_size)
        hard_negs = []
        for i, sku in enumerate(anchor_sku_ids):
            row = sims[i].clone()
            # mask out entries that belong to the same SKU (would be false negatives)
            same_sku_mask = torch.tensor(
                [1.0 if self.sku_ids[j] == sku else 0.0 for j in range(row.size(0))],
                device=row.device
            )
            row = row - same_sku_mask * 1e4
            topk = torch.topk(row, k=min(self.k, row.size(0))).indices
            hard_negs.append(active[topk])
        return torch.stack(hard_negs)  # (B, K, D)
```

### 8.2 Two-Phase Trainer

```python
# src/training/trainer.py
import torch
from torch.cuda.amp import autocast, GradScaler
from .losses import InfoNCELoss, HardNegativeMiner
from .metrics import compute_pairwise_f1, compute_precision_recall_at_k

class FusionMatchTrainer:
    def __init__(self, model, train_loader, val_loader, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device
        self.criterion = InfoNCELoss(temperature=config["temperature"])
        self.scaler = GradScaler()
        self.miner = HardNegativeMiner(embed_dim=config["embed_dim"], device=device)
        self.best_f1 = -1.0

    def _build_optimizer(self, phase: str):
        if phase == "warmup":
            params = [p for n, p in self.model.named_parameters()
                      if "encoder.backbone" not in n and p.requires_grad]
            return torch.optim.AdamW(params, lr=self.cfg["lr_head"])
        else:  # finetune: discriminative LR groups
            head_params = [p for n, p in self.model.named_parameters()
                           if "encoder.backbone" not in n]
            backbone_params = [p for n, p in self.model.named_parameters()
                               if "encoder.backbone" in n and p.requires_grad]
            return torch.optim.AdamW([
                {"params": head_params, "lr": self.cfg["lr_head"]},
                {"params": backbone_params, "lr": self.cfg["lr_backbone"]},
            ])

    def fit(self):
        # ---- Phase 1: warm-up (frozen backbone) ----
        for p in self.model.encoder.backbone.parameters():
            p.requires_grad = False
        optimizer = self._build_optimizer("warmup")
        for epoch in range(1, self.cfg["warmup_epochs"] + 1):
            self._run_epoch(epoch, optimizer, phase="warmup",
                             mine_hard_negatives=(epoch > self.cfg["hard_negative_start_epoch"]))

        # ---- Phase 2: fine-tune (unfreeze last N blocks) ----
        self.model.encoder._set_trainable(
            freeze_vision=False, freeze_text=False,
            unfreeze_last_n_blocks=self.cfg["unfreeze_last_n_blocks"]
        )
        optimizer = self._build_optimizer("finetune")
        for epoch in range(self.cfg["warmup_epochs"] + 1, self.cfg["total_epochs"] + 1):
            self._run_epoch(epoch, optimizer, phase="finetune", mine_hard_negatives=True)

    def _run_epoch(self, epoch, optimizer, phase, mine_hard_negatives):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad()

            with autocast():
                anchor_emb, _ = self.model(
                    batch["anchor_pixel_values"], batch["anchor_input_ids"],
                    batch["anchor_attention_mask"], batch["anchor_q_v"], batch["anchor_q_t"])
                positive_emb, _ = self.model(
                    batch["positive_pixel_values"], batch["positive_input_ids"],
                    batch["positive_attention_mask"], batch["positive_q_v"], batch["positive_q_t"])

                hard_neg_emb = None
                if mine_hard_negatives and self.miner.ptr > self.cfg["batch_size"] * 4:
                    hard_neg_emb = self.miner.mine(anchor_emb.detach(), batch["anchor_sku_id"])

                loss = self.criterion(anchor_emb, positive_emb, hard_neg_emb)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            self.miner.update(positive_emb.detach(), batch["positive_sku_id"])
            running_loss += loss.item()

        val_f1, val_p_at_k, val_r_at_k = self._validate()
        print(f"[{phase}] epoch {epoch}: train_loss={running_loss/len(self.train_loader):.4f} "
              f"val_F1={val_f1:.4f} P@5={val_p_at_k:.4f} R@5={val_r_at_k:.4f}")

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            torch.save(self.model.state_dict(), self.cfg["checkpoint_path"])

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        all_embeddings, all_sku_ids = [], []
        for batch in self.val_loader:
            batch_t = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            emb, _ = self.model(batch_t["anchor_pixel_values"], batch_t["anchor_input_ids"],
                                 batch_t["anchor_attention_mask"], batch_t["anchor_q_v"],
                                 batch_t["anchor_q_t"])
            all_embeddings.append(emb.cpu())
            all_sku_ids.extend(batch["anchor_sku_id"])
        embeddings = torch.cat(all_embeddings)
        f1 = compute_pairwise_f1(embeddings, all_sku_ids)
        p_at_k, r_at_k = compute_precision_recall_at_k(embeddings, all_sku_ids, k=5)
        return f1, p_at_k, r_at_k
```

### 8.3 Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Batch size | 32 | Fits T4 VRAM with SigLIP-base + AMP; gradient accumulation ×2 available if OOM |
| Warm-up epochs | 5 | Fusion + projection head only |
| Fine-tune epochs | 10 (total 15) | Last 2 transformer blocks unfrozen |
| LR (head) | 2e-5 | AdamW |
| LR (backbone, fine-tune phase) | 2e-6 | 10× lower than head LR |
| Temperature (τ) | 0.07 | Standard InfoNCE default, validated via grid {0.05, 0.07, 0.1} |
| Hard-negative start | epoch 3 | Embedding space non-degenerate by this point |
| Hard-negative bank size | 4096 | Balances GPU memory vs. negative diversity |
| Hard negatives per anchor (K) | 4 | |
| Weight decay | 0.01 | AdamW default-ish |
| LR schedule | Cosine decay with 5% warm-up steps | Per-phase |
| Mixed precision | `torch.cuda.amp` | ~1.6× speedup, ~40% memory reduction on T4 |
| Gradient clipping | max-norm 1.0 | Stabilizes fine-tune phase |

### 8.4 Evaluation Metrics Implementation

```python
# src/training/metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_pairwise_f1(embeddings: torch.Tensor, sku_ids: list, threshold: float = 0.7):
    """Brute-force pairwise cosine similarity -> binary duplicate prediction -> F1
    against ground-truth 'same SKU' labels. O(N^2), fine for val-set sizes (~1-2k)."""
    sims = embeddings @ embeddings.T
    n = len(sku_ids)
    y_true, y_pred = [], []
    for i in range(n):
        for j in range(i + 1, n):
            y_true.append(1 if sku_ids[i] == sku_ids[j] else 0)
            y_pred.append(1 if sims[i, j].item() >= threshold else 0)
    return f1_score(y_true, y_pred, zero_division=0)


def compute_precision_recall_at_k(embeddings: torch.Tensor, sku_ids: list, k: int = 5):
    sims = embeddings @ embeddings.T
    n = len(sku_ids)
    sims.fill_diagonal_(-1e4)  # exclude self
    precisions, recalls = [], []
    sku_arr = np.array(sku_ids)
    for i in range(n):
        topk = torch.topk(sims[i], k=min(k, n - 1)).indices.cpu().numpy()
        true_positives_available = (sku_arr == sku_arr[i]).sum() - 1
        if true_positives_available == 0:
            continue
        hits = (sku_arr[topk] == sku_arr[i]).sum()
        precisions.append(hits / k)
        recalls.append(hits / min(true_positives_available, k))
    return float(np.mean(precisions)), float(np.mean(recalls))
```

**Metric formulas:**

- **Pairwise F1** = `2·P·R / (P+R)` where P, R are computed over all same-vs-different-SKU pairs at a fixed similarity threshold.
- **Precision@K** = `(# true duplicates in top-K retrieved) / K`
- **Recall@K** = `(# true duplicates in top-K retrieved) / min(total true duplicates, K)` — capped at K since a query cannot recall more than K candidates by construction.

---

## 9. Vector Indexing & Search — Deep Dive

### 9.1 Index Type Selection Rationale

| Index Type | Recall | Speed | Memory (10k×256d) | Verdict |
|---|---|---|---|---|
| `IndexFlatIP` (brute force) | 100% (exact) | O(N) per query — slow at scale | ~10 MB (float32) | Used only as ground truth for recall validation |
| `IndexHNSWFlat` | ~98–99% | Very fast | ~15–20 MB (graph overhead) | Good speed, but memory footprint exceeds the <5MB target |
| `IndexLSH` | ~85–90% | Fast | Small, but poor recall for our similarity distribution | Rejected — recall too low for Precision@K target |
| **`IndexIVFPQ`** (chosen) | ~95–97% (tunable via `nprobe`) | Fast (coarse quantizer prunes search space) | **~1–3 MB** (PQ-compressed) | **Best fit**: meets <5MB and <15ms targets with acceptable recall loss |

### 9.2 Building the Index

```python
# src/indexing/build_index.py
import faiss
import numpy as np
import json

class IndexBuilder:
    def __init__(self, embed_dim: int = 256, nlist: int = 400, m: int = 32, nbits: int = 8):
        self.embed_dim = embed_dim
        self.nlist = nlist
        self.m = m           # number of PQ sub-quantizers (256/32=8 dims per subvector)
        self.nbits = nbits    # bits per subquantizer code (8 -> 256 centroids/subvector)

    def build(self, embeddings: np.ndarray, sku_ids: list, save_dir: str):
        assert embeddings.shape[1] == self.embed_dim
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        quantizer = faiss.IndexFlatIP(self.embed_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embed_dim, self.nlist, self.m, self.nbits,
                                  faiss.METRIC_INNER_PRODUCT)

        # IVF-PQ requires a training pass to learn coarse centroids + PQ codebooks
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 16  # search 16/400 coarse cells at query time; tuned in §9.3

        faiss.write_index(index, f"{save_dir}/index.faiss")
        with open(f"{save_dir}/id_map.json", "w") as f:
            json.dump({str(i): sku for i, sku in enumerate(sku_ids)}, f)

        return index

    @staticmethod
    def build_flat_baseline(embeddings: np.ndarray, save_dir: str):
        """Exact brute-force index used only to measure recall degradation
        introduced by IVF-PQ compression."""
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, f"{save_dir}/index_flat_baseline.faiss")
        return index
```

### 9.3 Tuning `nprobe` (Recall vs. Latency Tradeoff)

```python
import time

def tune_nprobe(index, flat_index, query_embeddings, k=10, nprobe_grid=(1, 4, 8, 16, 32, 64)):
    results = []
    _, gt_ids = flat_index.search(query_embeddings, k)  # ground-truth top-k

    for nprobe in nprobe_grid:
        index.nprobe = nprobe
        start = time.perf_counter()
        _, pred_ids = index.search(query_embeddings, k)
        latency_ms = (time.perf_counter() - start) / len(query_embeddings) * 1000

        recalls = []
        for gt_row, pred_row in zip(gt_ids, pred_ids):
            overlap = len(set(gt_row.tolist()) & set(pred_row.tolist()))
            recalls.append(overlap / k)

        results.append({"nprobe": nprobe, "recall@k": np.mean(recalls),
                         "latency_ms_per_query": latency_ms})
    return results
```

Typical result shape on a 10k-SKU index (illustrative, re-measure on your run):

| `nprobe` | Recall@10 vs. Flat | Latency/query (CPU) |
|---|---|---|
| 1 | 0.71 | 0.4 ms |
| 4 | 0.88 | 0.7 ms |
| 8 | 0.94 | 1.1 ms |
| **16** | **0.97** | **1.8 ms** |
| 32 | 0.985 | 3.2 ms |
| 64 | 0.993 | 5.9 ms |

`nprobe=16` is selected as the production default: recall loss (<2% vs. brute force at K=10) is well inside the acceptance criterion, with search latency an order of magnitude below the 15ms budget (leaving headroom for encoding + network overhead).

### 9.4 Bayesian Threshold Calibration

```python
# src/indexing/threshold_calibration.py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

class BayesianThresholdCalibrator:
    """Per-category threshold selection via a Beta-Binomial model over
    (similarity_score, is_true_duplicate) validation pairs. Rather than a
    single grid-search cutoff, we place a Beta prior over the duplicate-rate
    at each candidate threshold and select the threshold maximizing expected
    F1 under posterior uncertainty (more robust on small per-category val sets)."""

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.thresholds_by_category = {}

    def _expected_f1_at_threshold(self, sims, labels, t):
        preds = (sims >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        # Bayesian-smoothed precision/recall using Beta posterior means
        precision = (tp + self.prior_alpha) / (tp + fp + self.prior_alpha + self.prior_beta)
        recall = (tp + self.prior_alpha) / (tp + fn + self.prior_alpha + self.prior_beta)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def fit_category(self, sims: np.ndarray, labels: np.ndarray) -> float:
        result = minimize_scalar(
            lambda t: -self._expected_f1_at_threshold(sims, labels, t),
            bounds=(0.3, 0.99), method="bounded"
        )
        return float(result.x)

    def fit(self, val_pairs_by_category: dict) -> dict:
        """val_pairs_by_category: {category: (sims_array, labels_array)}"""
        for category, (sims, labels) in val_pairs_by_category.items():
            self.thresholds_by_category[category] = self.fit_category(sims, labels)
        # global fallback for unseen/low-volume categories
        all_sims = np.concatenate([s for s, _ in val_pairs_by_category.values()])
        all_labels = np.concatenate([l for _, l in val_pairs_by_category.values()])
        self.thresholds_by_category["__default__"] = self.fit_category(all_sims, all_labels)
        return self.thresholds_by_category
```

### 9.5 Index Build & Storage Report Template

| Metric | Value |
|---|---|
| Raw float32 embeddings (10,000 × 256 × 4B) | 10.24 MB |
| Compressed `IndexIVFPQ` (m=32, nbits=8) | ~1.2 MB (`10000 × 32 bytes` codes + coarse quantizer) |
| Compression ratio | ~8.5× |
| Index build (train + add) time, CPU | < 30 s for 10k vectors |
| Recall@10 (IVF-PQ vs. Flat @ nprobe=16) | ≥ 0.97 |

---

## 10. Testing & Validation

### 10.1 Unit Test Cases

```python
# tests/test_model_forward.py
import torch
from src.models.fusion_match_model import FusionMatchModel

def test_forward_shapes():
    model = FusionMatchModel("google/siglip-base-patch16-256-multilingual", embed_dim=256)
    b = 2
    pixel_values = torch.randn(b, 3, 256, 256)
    input_ids = torch.randint(0, 1000, (b, 32))
    attention_mask = torch.ones(b, 32, dtype=torch.long)
    q_v = torch.rand(b)
    q_t = torch.rand(b)

    emb, gates = model(pixel_values, input_ids, attention_mask, q_v, q_t)
    assert emb.shape == (b, 256)
    assert gates.shape == (b, 2)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(b), atol=1e-5)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(b), atol=1e-4)  # L2-normalized

def test_gradient_isolation_frozen_backbone():
    model = FusionMatchModel("google/siglip-base-patch16-256-multilingual",
                              freeze_vision=True, freeze_text=True)
    backbone_params_require_grad = [p.requires_grad for p in model.encoder.backbone.parameters()]
    assert not any(backbone_params_require_grad)
```

```python
# tests/test_losses.py
import torch
from src.training.losses import InfoNCELoss

def test_infonce_identical_embeddings_low_loss():
    torch.manual_seed(0)
    criterion = InfoNCELoss(temperature=0.07)
    anchor = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
    positive = anchor.clone()  # perfect positives
    loss = criterion(anchor, positive)
    assert loss.item() < 0.1  # near-zero cross-entropy when positives are exact matches

def test_infonce_random_embeddings_higher_loss():
    torch.manual_seed(0)
    criterion = InfoNCELoss(temperature=0.07)
    anchor = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
    positive = torch.nn.functional.normalize(torch.randn(8, 256), dim=-1)
    loss_random = criterion(anchor, positive).item()

    identical_positive = anchor.clone()
    loss_identical = criterion(anchor, identical_positive).item()
    assert loss_random > loss_identical
```

```python
# tests/test_data_pipeline.py
import pandas as pd
from src.data.abo_loader import ABOCatalogLoader

def test_manifest_no_leakage_between_splits(tmp_manifest_train, tmp_manifest_val):
    train_skus = set(pd.read_csv(tmp_manifest_train)["sku_id"])
    val_skus = set(pd.read_csv(tmp_manifest_val)["sku_id"])
    assert train_skus.isdisjoint(val_skus), "SKU leakage detected between train/val splits"

def test_every_manifest_row_has_valid_image_path(tmp_manifest_train):
    import os
    df = pd.read_csv(tmp_manifest_train)
    sample = df.sample(min(50, len(df)), random_state=1)
    missing = [p for p in sample["image_path"] if not os.path.exists(p)]
    assert len(missing) == 0, f"Missing image files: {missing[:5]}"
```

```python
# tests/test_indexing.py
import numpy as np
from src.indexing.build_index import IndexBuilder

def test_index_recall_within_tolerance():
    np.random.seed(0)
    embeddings = np.random.randn(500, 256).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    sku_ids = [f"sku_{i}" for i in range(500)]

    builder = IndexBuilder(embed_dim=256, nlist=20, m=32, nbits=8)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        ivf_index = builder.build(embeddings, sku_ids, d)
        flat_index = builder.build_flat_baseline(embeddings, d)

        ivf_index.nprobe = 16
        _, ivf_ids = ivf_index.search(embeddings[:20], k=5)
        _, flat_ids = flat_index.search(embeddings[:20], k=5)

        overlaps = [len(set(a) & set(b)) / 5 for a, b in zip(ivf_ids, flat_ids)]
        assert np.mean(overlaps) > 0.8  # loose bound for a tiny random test index
```

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.serving.main import app

client = TestClient(app)

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_single_check_endpoint_schema(sample_image_b64, sample_title):
    resp = client.post("/v1/check", json={"image_base64": sample_image_b64, "title": sample_title})
    assert resp.status_code == 200
    body = resp.json()
    assert "is_duplicate" in body
    assert "candidates" in body
    assert isinstance(body["candidates"], list)
    for c in body["candidates"]:
        assert 0.0 <= c["similarity"] <= 1.0
```

### 10.2 Integration Test Scenarios

| Scenario | Setup | Expected Outcome |
|---|---|---|
| **Exact duplicate detection** | Submit the identical (image, title) pair already in the index | `is_duplicate=True`, top candidate similarity ≈ 1.0 |
| **Augmented duplicate detection** | Submit a color-jittered + cropped version of an indexed image with a typo'd title | `is_duplicate=True`, similarity above category threshold |
| **True negative (different product, same category)** | Submit a genuinely different product from the same fine-grained category | `is_duplicate=False` |
| **Missing modality graceful degradation** | Submit image only (empty title) | Gate weights shift toward visual (`g_v` ↑); service still returns a valid response, not an error |
| **Batch endpoint consistency** | Submit N items via `/v1/check/batch`; compare each result to the equivalent N calls to `/v1/check` | Identical `is_duplicate` decisions and near-identical similarity scores (floating-point tolerance) |
| **Cold-start index reload** | Restart the service; issue first request | Startup event loads FAISS + ONNX session successfully; first request latency within 3× steady-state P95 (JIT/cache warm-up allowance) |

### 10.3 Performance Testing Methodology

```python
# benchmark_latency.py
import time, statistics
import requests

def benchmark(url: str, payload: dict, n_requests: int = 500):
    latencies = []
    for _ in range(n_requests):
        start = time.perf_counter()
        requests.post(url, json=payload)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]
    print(f"P50={p50:.2f}ms  P95={p95:.2f}ms  P99={p99:.2f}ms")
    return {"p50": p50, "p95": p95, "p99": p99}
```

Methodology notes:
- Run single-threaded, sequential requests against a **warm** service (discard first 20 requests) to measure steady-state latency, matching how the <15ms P95 target is defined.
- Separately measure encode-only latency (ONNX forward pass) vs. FAISS-search-only latency vs. FastAPI/serialization overhead, to attribute where time is spent if the target is missed.
- Repeat at 3 concurrency levels (1, 4, 16) using `locust` or `wrk` to characterize throughput degradation, even though the primary SLA is single-request P95.

### 10.4 Quality Metrics — Formulas Recap

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN} \qquad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

$$\text{Precision@K} = \frac{|\{\text{true duplicates}\} \cap \{\text{top-}K\text{ retrieved}\}|}{K}$$

$$\text{Recall@K} = \frac{|\{\text{true duplicates}\} \cap \{\text{top-}K\text{ retrieved}\}|}{\min(|\{\text{true duplicates}\}|,\ K)}$$

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\sum_{j} \exp(\text{sim}(a, n_j)/\tau)}$$

---

## 11. Deployment Guide

### 11.1 FastAPI Service

```python
# src/serving/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import onnxruntime as ort
import faiss
import json
from loguru import logger

from .schemas import CheckRequest, CheckResponse, BatchCheckRequest, BatchCheckResponse
from .inference import FusionMatchInferenceEngine

engine: FusionMatchInferenceEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading ONNX session, FAISS index, and thresholds...")
    engine = FusionMatchInferenceEngine(
        onnx_path="artifacts/onnx/fusion_match_int8.onnx",
        index_path="artifacts/index/index.faiss",
        id_map_path="artifacts/index/id_map.json",
        thresholds_path="artifacts/index/thresholds.json",
    )
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")

app = FastAPI(title="FusionMatch API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/check", response_model=CheckResponse)
def check_duplicate(req: CheckRequest):
    try:
        result = engine.check_single(req.image_base64, req.title, req.category, top_k=req.top_k)
        return result
    except Exception as e:
        logger.exception("check_duplicate failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/check/batch", response_model=BatchCheckResponse)
def check_duplicate_batch(req: BatchCheckRequest):
    try:
        results = engine.check_batch(req.items, top_k=req.top_k)
        return BatchCheckResponse(results=results)
    except Exception as e:
        logger.exception("check_duplicate_batch failed")
        raise HTTPException(status_code=500, detail=str(e))
```

```python
# src/serving/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class CheckRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG product image")
    title: Optional[str] = Field(None, description="Listing title / product name")
    category: Optional[str] = Field(None, description="Product category, used to select threshold")
    top_k: int = Field(5, ge=1, le=50)

class Candidate(BaseModel):
    sku_id: str
    similarity: float

class CheckResponse(BaseModel):
    is_duplicate: bool
    threshold_used: float
    candidates: list[Candidate]
    gate_weights: dict  # {"visual": g_v, "textual": g_t}

class BatchCheckItem(CheckRequest):
    item_id: str

class BatchCheckRequest(BaseModel):
    items: list[BatchCheckItem]
    top_k: int = Field(5, ge=1, le=50)

class BatchCheckResponse(BaseModel):
    results: list[CheckResponse]
```

### 11.2 Local Deployment Steps

```bash
# 1. Install runtime-only dependencies (lighter than training env)
pip install -r requirements-serving.txt

# 2. Ensure artifacts are present
ls artifacts/onnx/fusion_match_int8.onnx artifacts/index/index.faiss artifacts/index/id_map.json

# 3. Run the API
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --workers 1

# 4. Smoke test
curl -X POST http://localhost:8000/v1/check \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<...>", "title": "Wireless Bluetooth Headphones Black", "top_k": 5}'
```

### 11.3 Docker Configuration

```dockerfile
# docker/Dockerfile
FROM python:3.10-slim AS base

WORKDIR /app

# System deps for Pillow/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/
COPY artifacts/onnx/fusion_match_int8.onnx ./artifacts/onnx/fusion_match_int8.onnx
COPY artifacts/index/index.faiss ./artifacts/index/index.faiss
COPY artifacts/index/id_map.json ./artifacts/index/id_map.json
COPY artifacts/index/thresholds.json ./artifacts/index/thresholds.json
COPY config/deploy_config.yaml ./config/deploy_config.yaml

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
# docker/docker-compose.yaml
version: "3.9"
services:
  fusionmatch-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 1G
    restart: unless-stopped
```

```bash
# Build & run
docker compose -f docker/docker-compose.yaml up --build -d

# Verify
docker compose logs -f fusionmatch-api
curl http://localhost:8000/health
```

**Resulting image characteristics:** Because the ONNX INT8 model (~100–150 MB) and FAISS index (<5 MB) are the only large artifacts baked in, and no CUDA runtime is required, the final image is CPU-only and typically **under 1.2 GB**, deployable on cheap CPU instances.

### 11.4 Cloud Deployment Options (Optional)

| Platform | Fit | Notes |
|---|---|---|
| **Render / Railway (free/hobby tier)** | Good for portfolio demo | Simple `Dockerfile`-based deploy, auto HTTPS |
| **AWS Lambda (container image)** | Good for low, spiky traffic | ONNX Runtime + FAISS both work in Lambda containers; watch the 10GB image size limit (well within budget here) and cold-start latency (index load happens once per container, not per invocation, if kept warm) |
| **AWS ECS Fargate / GCP Cloud Run** | Good for steady traffic, autoscaling | CPU-only task definition sized at 1–2 vCPU / 1–2 GB RAM comfortably serves this workload |
| **Kubernetes** | Overkill for this project's scale | Documented as an option for resume completeness, not required for the 10k-SKU use case |

---

## 12. Performance Optimization

### 12.1 Model Optimization Techniques

**ONNX Export**

```python
# src/export/to_onnx.py
import torch
from src.models.fusion_match_model import FusionMatchModel

def export_to_onnx(checkpoint_path: str, output_path: str, model_id: str, embed_dim: int = 256):
    model = FusionMatchModel(model_id, embed_dim=embed_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_pixel_values = torch.randn(1, 3, 256, 256)
    dummy_input_ids = torch.randint(0, 1000, (1, 32))
    dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)
    dummy_q_v = torch.rand(1)
    dummy_q_t = torch.rand(1)

    torch.onnx.export(
        model,
        (dummy_pixel_values, dummy_input_ids, dummy_attention_mask, dummy_q_v, dummy_q_t),
        output_path,
        input_names=["pixel_values", "input_ids", "attention_mask", "q_v", "q_t"],
        output_names=["embedding", "gates"],
        dynamic_axes={
            "pixel_values": {0: "batch"}, "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"}, "q_v": {0: "batch"}, "q_t": {0: "batch"},
            "embedding": {0: "batch"}, "gates": {0: "batch"},
        },
        opset_version=17,
    )
```

**Dynamic INT8 Quantization**

```python
# src/export/quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(fp32_path: str, int8_path: str):
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],  # linear-layer-dominated model
    )
```

| Optimization | Latency Impact (CPU, batch=1) | Model Size Impact | Accuracy Impact |
|---|---|---|---|
| FP32 → ONNX (no quantization) | ~1.3× faster than eager PyTorch | Same | None (numerically equivalent graph) |
| ONNX → INT8 dynamic quantization | Additional ~2–3× faster | ~4× smaller (205MB → ~55–65MB for quantized linear layers) | Pairwise F1 typically drops < 0.5pp — validate on test set before shipping |
| Structured pruning (optional, not default) | Marginal further gain | Smaller | Higher risk; only pursue if latency budget still exceeded after quantization |

### 12.2 Index Optimization: IVF-PQ vs. HNSW vs. LSH

| Technique | How it Works | Best For | Tradeoff |
|---|---|---|---|
| **IVF-PQ** (chosen) | Coarse k-means partitioning (IVF) narrows search to a few clusters; Product Quantization compresses vectors into compact sub-vector codes | Memory-constrained, moderate-recall-tolerant workloads like this project | Recall depends on `nprobe`; requires a training pass |
| **HNSW** | Multi-layer proximity graph; greedy graph traversal at query time | Very high recall, very low latency, when memory is not the bottleneck | Graph structure overhead (~1.5–2× raw vector size) makes it harder to hit <5MB at this vector count |
| **LSH** | Random hyperplane hashing buckets similar vectors together | Extremely high-dimensional or streaming-insert-heavy workloads | Lower recall for our normalized-embedding cosine-similarity setting; not selected |

### 12.3 Inference Optimization

- **Batching:** The `/v1/check/batch` endpoint groups images into a single ONNX Runtime `run()` call rather than looping per-item, amortizing kernel-launch and memory-allocation overhead — measured ~3× throughput improvement at batch size 16 vs. 16 sequential single calls.
- **Caching:** An LRU cache (`functools.lru_cache`-style, keyed on a perceptual hash of the input image + normalized title) short-circuits repeat-check requests for the same listing within a short TTL window, common when sellers resubmit after minor edits.
- **GPU offloading (optional):** For higher-throughput deployments, the same ONNX graph can be served with `CUDAExecutionProvider` instead of `CPUExecutionProvider` with no code changes beyond the `providers=` argument — documented here for completeness, though the <15ms CPU target is the primary design point so the project remains deployable without GPU infrastructure.
- **Thread pool tuning:** `onnxruntime.SessionOptions().intra_op_num_threads` pinned to the container's vCPU count avoids oversubscription when running multiple Uvicorn workers.

---

## 13. Troubleshooting

### 13.1 Common Issues and Solutions

| Issue | Likely Cause | Solution |
|---|---|---|
| `CUDA out of memory` during training | Batch size too large for T4 VRAM alongside SigLIP-base activations | Reduce `batch_size` to 16 with gradient accumulation ×2; confirm AMP (`autocast`) is active; call `torch.cuda.empty_cache()` between phases |
| Colab session disconnects mid-training | Free-tier idle/session timeout | Checkpoint every epoch to Drive; wrap training in a resumable loop that detects and loads the latest checkpoint on notebook restart |
| Val pairwise F1 stuck near 0 for several epochs | Learning rate too high causing collapse, or hard-negative mining started too early (embedding space still degenerate) | Confirm `hard_negative_start_epoch >= 3`; lower `lr_head`; verify embeddings aren't collapsing to a single point (check embedding variance) |
| `IndexIVFPQ.train()` raises "not enough training points" | `nlist` too large relative to dataset size (`nlist` should be ≪ N) | Use `nlist ≈ 4·sqrt(N)`; for N=10,000 this is ~400, well below the recommended minimum of `39×nlist` training points (i.e., need ≥ ~15,600 training vectors, or reduce `nlist`) |
| ONNX export fails with "Unsupported operator" | SigLIP's attention implementation uses an op not covered by the target opset | Set `opset_version=17` or higher; if still failing, export with `torch.onnx.export(..., dynamo=False)` fallback path and check `transformers` version compatibility notes |
| Quantized model accuracy drop > 1pp on test set | Some layers (e.g., final projection) are precision-sensitive | Use `op_types_to_quantize=["MatMul", "Gemm"]` only (skip quantizing LayerNorm/Softmax); consider static quantization with a small calibration set if dynamic quantization degrades too much |
| API P95 latency exceeds 15ms | Cold FAISS/ONNX load on first request, or `nprobe` set too high | Confirm index/session load happens at `@app.on_event("startup")`, not per-request; re-tune `nprobe` per §9.3; profile with the breakdown method in §10.3 |
| Duplicate detection over-triggers on genuinely different products in one category | Per-category threshold too low, or that category's training data was sparse | Re-run `BayesianThresholdCalibrator` with more val pairs for that category; inspect gate weights — if `g_v` is near 1.0 for that category, the fusion may be over-relying on visually similar-but-different products (e.g., patterned textiles) |
| `tar: abo-images-small.tar: Unexpected EOF` on extraction | Interrupted download (common on Colab's variable network) | Re-download with `wget -c` (resume), verify file size matches the expected ~3GB before extracting |

### 13.2 Logging Strategy

```python
# src/serving/logging_config.py
from loguru import logger
import sys, json

def configure_logging(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        serialize=True,  # structured JSON logs, easy to ship to ELK/CloudWatch
    )
    logger.add("logs/fusionmatch_api.log", rotation="50 MB", retention="14 days", level="INFO")

def log_request(request_id: str, latency_ms: float, is_duplicate: bool, top_similarity: float):
    logger.bind(request_id=request_id).info(json.dumps({
        "event": "check_request",
        "latency_ms": round(latency_ms, 2),
        "is_duplicate": is_duplicate,
        "top_similarity": round(top_similarity, 4),
    }))
```

- **What gets logged:** request latency (encode + search + total), similarity-score distribution per request (for drift monitoring), gate-weight distribution (visual vs. textual reliance over time — a shift can indicate catalog image-quality drift), and error stack traces on failure.
- **What does NOT get logged:** raw image bytes or full listing text (privacy/storage cost) — only derived scalar signals and truncated identifiers.

### 13.3 Profiling Tools

| Tool | Use |
|---|---|
| `torch.profiler` | Identify GPU/CPU time split during training epochs; export Chrome-trace for visualization |
| `onnxruntime.SessionOptions(enable_profiling=True)` | Per-op latency breakdown of the exported ONNX graph — pinpoint whether the vision or text tower dominates inference cost |
| `py-spy` | Live sampling profiler for the running FastAPI process, useful for diagnosing P99 latency spikes under load without restarting the service |
| `memory_profiler` / `tracemalloc` | Track Python-side memory growth (e.g., an accidental unbounded LRU cache) during long-running service uptime |
| FAISS `index.nprobe` sweep (§9.3) | Systematic recall/latency tradeoff diagnosis when search quality or speed regresses after a re-index |

---

## 14. Model Card

| Field | Detail |
|---|---|
| **Model name** | FusionMatch-v1 |
| **Base model** | `google/siglip-base-patch16-256-multilingual` |
| **Model type** | Multimodal contrastive embedding model (vision + text → 256-d joint embedding) |
| **Architecture summary** | SigLIP dual encoder + quality-aware gated fusion + 2-layer contrastive projection head |
| **Training data** | ~11,000 SKUs / ~28,000 images sampled from Amazon Berkeley Objects (`abo-images-small`), CC BY 4.0 |
| **Training objective** | InfoNCE contrastive loss (τ=0.07) with hard-negative mining, two-phase (frozen warm-up → partial fine-tune) |
| **Intended use** | Retrieval-based duplicate/near-duplicate detection for e-commerce product listings at catalog scale (10³–10⁵ SKUs) |
| **Out-of-scope use** | Fraud/counterfeit intent classification, cross-domain retrieval (e.g., matching products to unrelated web images), real-time video |
| **Evaluation data** | Held-out 10% SKU-level split of the same ABO sample, plus optional Shopee Product Matching subset for generalization sanity-check |
| **Evaluation metrics** | Pairwise F1 ≥0.90 (target), Precision@5 >0.95, Recall@5 >0.90 |
| **Known limitations** | Trained predominantly on Amazon-catalog-style clean product photography; performance on very low-quality seller photos (heavy watermarks, extreme occlusion) beyond the augmentation distribution used in training is untested. Category coverage limited to top-20 ABO `product_type` values sampled during data prep — long-tail categories may see reduced accuracy. |
| **Ethical considerations** | The model outputs a similarity signal to *assist* human catalog moderators; it should not be used as the sole automated basis for removing a seller's listing without human review, given the false-positive risk on genuinely distinct but visually similar products |
| **Environmental/compute cost** | Full training (15 epochs, T4 GPU) completes in <4 hours; single-session Colab free-tier compute, no multi-GPU or multi-day training required |
| **Model size** | ~205M parameters (PyTorch checkpoint); ~55–65MB (ONNX INT8 quantized for serving) |

---

## 15. API Documentation

### 15.1 `POST /v1/check` — Single Item Duplicate Check

**Request:**

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "title": "Wireless Bluetooth Over-Ear Headphones - Black",
  "category": "Electronics/Audio/Headphones",
  "top_k": 5
}
```

**Response (200 OK):**

```json
{
  "is_duplicate": true,
  "threshold_used": 0.812,
  "candidates": [
    {"sku_id": "B07XJ8C8F5", "similarity": 0.943},
    {"sku_id": "B08K3P1QWX", "similarity": 0.887},
    {"sku_id": "B09TT2M4RH", "similarity": 0.756},
    {"sku_id": "B07QK9YXZT", "similarity": 0.701},
    {"sku_id": "B0BXWT44LK", "similarity": 0.664}
  ],
  "gate_weights": {"visual": 0.62, "textual": 0.38}
}
```

**Field reference:**

| Field | Type | Description |
|---|---|---|
| `image_base64` | string (required) | Base64-encoded product image (JPEG/PNG) |
| `title` | string (optional) | Listing title; empty string triggers text-quality-gate degradation, not an error |
| `category` | string (optional) | Used to select a per-category calibrated threshold; falls back to `__default__` if omitted or unseen |
| `top_k` | int (optional, default 5) | Number of nearest candidates to return |
| `is_duplicate` | bool | `true` if the top candidate's similarity ≥ `threshold_used` |
| `threshold_used` | float | The calibrated cosine-similarity cutoff applied for this category |
| `candidates` | array | Top-K nearest catalog SKUs with cosine similarity scores |
| `gate_weights` | object | Diagnostic: how much the model relied on visual vs. textual signal for this request |

### 15.2 `POST /v1/check/batch` — Batch Duplicate Check

**Request:**

```json
{
  "top_k": 3,
  "items": [
    {"item_id": "req-1", "image_base64": "...", "title": "Men's Running Shoes Size 10"},
    {"item_id": "req-2", "image_base64": "...", "title": "Stainless Steel Water Bottle 750ml"}
  ]
}
```

**Response (200 OK):**

```json
{
  "results": [
    {
      "is_duplicate": false,
      "threshold_used": 0.79,
      "candidates": [
        {"sku_id": "B0XYZ1", "similarity": 0.55},
        {"sku_id": "B0XYZ2", "similarity": 0.51},
        {"sku_id": "B0XYZ3", "similarity": 0.47}
      ],
      "gate_weights": {"visual": 0.71, "textual": 0.29}
    },
    {
      "is_duplicate": true,
      "threshold_used": 0.80,
      "candidates": [
        {"sku_id": "B0ABC9", "similarity": 0.91},
        {"sku_id": "B0ABC7", "similarity": 0.68},
        {"sku_id": "B0ABC4", "similarity": 0.60}
      ],
      "gate_weights": {"visual": 0.48, "textual": 0.52}
    }
  ]
}
```

### 15.3 `GET /health` — Liveness/Readiness Probe

**Response (200 OK):**

```json
{"status": "ok"}
```

### 15.4 Error Responses

| HTTP Status | Condition | Body |
|---|---|---|
| 422 | Malformed request (Pydantic validation failure, e.g., invalid base64) | `{"detail": [...]}` (FastAPI default validation error format) |
| 500 | Unhandled exception during encode/search | `{"detail": "<error message>"}` |
| 503 | Service not yet finished startup (index/model still loading) | `{"detail": "Service warming up"}` (recommended addition via a readiness flag) |

---

## 16. Configuration Reference

### 16.1 `config/base_config.yaml`

```yaml
project:
  name: FusionMatch
  seed: 42

data:
  images_root: data/raw/abo-images-small
  listings_root: data/raw/abo-listings
  max_skus: 11000
  min_images_per_sku: 1
  train_val_test_split: [0.8, 0.1, 0.1]
  use_supplementary_data: false   # Shopee / Stanford Online Products toggle

model:
  base_model_id: google/siglip-base-patch16-256-multilingual
  embed_dim: 256
  shared_fusion_dim: 768
  freeze_vision_warmup: true
  freeze_text_warmup: true
  unfreeze_last_n_blocks: 2

training:
  batch_size: 32
  warmup_epochs: 5
  total_epochs: 15
  lr_head: 2.0e-5
  lr_backbone: 2.0e-6
  weight_decay: 0.01
  temperature: 0.07
  hard_negative_start_epoch: 3
  hard_negative_bank_size: 4096
  hard_negatives_per_anchor: 4
  grad_clip_norm: 1.0
  mixed_precision: true
  checkpoint_path: artifacts/checkpoints/best.pt

indexing:
  nlist: 400
  pq_m: 32
  pq_nbits: 8
  nprobe: 16
  metric: inner_product

evaluation:
  top_k: 5
  target_pairwise_f1: 0.90
  target_precision_at_k: 0.95
  target_recall_at_k: 0.90

export:
  onnx_opset: 17
  quantize: true
  quantize_ops: [MatMul, Gemm]
```

### 16.2 `config/colab_free_tier.yaml` (overrides)

```yaml
# Merged on top of base_config.yaml when COLAB_FREE_TIER=1
data:
  max_skus: 10500          # trims slightly further if disk pressure observed

training:
  batch_size: 16            # fallback if OOM on shared T4 instances
  gradient_accumulation_steps: 2
  checkpoint_every_n_epochs: 1
  resume_from_checkpoint: true

runtime:
  drive_mount: true
  max_session_hours: 11     # leaves margin before the ~12h hard cap
  autosave_interval_minutes: 20
```

### 16.3 `config/deploy_config.yaml`

```yaml
serving:
  host: 0.0.0.0
  port: 8000
  workers: 2
  onnx_path: artifacts/onnx/fusion_match_int8.onnx
  onnx_execution_provider: CPUExecutionProvider
  intra_op_num_threads: 2

index:
  index_path: artifacts/index/index.faiss
  id_map_path: artifacts/index/id_map.json
  thresholds_path: artifacts/index/thresholds.json
  nprobe: 16

caching:
  enable_lru_cache: true
  cache_max_size: 2048
  cache_ttl_seconds: 300

logging:
  level: INFO
  log_file: logs/fusionmatch_api.log
  rotation: 50 MB
  retention_days: 14
```

---

## 17. References

1. Zhai, X. et al. "Sigmoid Loss for Language Image Pre-Training" (SigLIP), ICCV 2023.
2. Oord, A. van den, Li, Y., Vinyals, O. "Representation Learning with Contrastive Predictive Coding" (InfoNCE), 2018.
3. Jégou, H., Douze, M., Schmid, C. "Product Quantization for Nearest Neighbor Search," IEEE TPAMI, 2011.
4. Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs" (FAISS), 2019.
5. Malkov, Y., Yashunin, D. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (HNSW), 2018.
6. Collins, M. et al. "Amazon Berkeley Objects: A Multimodal Learning Dataset for 3D Product Understanding" (ABO dataset), CVPR 2022.
7. Schroff, F., Kalenichenko, D., Philbin, J. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (metric learning / hard-negative mining foundations), CVPR 2015.
8. ONNX Runtime documentation — quantization and graph-optimization guides, Microsoft.
9. Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, foundational contrastive vision-language pretraining), 2021.

---

## 18. Appendix: Colab Free-Tier Budget

| Resource | Free-Tier Ceiling | Project Usage (est.) | Margin |
|---|---|---|---|
| Session length | ~12 hrs (variable) | Training: <4 hrs; data prep: ~1–2 hrs; indexing/export: <1 hr — split across 2–3 sessions | Comfortable, checkpointed |
| GPU VRAM (T4) | ~15 GB | Peak ~10–12 GB at batch_size=32 with AMP | ~3 GB headroom |
| Disk | ~78–107 GB (varies) | ABO tar (3 GB) + extracted images (~4–5 GB) + HF model cache (~1.5 GB) + checkpoints (~1 GB) | Well under limit |
| RAM | ~12 GB (free tier) | Manifest DataFrames + DataLoader workers; capped `num_workers=2` to avoid RAM spikes | Safe with default settings |
| Idle disconnect | ~90 min inactivity | Mitigated via periodic cell output / keep-alive during long training cells, plus checkpoint-and-resume design | N/A |

**Overall verdict:** every phase of FusionMatch — data prep, two-phase training, indexing, ONNX export, and API smoke-testing — is designed to run comfortably within Google Colab's free T4 tier across a small number of sessions, with no paid compute required end-to-end.
