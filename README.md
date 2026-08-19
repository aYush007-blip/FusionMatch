# FusionMatch

Cross-modal product deduplication for e-commerce catalogs using product images, titles, and category metadata.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IVF--PQ-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-CPU--optimized-2496ed.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)](tests/)

## Overview

FusionMatch identifies duplicate product listings from multiple product views, noisy titles, and catalog metadata.

The system uses a SigLIP-based vision-language encoder, multi-view pooling, quality-aware gated fusion, FAISS candidate retrieval, and category-specific decision thresholds. The production inference path is exported to ONNX and dynamically quantized to INT8 for CPU serving.

The model was trained and evaluated on the Amazon Berkeley Objects (ABO) dataset using an 80/10/10 SKU-stratified split.

## Results

Results from the final evaluation run:

| Metric | Target | Result |
|---|---:|---:|
| Validation Pairwise F1 | ≥ 90.0% | **97.98%** |
| Test Pairwise F1 | ≥ 90.0% | **97.19%** |
| Test Precision@1 | ≥ 95.0% | **99.23%** |
| Test Recall@1 | ≥ 95.0% | **99.23%** |
| Test Recall@5 | ≥ 98.0% | **99.50%** |
| Test Recall@10 | ≥ 98.0% | **99.55%** |
| FAISS search latency (`nprobe=16`) | < 1.0 ms | **0.028 ms/query** |
| FAISS index memory (10k SKUs) | < 5.0 MB | **1.20 MB** |
| INT8 ONNX model size | < 250 MB | **202.27 MB** |
| End-to-end P50 latency | < 8.0 ms | **3.85 ms** |
| End-to-end P95 latency | < 15.0 ms | **6.42 ms** |
| Docker image footprint | < 1.5 GB | **1.15 GB** |

## Architecture

```mermaid
flowchart TD
    subgraph IN["1. Product Ingestion"]
        A["<b>Incoming Product Request</b><br/>• Multi-View Images (JPEG/PNG)<br/>• Product Title & Category Metadata"]
    end

    subgraph QS["2. Modality Quality Scoring"]
        B1["<b>Visual Quality Estimator</b><br/>Laplacian Blur Variance (q_v ∈ [0, 1])"]
        B2["<b>Text Quality Estimator</b><br/>Token Count & Density (q_t ∈ [0, 1])"]
    end

    subgraph ORT["3. ONNX Runtime INT8 Engine (202 MB)"]
        C1["<b>SigLIP Dual Encoder</b><br/>Vision (Patch16-224) + Text Transformer"]
        C2["<b>Multi-View Attention Pooling</b><br/>Aggregates multi-perspective views"]
        C3["<b>Dynamic Gated Fusion</b><br/>g = σ(W · [e_v; e_t; q_v; q_t])"]
        C4["<b>L2 Unit Projection Head</b><br/>256-d Unit Vector (z ∈ S²⁵⁵)"]
    end

    subgraph RET["4. Vector Indexing & Decision Calibration"]
        D["<b>FAISS IndexIVFPQ</b><br/>• nlist=400, m=32, nbits=8, nprobe=16<br/>• Sub-0.03ms Inner-Product Retrieval"]
        E["<b>Bayesian Threshold Calibration</b><br/>• 78 Category-Specific Cutoffs<br/>• Posterior Beta-Binomial Calibration"]
    end

    subgraph OUT["5. Production Serving"]
        F["<b>FastAPI JSON Response (P95 < 6.5 ms)</b><br/>• is_duplicate: true/false<br/>• Top-K Candidate SKUs & Similarities<br/>• Visual vs. Textual Gate Weights"]
    end

    A --> B1
    A --> B2
    B1 --> C1
    B2 --> C1
    C1 --> C2 --> C3 --> C4
    C4 --> D
    D --> E
    E --> F
```


### Retrieval configuration

The production index uses `IndexIVFPQ` with:

- `nlist=400`
- `m=32`
- `nbits=8`
- `nprobe=16`
- Inner-product similarity

Measured FAISS search latency is approximately 0.028 ms/query on the evaluated setup.

## Repository Structure

```text
FusionMatch/
├── artifacts/                           # Persistent weights, indexes, and evaluation benchmarks
│   ├── checkpoints/                     # Model weights (warmup_best.pt, best.pt)
│   ├── index/                           # Compressed FAISS IndexIVFPQ & Bayesian thresholds
│   │   ├── id_map.json                  # SKU identifier index mapping
│   │   ├── index.faiss                  # Production IVF-PQ index (0.73 MB)
│   │   └── thresholds.json              # 78 calibrated category-specific decision thresholds
│   ├── metrics/                         # Benchmark reports & training trajectories
│   │   ├── final_summary_report.csv     # Master metric report (Val/Test F1, P@K, R@K)
│   │   ├── nprobe_benchmark.csv         # FAISS latency-recall Pareto frontier
│   │   ├── test_metrics.json            # Final evaluation metrics on test split
│   │   └── training_curves.png          # High-resolution loss & F1 trajectory plot
│   └── onnx/                            # Exported and quantized computation graphs
│       ├── fusion_match_fp32.onnx       # Full-precision ONNX model graph
│       └── fusion_match_int8.onnx       # INT8 dynamic quantized model (202 MB)
├── config/                              # Hyperparameters & serving configurations
│   ├── base_config.yaml                 # Core model & training hyperparameters
│   ├── colab_free_tier.yaml             # Memory-optimized Colab settings
│   └── deploy_config.yaml               # Serving engine & search configuration
├── docker/                              # Container orchestration & CPU-optimized builds
│   ├── Dockerfile                       # Multi-stage lightweight CPU image (<1.2 GB)
│   └── docker-compose.yaml              # Service orchestration with healthchecks
├── notebooks/                           # Colab training & evaluation workflows
│   └── FusionMatch_Colab_Master.ipynb   # 11-step master notebook with Drive checkpointing
├── src/                                 # Production source code
│   ├── data/                            # Dataset loaders, augmentations & quality scorers
│   │   ├── abo_loader.py                # Amazon Berkeley Objects parser & stratified splits
│   │   ├── augmentations.py             # Vision (PIL) and text (token masking) augmenters
│   │   ├── dataset.py                   # PyTorch Dataset & custom batch collator
│   │   └── quality_proxies.py           # Laplacian blur & text density estimators
│   ├── export/                          # ONNX conversion & INT8 quantization
│   │   ├── quantize.py                  # Dynamic INT8 weight quantization
│   │   └── to_onnx.py                   # PyTorch to ONNX graph export (opset 17)
│   ├── indexing/                        # Vector indexing & decision calibration
│   │   ├── build_index.py               # FAISS IndexIVFPQ builder & nprobe tuner
│   │   └── threshold_calibration.py     # Bayesian Beta-Binomial threshold calibrator
│   ├── models/                          # Neural network architectures
│   │   ├── fusion_match_model.py        # Top-level end-to-end model module
│   │   ├── gated_fusion.py              # Quality-aware softmax gating layer
│   │   ├── multi_view_pooling.py        # Multi-perspective attention pooling
│   │   └── siglip_encoder.py            # SigLIP vision & text dual encoder
│   ├── serving/                         # FastAPI inference microservice
│   │   ├── inference.py                 # ONNX Runtime + FAISS retrieval engine
│   │   ├── logging_config.py            # Structured JSON logging with Loguru
│   │   ├── main.py                      # FastAPI application & lifespan manager
│   │   └── schemas.py                   # Pydantic V2 request & candidate schemas
│   ├── training/                        # Contrastive InfoNCE training loop
│   │   ├── losses.py                    # InfoNCE loss with hard negative mining
│   │   ├── metrics.py                   # Pairwise F1 & Precision/Recall@K evaluators
│   │   └── trainer.py                   # 2-phase progressive model trainer
│   └── utils/                           # I/O, random seed, and logging utilities
├── tests/                               # Comprehensive unit & integration test suite (29 tests)
│   ├── test_api.py                      # FastAPI endpoints & ONNX inference tests
│   ├── test_data_pipeline.py            # Zero-leakage split & augmentation tests
│   ├── test_indexing.py                 # FAISS IVF-PQ & Bayesian calibrator tests
│   ├── test_losses.py                   # InfoNCE & metric calculation tests
│   └── test_model_forward.py            # Forward pass shape & gradient flow tests
├── Dockerfile                           # Root Dockerfile for container builds
├── requirements.txt                     # Development and training dependencies
└── requirements-serving.txt             # Lean serving dependencies (FastAPI + ONNX)
```

## Setup

### Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch 2.2+
- Docker (optional)

### Install

```bash
git clone https://github.com/aYush007-blip/FusionMatch.git
cd FusionMatch

python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Tests

Run the test suite with:

```bash
pytest tests/ -v
```

The evaluated repository contains 29 unit and integration tests.

## Running the API

Start the FastAPI service with:

```bash
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Once running, Swagger UI is available at:

```text
http://localhost:8000/docs
```

## API

### Health check

```bash
curl -X GET http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "indexed_vectors": 2200,
  "embed_dim": 256,
  "device": "CPU"
}
```

### Check a product

`POST /v1/check`

```bash
curl -X POST http://localhost:8000/v1/check \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64_encoded_jpeg>",
    "title": "AmazonBasics High-Speed HDMI Cable 6 Feet",
    "category": "ELECTRONICS",
    "top_k": 3
  }'
```

Example response:

```json
{
  "is_duplicate": true,
  "threshold_used": 0.8893,
  "candidates": [
    {
      "sku_id": "B014I8SSD0",
      "similarity": 0.9642
    },
    {
      "sku_id": "B014I8SX4Y",
      "similarity": 0.821
    }
  ],
  "gate_weights": {
    "visual": 0.542,
    "textual": 0.458
  }
}
```

### Batch requests

`POST /v1/check/batch`

The batch endpoint accepts up to 100 products per request and processes them in parallel.

## Docker

Build and start the service with Docker Compose:

```bash
docker compose -f docker/docker-compose.yaml up --build -d
```

Check the running service:

```bash
curl http://localhost:8000/health
```

The production container is CPU-only and has an evaluated footprint of approximately 1.15 GB.

## Model and Inference

FusionMatch combines visual and textual product information into a shared 256-dimensional embedding.

The inference pipeline includes:

1. Image and text quality estimation.
2. SigLIP-based feature extraction.
3. Attention-based aggregation of multiple product views.
4. Quality-aware gated fusion of visual and textual features.
5. L2 normalization of the fused embedding.
6. Approximate nearest-neighbor retrieval using FAISS IVF-PQ.
7. Category-specific duplicate decision using calibrated thresholds.

The production model is exported to ONNX and dynamically quantized to INT8. The evaluated model size is 202.27 MB, compared with 782 MB for the full-precision model.

## Training

The training pipeline uses contrastive learning with InfoNCE loss and online semi-hard negative mining.

Training and evaluation use SKU-level separation to prevent product-level leakage between splits. The final evaluation follows an 80/10/10 train/validation/test split.

The training workflow is available in:

```text
notebooks/FusionMatch_Colab_Master.ipynb
```

## Artifacts

Evaluation and deployment artifacts are stored under `artifacts/`:

- Trained model checkpoints
- FAISS production index
- SKU-to-index mapping
- Category-specific thresholds
- Test metrics
- FAISS latency benchmarks
- Training curves
- FP32 and INT8 ONNX models

## License

MIT License.

This project is intended for academic and industrial research use.
