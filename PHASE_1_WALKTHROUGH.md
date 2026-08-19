# Phase 1: Data Pipeline — Comprehensive Walkthrough & Verification Report

## Executive Summary

Phase 1 establishes the production-grade **Data Pipeline** for the **FusionMatch** Cross-Modal & Multi-View Product Deduplication Engine. Built strictly to the project specification (§5 & §6 Phase 1), this pipeline ingests the real Amazon Berkeley Objects (ABO) dataset, processes multi-lingual product metadata, resolves multi-angle image pathways, performs stratified SKU-level train/validation/test splitting with zero cross-split leakage, applies robust multi-modal data augmentations, and calculates visual/textual quality proxy scores for adaptive embedding fusion.

---

## 1. Pipeline Architecture & Data Flow

```text
[Raw ABO Archives]
   ├── abo-listings.tar (87.5 MB, 16 sharded JSONL files)
   └── abo-images-small.tar (3.25 GB, 398,212 images across 256 subfolders)
            │
            ▼
[ABOCatalogLoader (src/data/abo_loader.py)]
   ├── Text Extraction & Language Prioritization (en_US > en_GB > other)
   ├── Image Index Resolution (mapping image_id -> local disk path)
   ├── Category Normalization (product_type / category extraction)
   └── Quality Proxy Calculations (Laplacian blur variance & token length)
            │
            ▼
[Stratified Sampling & Partitioning]
   ├── 11,000 SKUs Sampled (Stratified across ~100 product categories)
   ├── SKU-Level Partitioning: 80% Train | 10% Val | 10% Test
   └── Verification: ZERO SKU / Image Leakage across splits
            │
            ▼
[Multi-Modal Pair & Triplet Sampler (src/data/pair_sampler.py)]
   ├── Positive Pair Construction:
   │     ├── Multi-Angle Real Photography (Item Image A <-> Item Image B)
   │     └── Synthetic Multimodal Augmentation (Augmented Visuals + Noisy Text)
   └── Hard Negative Mining:
         └── In-category different SKU sampling for contrastive discrimination
            │
            ▼
[PyTorch Dataset & DataLoaders (src/data/dataset.py)]
   ├── Dynamic Albumentations (Flip, Color Jitter, JPEG Compression, Erasing)
   ├── Text Perturbations (Keyboard typo, brand dropout, character swap)
   └── Batch Collation with Padded Tensors & Quality Metadata
```

---

## 2. Dataset & Split Verification Metrics

From the complete ABO universe of **145,454 unique SKUs** and **710,650 image-item records**, 11,000 SKUs were sampled through category-stratified selection with 100% verified on-disk images.

### Manifest Distribution Breakdown
| Split | On-Disk Images (Rows) | Unique SKUs | SKU Percentage | Row Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** (`manifest_train.csv`) | 42,957 | 8,800 | **80.00%** | 80.06% |
| **Validation** (`manifest_val.csv`) | 5,332 | 1,100 | **10.00%** | 9.94% |
| **Test** (`manifest_test.csv`) | 5,365 | 1,100 | **10.00%** | 10.00% |
| **Total** | **53,654** | **11,000** | **100.00%** | **100.00%** |

### SKU Disjointness & Zero-Leakage Confirmation
- $\text{Train} \cap \text{Val} = \emptyset$ (0 common SKUs)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 common SKUs)
- $\text{Val} \cap \text{Test} = \emptyset$ (0 common SKUs)
- **Programmatic Assertion**: `assert len(train_skus & val_skus) == 0 and len(train_skus & test_skus) == 0 and len(val_skus & test_skus) == 0` $\longrightarrow$ **PASSED (Strict Zero Leakage)**.

---

## 3. Top 20 Category Distribution (Stratified Representation)

The dataset captures wide catalog diversity across electronics, apparel, furniture, hardware, and groceries:

| Category | Image Count | % of Dataset |
| :--- | :--- | :--- |
| `CELLULAR_PHONE_CASE` | 21,212 | 39.53% |
| `SHOES` | 5,791 | 10.79% |
| `GROCERY` | 2,659 | 4.96% |
| `HOME` | 1,858 | 3.46% |
| `HOME_BED_AND_BATH` | 1,059 | 1.97% |
| `HOME_FURNITURE_AND_DECOR` | 973 | 1.81% |
| `CHAIR` | 969 | 1.81% |
| `SANDAL` | 960 | 1.79% |
| `OTHER` | 831 | 1.55% |
| `BOOT` | 801 | 1.49% |
| `HEALTH_PERSONAL_CARE` | 640 | 1.19% |
| `SOFA` | 590 | 1.10% |
| `TABLE` | 442 | 0.82% |
| `OFFICE_PRODUCTS` | 425 | 0.79% |
| `PET_SUPPLIES` | 413 | 0.77% |
| `HARDWARE_HANDLE` | 367 | 0.68% |
| `HANDBAG` | 355 | 0.66% |
| `SPORTING_GOODS` | 330 | 0.62% |
| `RUG` | 321 | 0.60% |
| `LIGHT_BULB` | 304 | 0.57% |

---

## 4. Sample Parsed Manifest Records

```text
[1] SKU: B07P8ML82R | Category: HARDWARE | Brand: AmazonBasics
    Title : 22" Bottom Mount Drawer Slides, White Powder Coat, 10 Pairs
    Image : ...\data\raw\small\9f\9f76d27b.jpg (exists=True, is_main=True)

[2] SKU: B07P8ML82R | Category: HARDWARE | Brand: AmazonBasics
    Title : 22" Bottom Mount Drawer Slides, White Powder Coat, 10 Pairs
    Image : ...\data\raw\small\12\12c8a5f8.jpg (exists=True, is_main=False)

[3] SKU: B07P8ML82R | Category: HARDWARE | Brand: AmazonBasics
    Title : 22" Bottom Mount Drawer Slides, White Powder Coat, 10 Pairs
    Image : ...\data\raw\small\0c\0c8168ef.jpg (exists=True, is_main=False)

[4] SKU: B075DQBBJZ | Category: HOME | Brand: Rivet
    Title : Arizona Desert Sand Horizon Photo with Wood Hanger
    Image : ...\data\raw\small\c6\c6889ed4.jpg (exists=True, is_main=True)

[5] SKU: B075DQBBJZ | Category: HOME | Brand: Rivet
    Title : Arizona Desert Sand Horizon Photo with Wood Hanger
    Image : ...\data\raw\small\43\4378ccb0.jpg (exists=True, is_main=False)
```

---

## 5. Core Source Modules Implemented

### 1. `src/data/abo_loader.py`
- **Purpose**: Ingestion, JSON schema extraction, image path mapping, stratification, and dataset splitting.
- **Key Functions**:
  - `ABOCatalogLoader.load_manifest()`: Parses 16 gzip JSONL shards into a unified DataFrame.
  - `split_by_sku(manifest_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)`: Enforces SKU disjointness.
  - `save_manifests_and_stats()`: Saves processed CSVs and distribution statistics.

### 2. `src/data/augmentations.py`
- **`ImageAugmenter`**:
  - Horizontal flipping ($p=0.5$).
  - Color jitter (brightness, contrast, saturation, hue $p=0.4$).
  - JPEG compression artifact simulation ($p=0.3$, quality 40–80).
  - Gaussian blur/noise ($p=0.2$) & Cutout / Random Erasing ($p=0.2$).
- **`TextAugmenter`**:
  - QWERTY keyboard adjacency typo injector (e.g. `'o'` $\to$ `'i'` or `'p'`).
  - Character swap & letter deletion.
  - Brand name dropout ($p=0.2$) to prevent trivial shortcut learning.
  - Random tail truncation & whitespace perturbation.

### 3. `src/data/pair_sampler.py`
- **`PairSampler`**:
  - Positive Pairs: Returns multiple perspectives of the same SKU or an augmented duplicate if single-image.
  - Hard Negatives: Mines non-identical SKUs sharing the exact same product category.
- **`TripletSampler`**:
  - Constructs `(Anchor, Positive, Negative)` tuples with in-category negative balancing.

### 4. `src/data/dataset.py`
- **`FusionMatchDataset`**: PyTorch `Dataset` loading images as RGB tensors, tokenizing titles, and computing quality proxies.
- **`compute_image_quality_proxy`**: Laplacian kernel variance for blur detection ($\sigma_{\text{Laplacian}}^2 \in [0, 1]$).
- **`compute_text_quality_proxy`**: Token length score assessing informational density ($S_{\text{text}} \in [0, 1]$).
- **`collate_fusion_match_batch`**: Padded tensor batch collation for PyTorch DataLoader.

---

## 6. Unit Testing & Verification Results

All unit tests were executed with Pytest against the real manifests and dataset components:

```bash
.venv\Scripts\pytest.exe tests/test_data_pipeline.py -v
```

### Test Suite Output
```text
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\HP\Desktop\Project\Deduplication\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\HP\Desktop\Project\Deduplication
plugins: anyio-4.14.2
collected 7 items

tests/test_data_pipeline.py::test_sku_split_strictly_disjoint PASSED     [ 14%]
tests/test_data_pipeline.py::test_real_manifest_files_no_leakage_if_present PASSED [ 28%]
tests/test_data_pipeline.py::test_image_augmenter PASSED                 [ 42%]
tests/test_data_pipeline.py::test_text_augmenter PASSED                  [ 57%]
tests/test_data_pipeline.py::test_quality_proxies PASSED                 [ 71%]
tests/test_data_pipeline.py::test_pair_sampler PASSED                    [ 85%]
tests/test_data_pipeline.py::test_dataset_and_dataloader PASSED          [100%]

============================== 7 passed in 6.60s ==============================
```

---

## 7. Next Steps: Phase 2 (Model Architecture & Dual-Encoder Training)

With Phase 1 complete, the data pipeline is ready for **Phase 2**:
1. **Vision-Language Dual Encoder (`src/model/`)**:
   - Initialize Google SigLIP (`google/siglip-base-patch16-224` or `google/siglip-so400m-patch14-384`).
   - Implement Multi-View Mean-Pooling across angle embeddings.
2. **Quality-Weighted Adaptive Fusion Layer**:
   - Weight combination: $v_{\text{prod}} = \text{Normalize}(\alpha \cdot v_{\text{img}} + (1-\alpha) \cdot v_{\text{text}})$ where $\alpha = f(q_{\text{img}}, q_{\text{text}})$.
3. **Contrastive Loss & Training Engine (`src/train/`)**:
   - InfoNCE / Multiple Negatives Ranking Loss with temperature scaling.
   - Mixed precision training (FP16/BF16) optimized for Colab T4 GPU.
