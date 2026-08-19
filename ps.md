*Project Title: Cross-Modal & Multi-View Product Deduplication Engine for E-Commerce Catalogs*

*Background & Business Context:*
Online marketplaces frequently suffer from catalog clutter caused by multi-merchant listings. Different sellers upload the exact same physical product using conflicting product titles, varied descriptions, and diverse photographic perspectives (e.g., front, side, packaging, zoom-in). Traditional text-only deduplication (Levenshtein distance, TF-IDF, BM25) fails due to adversarial or unstandardized naming conventions. Standard unimodal computer vision approaches (ResNet embeddings) fail because identical items photographed under different lighting, angles, or backgrounds diverge in pixel space.

*Problem Statement:*
Develop a lightweight, local-first deduplication system capable of:
    Aggregating multi-angle product photography into a single representative visual embedding using mean-pooled representations.

    Aligning noisy textual attributes with visual features in a shared latent space via a pre-trained Vision-Language Model (SigLIP).

    Performing vector indexing and nearest-neighbor retrieval across thousands of catalog items with sub-50ms query latency.

    Establishing an automated duplicate-classification threshold using pairwise cosine similarity and evaluating performance using Precision, Recall, and F1-score.

*Success Metrics:*
    Retrieval Quality: Precision@K, Pairwise F1-Score on ground-truth duplicate clusters.
    Latency & Footprint: Sub-50ms inference/search latency on CPU; total vector index storage <100 MB a 10,000-SKU catalog.

