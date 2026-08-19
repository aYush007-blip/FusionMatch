"""Evaluation Metrics for Cross-Modal Product Deduplication."""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_pairwise_f1(
    embeddings: Union[torch.Tensor, np.ndarray],
    sku_ids: List[str],
    threshold: float = 0.70,
) -> float:
    """Computes pairwise duplicate F1-score across all (i, j) pairs in catalog.
    
    Args:
        embeddings: Tensor or array of shape (N, D), L2-normalized.
        sku_ids: List of N string SKU identifiers.
        threshold: Cosine similarity cutoff for predicting duplicate status.
        
    Returns:
        float: Pairwise F1 score in range [0.0, 1.0].
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    embeddings = embeddings.float()
    # Normalize if not already unit norm
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    n = len(sku_ids)
    if n <= 1:
        return 0.0

    # Compute pairwise similarity matrix: (N, N)
    sims = (embeddings @ embeddings.T).cpu().numpy()

    # Extract upper triangle indices (i < j)
    triu_indices = np.triu_indices(n, k=1)
    sim_pairs = sims[triu_indices]

    sku_arr = np.array(sku_ids)
    y_true = (sku_arr[triu_indices[0]] == sku_arr[triu_indices[1]]).astype(np.int32)
    y_pred = (sim_pairs >= threshold).astype(np.int32)

    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_precision_recall_at_k(
    embeddings: Union[torch.Tensor, np.ndarray],
    sku_ids: List[str],
    k: int = 5,
) -> Tuple[float, float]:
    """Computes Mean Precision@K and Mean Recall@K for catalog duplicate retrieval.
    
    Args:
        embeddings: Tensor or array of shape (N, D), L2-normalized.
        sku_ids: List of N string SKU identifiers.
        k: Top-K neighbors retrieved per item.
        
    Returns:
        Tuple of (precision_at_k, recall_at_k).
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    embeddings = embeddings.float()
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    n = len(sku_ids)
    if n <= 1:
        return 0.0, 0.0

    sims = (embeddings @ embeddings.T).clone()
    sims.fill_diagonal_(-1e4)  # Exclude self from neighbor candidates

    sku_arr = np.array(sku_ids)
    precisions = []
    recalls = []

    actual_k = min(k, n - 1)

    for i in range(n):
        # Total positive duplicates for item i in the catalog (excluding itself)
        true_positives_avail = int((sku_arr == sku_arr[i]).sum() - 1)
        if true_positives_avail == 0:
            # Query has no true duplicates in the split
            continue

        topk_idx = torch.topk(sims[i], k=actual_k).indices.cpu().numpy()
        hits = int((sku_arr[topk_idx] == sku_arr[i]).sum())

        precisions.append(hits / float(k))
        recalls.append(hits / float(min(true_positives_avail, k)))

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))


def evaluate_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    sku_ids: List[str],
    k_values: Tuple[int, ...] = (1, 5, 10),
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """Runs a full suite of retrieval and clustering metrics on embedding vectors."""
    pairwise_f1 = compute_pairwise_f1(embeddings, sku_ids, threshold=threshold)
    
    results = {
        "pairwise_f1": pairwise_f1,
        "threshold": threshold,
        "total_items": len(sku_ids),
        "unique_skus": len(set(sku_ids)),
    }

    for k in k_values:
        p_k, r_k = compute_precision_recall_at_k(embeddings, sku_ids, k=k)
        results[f"precision@{k}"] = p_k
        results[f"recall@{k}"] = r_k

    return results
