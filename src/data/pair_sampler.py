"""Pair and Triplet Sampler for FusionMatch Contrastive Learning.

Constructs positive pairs (multi-angle photography or multimodal augmented views)
and hard negative pairs (different SKUs in the same fine-grained category).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .augmentations import apply_multimodal_augmentations


class PairSampler:
    """Samples multimodal positive and hard-negative pairs from catalog manifests.

    Attributes:
        manifest: DataFrame with columns [sku_id, image_path, title, brand, category].
        augment_single_image_skus: If True, uses augmentation to generate positive pairs
                                   for SKUs having only 1 image.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        augment_single_image_skus: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        self.manifest = manifest.copy().reset_index(drop=True)
        self.augment_single_image_skus = augment_single_image_skus
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Build fast index lookups
        self._build_indices()

    def _build_indices(self) -> None:
        """Constructs fast SKU and category indexing dictionaries."""
        self.sku_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.category_to_skus: Dict[str, List[str]] = defaultdict(list)
        self.unique_skus: List[str] = list(self.manifest["sku_id"].unique())

        for idx, row in self.manifest.iterrows():
            sku = row["sku_id"]
            cat = row["category"]
            self.sku_to_indices[sku].append(idx)

        for sku, idxs in self.sku_to_indices.items():
            cat = self.manifest.loc[idxs[0], "category"]
            self.category_to_skus[cat].append(sku)

        self.multi_image_skus = [
            sku for sku, idxs in self.sku_to_indices.items() if len(idxs) >= 2
        ]
        self.single_image_skus = [
            sku for sku, idxs in self.sku_to_indices.items() if len(idxs) == 1
        ]

    def sample_positive_pair(self, sku_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Samples or generates a positive pair for a given SKU.

        Returns:
            (anchor_record, positive_record, is_multi_angle_pair)
        """
        indices = self.sku_to_indices[sku_id]
        
        if len(indices) >= 2:
            # Pick two distinct images of the same SKU
            idx1, idx2 = self.rng.sample(indices, 2)
            rec1 = self.manifest.iloc[idx1].to_dict()
            rec2 = self.manifest.iloc[idx2].to_dict()
            return rec1, rec2, True
        else:
            # Single image SKU -> anchor + augmented duplicate
            idx = indices[0]
            rec1 = self.manifest.iloc[idx].to_dict()
            # Positive is the same item record (will be perturbed by augmentations)
            rec2 = dict(rec1)
            return rec1, rec2, False

    def sample_hard_negative(self, anchor_sku_id: str) -> Dict[str, Any]:
        """Samples an in-category hard negative SKU (same category, different SKU)."""
        anchor_idx = self.sku_to_indices[anchor_sku_id][0]
        anchor_cat = self.manifest.iloc[anchor_idx]["category"]
        candidate_skus = self.category_to_skus.get(anchor_cat, [])

        # Filter out the anchor SKU itself
        valid_candidates = [s for s in candidate_skus if s != anchor_sku_id]

        if valid_candidates:
            neg_sku = self.rng.choice(valid_candidates)
        else:
            # Fallback to random SKU across catalog
            other_skus = [s for s in self.unique_skus if s != anchor_sku_id]
            neg_sku = self.rng.choice(other_skus)

        neg_idx = self.rng.choice(self.sku_to_indices[neg_sku])
        return self.manifest.iloc[neg_idx].to_dict()

    def sample_triplet(self, sku_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Samples an (anchor, positive, hard_negative) record triplet."""
        if not self.unique_skus:
            raise ValueError("Cannot sample triplet from an empty manifest.")
        if sku_id is None:
            sku_id = self.rng.choice(self.unique_skus)

        anchor_rec, pos_rec, is_multi_angle = self.sample_positive_pair(sku_id)
        neg_rec = self.sample_hard_negative(sku_id)
        return anchor_rec, pos_rec, neg_rec

    def sample_batch_records(self, batch_size: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Samples a batch of positive pairs for contrastive learning."""
        if not self.unique_skus:
            return []
        sampled_skus = self.rng.choices(self.unique_skus, k=batch_size)
        batch = []
        for sku in sampled_skus:
            anc, pos, _ = self.sample_positive_pair(sku)
            batch.append((anc, pos))
        return batch


class TripletSampler(PairSampler):
    """Specialized sampler for metric learning triplet evaluation."""

    def __iter__(self):
        while True:
            yield self.sample_triplet()
