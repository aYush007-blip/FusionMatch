"""PyTorch Dataset and DataLoader Wrappers for FusionMatch Data Pipeline.

Provides FusionMatchDataset for loading multimodal anchor-positive pairs,
extracting quality proxies (blur/entropy and text length), and formatting
tensors for SigLIP dual encoder training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augmentations import apply_multimodal_augmentations, get_image_augmentations, get_text_augmentations
from .pair_sampler import PairSampler


def compute_image_quality_proxy(image: Image.Image) -> float:
    """Calculates image quality proxy score in [0.0, 1.0].

    Uses normalized Laplacian variance (sharpness/blur proxy) and resolution factor.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    # 3x3 Laplacian filter kernel
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.5

    # Compute fast discrete Laplacian variance
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    lap_var = float(lap.var())
    blur_score = min(lap_var / 500.0, 1.0)
    res_score = min((w * h) / (256.0 * 256.0), 1.0)
    quality = 0.7 * blur_score + 0.3 * res_score
    return float(np.clip(quality, 0.0, 1.0))


def compute_text_quality_proxy(text: str, max_tokens: int = 20) -> float:
    """Calculates text quality proxy score in [0.0, 1.0] based on length & completeness."""
    if not text or not text.strip():
        return 0.0
    words = text.strip().split()
    score = min(len(words) / float(max_tokens), 1.0)
    return float(np.clip(score, 0.0, 1.0))


def preprocess_image_tensor(image: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Resizes and normalizes PIL Image to PyTorch (3, H, W) float tensor in [-1, 1]."""
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0  # [0, 1]
    # Normalize with standard SigLIP / ImageNet mean & std
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm).permute(2, 0, 1).float()  # (3, H, W)
    return tensor


class SimpleTokenizer:
    """Lightweight fallback whitespace/hash tokenizer if HF transformers is not initialized."""

    def __init__(self, max_length: int = 64) -> None:
        self.max_length = max_length

    def __call__(
        self,
        texts: List[str] | str,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max_length or self.max_length
        input_ids_list = []
        attention_mask_list = []

        for t in texts:
            words = t.strip().lower().split() if t else []
            # Hash words to 1..30000 range for token IDs
            ids = [((hash(w) % 30000) + 1) for w in words[:max_len]]
            mask = [1] * len(ids)

            # Pad
            pad_len = max_len - len(ids)
            ids.extend([0] * pad_len)
            mask.extend([0] * pad_len)

            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }


class FusionMatchDataset(Dataset):
    """PyTorch Dataset yielding multimodal positive pairs for FusionMatch training/eval.

    For each SKU, samples/generates an anchor item and a positive item
    (either multi-angle photo or augmented duplicate), returning image tensors,
    tokenized text IDs, and quality proxy scores.
    """

    def __init__(
        self,
        manifest: Optional[Union[pd.DataFrame, str, Path]] = None,
        manifest_path: Optional[Union[pd.DataFrame, str, Path]] = None,
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        split: str = "train",
        is_training: Optional[bool] = None,
        img_size: int = 224,
        max_text_len: int = 64,
        cache_dir: Optional[str | Path] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        target_manifest = manifest if manifest is not None else manifest_path
        if target_manifest is None:
            raise ValueError("Must provide either 'manifest' or 'manifest_path' to FusionMatchDataset.")

        if is_training is not None:
            split = "train" if is_training else "val"

        if isinstance(target_manifest, (str, Path)):
            self.manifest = pd.read_csv(target_manifest)
        else:
            self.manifest = target_manifest.copy().reset_index(drop=True)

        self.processor = processor
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_text_len)
        self.split = split
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sampler = PairSampler(self.manifest, seed=seed)
        self.unique_skus = self.sampler.unique_skus
        self.img_augmenter = get_image_augmentations(split=split, img_size=img_size)
        self.txt_augmenter = get_text_augmentations(split=split)

    def __len__(self) -> int:
        return len(self.unique_skus)

    def _load_image(self, image_path: str) -> Image.Image:
        """Safely loads a PIL Image with RGB conversion and fallback dummy if missing."""
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                return img
            except Exception:
                pass
        # Fallback dummy RGB image (neutral gray) if image missing during testing
        return Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

    def _process_item(
        self,
        image_path: str,
        title: str,
        brand: str,
        is_augmented: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Loads image and text, applies augmentations if needed, computes quality scores, and returns tensors."""
        img = self._load_image(image_path)
        txt = title or ""

        # Compute quality proxies before or after augmentations
        q_v = compute_image_quality_proxy(img)
        q_t = compute_text_quality_proxy(txt)

        if is_augmented and self.split == "train":
            img = self.img_augmenter.augment(img)
            txt = self.txt_augmenter.augment(txt, brand=brand)

        # Process image tensor
        if self.processor is not None and hasattr(self.processor, "image_processor"):
            pixel_values = self.processor.image_processor(
                img, return_tensors="pt"
            ).pixel_values.squeeze(0)
        else:
            pixel_values = preprocess_image_tensor(img, self.img_size)

        # Process text tokens
        if self.processor is not None and hasattr(self.processor, "tokenizer"):
            token_out = self.processor.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = token_out.input_ids.squeeze(0)
            attention_mask = token_out.attention_mask.squeeze(0)
        elif callable(self.tokenizer):
            token_out = self.tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = token_out["input_ids"].squeeze(0)
            attention_mask = token_out["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return pixel_values, input_ids, attention_mask, q_v, q_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sku_id = self.unique_skus[idx]
        anc_rec, pos_rec, is_multi_angle = self.sampler.sample_positive_pair(sku_id)

        # Anchor item
        anc_pv, anc_ids, anc_mask, anc_qv, anc_qt = self._process_item(
            anc_rec["image_path"],
            anc_rec.get("title", ""),
            anc_rec.get("brand", ""),
            is_augmented=False,
        )

        # Positive item (augmented if single image, or multi-angle)
        pos_pv, pos_ids, pos_mask, pos_qv, pos_qt = self._process_item(
            pos_rec["image_path"],
            pos_rec.get("title", ""),
            pos_rec.get("brand", ""),
            is_augmented=not is_multi_angle,
        )

        return {
            "anchor_sku_id": sku_id,
            "anchor_pixel_values": anc_pv,
            "anchor_input_ids": anc_ids,
            "anchor_attention_mask": anc_mask,
            "anchor_q_v": torch.tensor(anc_qv, dtype=torch.float32),
            "anchor_q_t": torch.tensor(anc_qt, dtype=torch.float32),
            "anchor_category": anc_rec.get("category", "OTHER"),
            "positive_sku_id": sku_id,
            "positive_pixel_values": pos_pv,
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_mask,
            "positive_q_v": torch.tensor(pos_qv, dtype=torch.float32),
            "positive_q_t": torch.tensor(pos_qt, dtype=torch.float32),
            "positive_category": pos_rec.get("category", "OTHER"),
            "is_multi_angle": is_multi_angle,
        }


def collate_fusion_match_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of sample dictionaries into batched tensors."""
    return {
        "anchor_sku_id": [item["anchor_sku_id"] for item in batch],
        "anchor_pixel_values": torch.stack([item["anchor_pixel_values"] for item in batch]),
        "anchor_input_ids": torch.stack([item["anchor_input_ids"] for item in batch]),
        "anchor_attention_mask": torch.stack([item["anchor_attention_mask"] for item in batch]),
        "anchor_q_v": torch.stack([item["anchor_q_v"] for item in batch]),
        "anchor_q_t": torch.stack([item["anchor_q_t"] for item in batch]),
        "anchor_category": [item["anchor_category"] for item in batch],
        "positive_sku_id": [item["positive_sku_id"] for item in batch],
        "positive_pixel_values": torch.stack([item["positive_pixel_values"] for item in batch]),
        "positive_input_ids": torch.stack([item["positive_input_ids"] for item in batch]),
        "positive_attention_mask": torch.stack([item["positive_attention_mask"] for item in batch]),
        "positive_q_v": torch.stack([item["positive_q_v"] for item in batch]),
        "positive_q_t": torch.stack([item["positive_q_t"] for item in batch]),
        "positive_category": [item["positive_category"] for item in batch],
        "is_multi_angle": [item["is_multi_angle"] for item in batch],
    }


def build_dataloaders(
    train_manifest: pd.DataFrame | str | Path,
    val_manifest: pd.DataFrame | str | Path,
    test_manifest: Optional[pd.DataFrame | str | Path] = None,
    processor: Optional[Any] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 256,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Builds PyTorch DataLoaders for train, val, and optional test splits."""
    train_ds = FusionMatchDataset(train_manifest, processor=processor, split="train", img_size=img_size)
    val_ds = FusionMatchDataset(val_manifest, processor=processor, split="val", img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fusion_match_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fusion_match_batch,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_manifest is not None:
        test_ds = FusionMatchDataset(test_manifest, processor=processor, split="test", img_size=img_size)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fusion_match_batch,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader
