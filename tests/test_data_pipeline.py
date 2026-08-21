"""Unit Tests for FusionMatch Data Pipeline (§10.1).

Covers manifest integrity, SKU leakage prevention, augmentation behavior,
pair sampling logic, quality proxy validity, and Dataset/DataLoader tensor collation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.abo_loader import ABOCatalogLoader, split_by_sku, save_manifests_and_stats
from src.data.augmentations import (
    ImageAugmenter,
    TextAugmenter,
    apply_multimodal_augmentations,
    get_image_augmentations,
    get_text_augmentations,
)
from src.data.dataset import (
    FusionMatchDataset,
    build_dataloaders,
    compute_image_quality_proxy,
    compute_text_quality_proxy,
    preprocess_image_tensor,
)
from src.data.pair_sampler import PairSampler, TripletSampler


@pytest.fixture
def mock_manifest() -> pd.DataFrame:
    """Creates a controlled synthetic manifest with multi-image and single-image SKUs."""
    records = [
        # SKU 1: 3 images (multi-angle) in Category A
        {"sku_id": "SKU_001", "image_path": "data/mock/sku1_1.jpg", "title": "Wireless Bluetooth Headphones Black", "brand": "AudioPro", "category": "HEADPHONES"},
        {"sku_id": "SKU_001", "image_path": "data/mock/sku1_2.jpg", "title": "Wireless Bluetooth Headphones Black", "brand": "AudioPro", "category": "HEADPHONES"},
        {"sku_id": "SKU_001", "image_path": "data/mock/sku1_3.jpg", "title": "Wireless Bluetooth Headphones Black", "brand": "AudioPro", "category": "HEADPHONES"},
        # SKU 2: 2 images in Category A
        {"sku_id": "SKU_002", "image_path": "data/mock/sku2_1.jpg", "title": "Over-Ear Noise Cancelling Headset", "brand": "SoundMax", "category": "HEADPHONES"},
        {"sku_id": "SKU_002", "image_path": "data/mock/sku2_2.jpg", "title": "Over-Ear Noise Cancelling Headset", "brand": "SoundMax", "category": "HEADPHONES"},
        # SKU 3: 1 image in Category A
        {"sku_id": "SKU_003", "image_path": "data/mock/sku3_1.jpg", "title": "Sport In-Ear Earbuds", "brand": "BeatFit", "category": "HEADPHONES"},
        # SKU 4: 2 images in Category B
        {"sku_id": "SKU_004", "image_path": "data/mock/sku4_1.jpg", "title": "Stainless Steel Water Bottle 750ml", "brand": "HydroPeak", "category": "KITCHEN"},
        {"sku_id": "SKU_004", "image_path": "data/mock/sku4_2.jpg", "title": "Stainless Steel Water Bottle 750ml", "brand": "HydroPeak", "category": "KITCHEN"},
        # SKU 5: 2 images in Category B
        {"sku_id": "SKU_005", "image_path": "data/mock/sku5_1.jpg", "title": "Insulated Thermal Coffee Mug", "brand": "HydroPeak", "category": "KITCHEN"},
        {"sku_id": "SKU_005", "image_path": "data/mock/sku5_2.jpg", "title": "Insulated Thermal Coffee Mug", "brand": "HydroPeak", "category": "KITCHEN"},
        # SKU 6: 2 images in Category C
        {"sku_id": "SKU_006", "image_path": "data/mock/sku6_1.jpg", "title": "Men Running Shoes Mesh Lightweight", "brand": "SpeedRun", "category": "SHOES"},
        {"sku_id": "SKU_006", "image_path": "data/mock/sku6_2.jpg", "title": "Men Running Shoes Mesh Lightweight", "brand": "SpeedRun", "category": "SHOES"},
    ]
    return pd.DataFrame(records)


def test_sku_split_strictly_disjoint(mock_manifest):
    """Asserts zero SKU overlap between train, val, and test partitions."""
    train_df, val_df, test_df = split_by_sku(mock_manifest, seed=42, ratios=(0.5, 0.25, 0.25))

    train_skus = set(train_df["sku_id"])
    val_skus = set(val_df["sku_id"])
    test_skus = set(test_df["sku_id"])

    assert len(train_skus) > 0
    assert len(val_skus) > 0
    assert len(test_skus) > 0

    assert train_skus.isdisjoint(val_skus), f"Leakage found between Train and Val: {train_skus & val_skus}"
    assert train_skus.isdisjoint(test_skus), f"Leakage found between Train and Test: {train_skus & test_skus}"
    assert val_skus.isdisjoint(test_skus), f"Leakage found between Val and Test: {val_skus & test_skus}"


def test_real_manifest_files_no_leakage_if_present():
    """If real manifest files exist on disk, verifies disjointness against real files."""
    train_file = Path("data/processed/manifest_train.csv")
    val_file = Path("data/processed/manifest_val.csv")
    test_file = Path("data/processed/manifest_test.csv")

    if train_file.exists() and val_file.exists() and test_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        train_skus = set(train_df["sku_id"])
        val_skus = set(val_df["sku_id"])
        test_skus = set(test_df["sku_id"])

        assert train_skus.isdisjoint(val_skus), "SKU leakage detected between manifest_train and manifest_val!"
        assert train_skus.isdisjoint(test_skus), "SKU leakage detected between manifest_train and manifest_test!"
        assert val_skus.isdisjoint(test_skus), "SKU leakage detected between manifest_val and manifest_test!"


def test_image_augmenter():
    """Validates image augmentation pipeline transformations and output format."""
    dummy_img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    augmenter = ImageAugmenter(img_size=256, seed=42)
    aug_img = augmenter.augment(dummy_img)

    assert isinstance(aug_img, Image.Image)
    assert aug_img.size == (256, 256)
    assert aug_img.mode == "RGB"

    # Tensor preprocessing check
    tensor = preprocess_image_tensor(aug_img, img_size=256)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32


def test_text_augmenter():
    """Validates text perturbations: typos, dropouts, swaps, and truncations."""
    original_text = "Apple iPhone 15 Pro Max 256GB Natural Titanium Unlocked"
    augmenter = TextAugmenter(seed=42)

    # Test typo injection
    typo_text = augmenter.inject_typo(original_text)
    assert isinstance(typo_text, str)

    # Test word dropout with brand
    dropped_text = augmenter.drop_words(original_text, brand="Apple")
    assert isinstance(dropped_text, str)
    assert len(dropped_text.split()) <= len(original_text.split())

    # Test full augmentation pipeline
    aug_text = augmenter.augment(original_text, brand="Apple")
    assert isinstance(aug_text, str)


def test_quality_proxies():
    """Validates heuristic quality proxy score ranges and edge cases."""
    # Sharp image vs blurry solid image
    sharp_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    # Draw some high contrast pattern
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noisy_img = Image.fromarray(arr)

    q_v_noisy = compute_image_quality_proxy(noisy_img)
    q_v_solid = compute_image_quality_proxy(sharp_img)

    assert 0.0 <= q_v_noisy <= 1.0
    assert 0.0 <= q_v_solid <= 1.0
    assert q_v_noisy >= q_v_solid

    # Text quality proxy
    q_t_full = compute_text_quality_proxy("A comprehensive detailed product listing with ten words or more")
    q_t_short = compute_text_quality_proxy("Short")
    q_t_empty = compute_text_quality_proxy("")

    assert q_t_full > q_t_short
    assert q_t_empty == 0.0
    assert 0.0 <= q_t_full <= 1.0


def test_pair_sampler(mock_manifest):
    """Validates positive and hard-negative pair sampling behavior."""
    sampler = PairSampler(mock_manifest, seed=42)

    # Multi-angle SKU positive pair
    anc, pos, is_multi = sampler.sample_positive_pair("SKU_001")
    assert is_multi is True
    assert anc["sku_id"] == pos["sku_id"] == "SKU_001"
    assert anc["image_path"] != pos["image_path"]

    # Single-image SKU positive pair
    anc_s, pos_s, is_multi_s = sampler.sample_positive_pair("SKU_003")
    assert is_multi_s is False
    assert anc_s["sku_id"] == pos_s["sku_id"] == "SKU_003"
    assert anc_s["image_path"] == pos_s["image_path"]

    # Hard negative sampling (same category HEADPHONES, different SKU)
    neg = sampler.sample_hard_negative("SKU_001")
    assert neg["sku_id"] != "SKU_001"
    assert neg["category"] == "HEADPHONES"


def test_dataset_and_dataloader(mock_manifest):
    """Tests FusionMatchDataset and DataLoader batch generation."""
    dataset = FusionMatchDataset(mock_manifest, split="train", img_size=256, max_text_len=32)
    assert len(dataset) == mock_manifest["sku_id"].nunique()

    sample = dataset[0]
    assert "anchor_pixel_values" in sample
    assert "positive_pixel_values" in sample
    assert sample["anchor_pixel_values"].shape == (3, 256, 256)
    assert sample["positive_pixel_values"].shape == (3, 256, 256)
    assert sample["anchor_input_ids"].shape == (32,)
    assert sample["positive_input_ids"].shape == (32,)
    assert 0.0 <= sample["anchor_q_v"].item() <= 1.0
    assert 0.0 <= sample["anchor_q_t"].item() <= 1.0

    # Test DataLoader batching
    train_loader, val_loader, _ = build_dataloaders(
        mock_manifest, mock_manifest, batch_size=4, num_workers=0
    )
    batch = next(iter(train_loader))
    assert batch["anchor_pixel_values"].shape == (4, 3, 256, 256)
    assert batch["positive_pixel_values"].shape == (4, 3, 256, 256)
    assert batch["anchor_input_ids"].shape == (4, 64)
    assert len(batch["anchor_sku_id"]) == 4


def test_single_category_split_no_crash():
    """Verify split_by_sku works gracefully when all SKUs belong to a single category."""
    single_cat_df = pd.DataFrame([
        {"sku_id": f"SKU_{i}", "image_path": f"img_{i}.jpg", "title": f"Item {i}", "brand": "B", "category": "ONLY_ONE"}
        for i in range(10)
    ])
    train_df, val_df, test_df = split_by_sku(single_cat_df, seed=42, ratios=(0.6, 0.2, 0.2))
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert set(train_df["sku_id"]).isdisjoint(set(val_df["sku_id"]))


def test_grayscale_and_2d_cutout():
    """Verify ImageAugmenter handles 2D grayscale image arrays without dimension error."""
    gray_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    augmenter = ImageAugmenter(cutout_prob=1.0, seed=42)
    cutout_img = augmenter.apply_cutout(Image.fromarray(gray_arr))
    assert isinstance(cutout_img, Image.Image)

