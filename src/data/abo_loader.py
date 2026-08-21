"""ABO Catalog Loader for FusionMatch Data Pipeline.

Handles downloading, extracting, parsing metadata, resolving image paths,
stratified sampling, and SKU-level train/val/test splitting for the Amazon Berkeley Objects dataset.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ABO_LISTINGS_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
ABO_IMAGES_SMALL_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"


def download_file(url: str, dest_path: Path | str, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with progress reporting and resume capability."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"File already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    temp_dest = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}...")
    
    with urllib.request.urlopen(url) as response, open(temp_dest, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

    shutil.move(str(temp_dest), str(dest))
    print(f"Download complete: {dest}")
    return dest


def extract_tar(tar_path: Path | str, extract_dir: Path | str) -> Path:
    """Extract a tar archive safely with directory preservation."""
    tar_p = Path(tar_path)
    ext_d = Path(extract_dir)
    ext_d.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_p.name} to {ext_d}...")
    with tarfile.open(tar_p, "r:*") as tar:
        try:
            tar.extractall(path=ext_d, filter="data")
        except TypeError:
            tar.extractall(path=ext_d)
    print(f"Extraction finished: {ext_d}")
    return ext_d


class ABOCatalogLoader:
    """Loads and preprocesses the Amazon Berkeley Objects (ABO) dataset for FusionMatch.

    Attributes:
        images_root: Path to extracted abo-images-small directory.
        listings_root: Path to extracted abo-listings directory.
        max_skus: Target number of SKUs to sample (~10,000 - 11,000).
        min_images_per_sku: Minimum number of images per SKU to retain.
    """

    def __init__(
        self,
        images_root: str | Path = "data/raw/abo-images-small",
        listings_root: str | Path = "data/raw/abo-listings",
        data_dir: Optional[str | Path] = None,
        max_skus: int = 11000,
        min_images_per_sku: int = 1,
        seed: int = 42,
    ) -> None:
        if data_dir is not None:
            d_path = Path(data_dir)
            self.images_root = d_path if (d_path / "small").exists() or (d_path / "images").exists() else (d_path / "abo-images-small" if (d_path / "abo-images-small").exists() else d_path)
            self.listings_root = d_path if (d_path / "listings").exists() else (d_path / "abo-listings" if (d_path / "abo-listings").exists() else d_path)
        else:
            self.images_root = Path(images_root)
            self.listings_root = Path(listings_root)
        self.max_skus = max_skus
        self.min_images_per_sku = min_images_per_sku
        self.seed = seed
        self._image_path_map: Optional[Dict[str, str]] = None

    def _find_image_metadata_csv(self) -> Path:
        """Locate images.csv.gz or images.csv in images_root or listings_root or parent raw directory."""
        candidates = [
            self.images_root / "images" / "metadata" / "images.csv.gz",
            self.images_root / "images" / "metadata" / "images.csv",
            self.images_root / "metadata" / "images.csv.gz",
            self.images_root / "metadata" / "images.csv",
            self.images_root / "images.csv.gz",
            self.images_root / "images.csv",
            self.images_root.parent / "metadata" / "images.csv.gz",
            self.images_root.parent / "metadata" / "images.csv",
            self.images_root.parent / "images.csv",
            self.listings_root / "listings" / "metadata" / "images.csv.gz",
            self.listings_root / "metadata" / "images.csv.gz",
            Path("data/raw/metadata/images.csv"),
            Path("data/raw/images.csv"),
        ]
        for c in candidates:
            if c.exists():
                return c

        # Recursive search if not found in standard paths
        found = list(self.images_root.rglob("images.csv*")) or list(self.images_root.parent.rglob("images.csv*"))
        if found:
            return found[0]

        raise FileNotFoundError(
            f"Could not find images.csv inside {self.images_root} or {self.listings_root}. "
            "Please ensure abo-images-small archive is extracted."
        )

    def _load_image_metadata_csv(self) -> Dict[str, str]:
        """Loads mapping from image_id to resolved absolute image path."""
        if self._image_path_map is not None:
            return self._image_path_map

        meta_csv = self._find_image_metadata_csv()
        print(f"Loading image index from {meta_csv}...", flush=True)

        # Detect base image directory once
        base_dir = None
        for candidate in [
            self.images_root / "images" / "small",
            self.images_root / "small",
            self.images_root.parent / "small",
            Path("data/raw/small"),
            self.images_root,
        ]:
            if candidate.exists() and candidate.is_dir():
                # Verify with sample subfolder
                if (candidate / "00").exists() or (candidate / "images" / "small").exists():
                    base_dir = candidate
                    break
                elif base_dir is None:
                    base_dir = candidate

        if base_dir is None:
            base_dir = self.images_root / "small"

        base_dir_str = str(base_dir.resolve())
        print(f"Resolved base image directory: {base_dir_str}", flush=True)
        
        mapping: Dict[str, str] = {}
        is_gz = meta_csv.name.endswith(".gz")
        open_fn = gzip.open if is_gz else open

        with open_fn(meta_csv, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                img_id = row.get("image_id")
                rel_path = row.get("path")
                if not img_id or not rel_path:
                    continue
                mapping[img_id] = os.path.normpath(os.path.join(base_dir_str, rel_path))

        print(f"Loaded {len(mapping)} image path mappings.", flush=True)
        self._image_path_map = mapping
        return mapping

    def _iter_listing_files(self) -> Generator[Path, None, None]:
        """Finds and yields all listing metadata json.gz files."""
        search_dirs = [
            self.listings_root / "listings" / "metadata",
            self.listings_root / "metadata",
            self.listings_root,
        ]
        
        found_files: List[Path] = []
        for sdir in search_dirs:
            if sdir.exists():
                found_files.extend(sorted(sdir.glob("*.json.gz")))
                found_files.extend(sorted(sdir.glob("*.json")))

        if not found_files:
            found_files = sorted(self.listings_root.rglob("*.json.gz"))

        if not found_files:
            raise FileNotFoundError(
                f"No listing JSON files found under {self.listings_root}. "
                "Please ensure abo-listings archive is extracted."
            )

        for f in found_files:
            # Skip non-listing metadata like images.json.gz if any
            if "images" in f.name.lower() and "listing" not in f.name.lower():
                continue
            yield f

    @staticmethod
    def _extract_text(field: Any) -> str:
        """Extracts text from ABO metadata field dict or list with en_US priority."""
        if field is None:
            return ""
        if isinstance(field, str):
            return field.strip()
        if isinstance(field, list):
            if len(field) == 0:
                return ""
            # Look for English tag first (e.g. en_US, en_GB, en)
            for entry in field:
                if isinstance(entry, dict):
                    tag = entry.get("language_tag", "")
                    if tag.startswith("en"):
                        return entry.get("value", "").strip()
            # Fallback to first entry
            first = field[0]
            if isinstance(first, dict):
                return first.get("value", "").strip()
            return str(first).strip()
        if isinstance(field, dict):
            return field.get("value", str(field)).strip()
        return str(field).strip()

    def parse_raw_listings(self) -> pd.DataFrame:
        """Parses raw listing shards into an unstratified full DataFrame."""
        image_meta = self._load_image_metadata_csv()
        records: List[Dict[str, Any]] = []

        listing_files = list(self._iter_listing_files())
        print(f"Parsing {len(listing_files)} listing shards...")

        for shard in tqdm(listing_files, desc="Parsing ABO shards"):
            is_gz = shard.name.endswith(".gz")
            open_fn = gzip.open if is_gz else open

            with open_fn(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sku_id = item.get("item_id")
                    if not sku_id:
                        continue

                    title = self._extract_text(item.get("item_name"))
                    brand = self._extract_text(item.get("brand"))
                    category = self._extract_text(item.get("product_type"))
                    if not category:
                        category = "OTHER"

                    # Collect image IDs (main + other)
                    img_ids: List[str] = []
                    main_img = item.get("main_image_id")
                    if main_img:
                        img_ids.append(main_img)
                    
                    other_imgs = item.get("other_image_id", [])
                    if isinstance(other_imgs, list):
                        for oid in other_imgs:
                            if oid and oid not in img_ids:
                                img_ids.append(oid)
                    elif isinstance(other_imgs, str) and other_imgs not in img_ids:
                        img_ids.append(other_imgs)

                    # Keep only images present in ABO image metadata
                    valid_paths = [image_meta[iid] for iid in img_ids if iid in image_meta]
                    if not valid_paths:
                        continue

                    for idx, img_path in enumerate(valid_paths):
                        records.append({
                            "sku_id": sku_id,
                            "image_path": img_path,
                            "image_index": idx,
                            "is_main_image": (idx == 0),
                            "title": title,
                            "brand": brand,
                            "category": category,
                        })

        df = pd.DataFrame(records)
        print(f"Parsed total {len(df)} rows across {df['sku_id'].nunique()} unique SKUs.")
        return df

    def filter_and_stratify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters SKUs by image count, deduplicates, and stratifies down to max_skus."""
        print("Filtering and stratifying catalog...")

        # 1. Filter by image existence on disk if files are extracted
        sample_path = df["image_path"].iloc[0] if len(df) > 0 else ""
        if os.path.exists(sample_path):
            existing_mask = df["image_path"].map(os.path.exists)
            df = df[existing_mask].copy()
            print(f"Rows with verified on-disk images: {len(df)}")

        # 2. Filter by min images per SKU
        sku_counts = df.groupby("sku_id")["image_path"].count()
        valid_skus = sku_counts[sku_counts >= self.min_images_per_sku].index
        df = df[df["sku_id"].isin(valid_skus)].copy()
        print(f"SKUs with >= {self.min_images_per_sku} images: {df['sku_id'].nunique()}")

        # 3. Clean category names & consolidate tiny tail categories
        cat_counts = df.groupby("category")["sku_id"].nunique()
        top_categories = cat_counts[cat_counts >= 20].index.tolist()
        df["category"] = df["category"].apply(lambda c: c if c in top_categories else "OTHER")

        # 4. Stratified SKU sampling down to max_skus
        sku_meta = df.groupby("sku_id").agg({
            "category": "first",
            "image_path": "count"
        }).rename(columns={"image_path": "num_images"}).reset_index()

        total_skus = len(sku_meta)
        if total_skus > self.max_skus:
            print(f"Sampling {self.max_skus} SKUs from {total_skus} total SKUs (stratified by category)...")
            # Group rare categories for stratification if needed
            strat_labels = sku_meta["category"]
            min_cat_count = strat_labels.value_counts().min()
            if min_cat_count < 2:
                strat_labels = None

            sampled_skus, _ = train_test_split(
                sku_meta,
                train_size=self.max_skus,
                stratify=strat_labels,
                random_state=self.seed,
            )
            df = df[df["sku_id"].isin(sampled_skus["sku_id"])].copy()

        print(f"Final sampled dataset: {len(df)} rows across {df['sku_id'].nunique()} SKUs.")
        return df

    def load_manifest(self) -> pd.DataFrame:
        """End-to-end load, parse, filter, and stratify manifest."""
        raw_df = self.parse_raw_listings()
        stratified_df = self.filter_and_stratify(raw_df)
        return stratified_df

    def create_stratified_splits(
        self,
        target_skus: Optional[int] = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        save_dir: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads manifest, splits into zero-leakage train/val/test partitions, and optionally saves."""
        if target_skus is not None:
            self.max_skus = target_skus
        manifest = self.load_manifest()
        train_df, val_df, test_df = split_by_sku(
            manifest,
            seed=self.seed,
            ratios=(train_ratio, val_ratio, test_ratio),
        )
        if save_dir is not None:
            save_manifests_and_stats(train_df, val_df, test_df, output_dir=save_dir)
        return train_df, val_df, test_df


def split_by_sku(
    manifest: pd.DataFrame,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits manifest at the SKU level into Train, Validation, and Test partitions.

    Ensures zero SKU leakage across splits while preserving category balance.
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    sku_categories = manifest.groupby("sku_id")["category"].first().reset_index()
    
    # Stratify by category if possible (requires at least 2 classes and min 2 items per class)
    cat_counts = sku_categories["category"].value_counts()
    valid_strat = len(cat_counts) > 1 and (cat_counts.min() >= 2)
    strat_col = sku_categories["category"] if valid_strat else None

    # Step 1: Train vs (Val + Test)
    temp_size = val_r + test_r
    train_skus, temp_skus = train_test_split(
        sku_categories,
        test_size=temp_size,
        stratify=strat_col,
        random_state=seed,
    )

    # Step 2: Val vs Test
    temp_cat_counts = temp_skus["category"].value_counts()
    valid_temp_strat = len(temp_cat_counts) > 1 and (temp_cat_counts.min() >= 2)
    temp_strat_col = temp_skus["category"] if valid_temp_strat else None
    val_prop = val_r / (val_r + test_r)

    val_skus, test_skus = train_test_split(
        temp_skus,
        train_size=val_prop,
        stratify=temp_strat_col,
        random_state=seed,
    )

    # Convert to sets for disjointness verification
    train_set = set(train_skus["sku_id"])
    val_set = set(val_skus["sku_id"])
    test_set = set(test_skus["sku_id"])

    # Strict disjointness check
    assert train_set.isdisjoint(val_set), "CRITICAL: SKU leakage between train and val!"
    assert train_set.isdisjoint(test_set), "CRITICAL: SKU leakage between train and test!"
    assert val_set.isdisjoint(test_set), "CRITICAL: SKU leakage between val and test!"

    def subset(sku_set: Set[str], split_name: str) -> pd.DataFrame:
        sub = manifest[manifest["sku_id"].isin(sku_set)].copy()
        sub["split"] = split_name
        return sub

    train_df = subset(train_set, "train")
    val_df = subset(val_set, "val")
    test_df = subset(test_set, "test")

    print(
        f"SKU-level split complete:\n"
        f"  - Train: {len(train_df)} rows ({train_df['sku_id'].nunique()} SKUs)\n"
        f"  - Val:   {len(val_df)} rows ({val_df['sku_id'].nunique()} SKUs)\n"
        f"  - Test:  {len(test_df)} rows ({test_df['sku_id'].nunique()} SKUs)"
    )
    return train_df, val_df, test_df


def save_manifests_and_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> None:
    """Saves manifest CSVs and category statistics JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "manifest_train.csv"
    val_path = out_path / "manifest_val.csv"
    test_path = out_path / "manifest_test.csv"
    stats_path = out_path / "category_stats.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    stats = {
        "total_rows": len(full_df),
        "total_skus": int(full_df["sku_id"].nunique()),
        "split_counts": {
            "train_rows": len(train_df),
            "train_skus": int(train_df["sku_id"].nunique()),
            "val_rows": len(val_df),
            "val_skus": int(val_df["sku_id"].nunique()),
            "test_rows": len(test_df),
            "test_skus": int(test_df["sku_id"].nunique()),
        },
        "category_distribution": full_df["category"].value_counts().to_dict(),
        "images_per_sku_stats": {
            "mean": float(full_df.groupby("sku_id")["image_path"].count().mean()),
            "median": float(full_df.groupby("sku_id")["image_path"].count().median()),
            "max": int(full_df.groupby("sku_id")["image_path"].count().max()),
            "min": int(full_df.groupby("sku_id")["image_path"].count().min()),
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved manifests and category stats to {out_path}")


def main() -> None:
    """CLI entrypoint for ABO dataset download, parsing, and splitting."""
    import argparse

    parser = argparse.ArgumentParser(description="FusionMatch ABO Data Pipeline Loader")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Root directory for raw archives")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory for processed manifests")
    parser.add_argument("--max_skus", type=int, default=11000, help="Max SKUs to sample")
    parser.add_argument("--min_images", type=int, default=1, help="Min images per SKU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading archives if present")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    listings_tar = data_dir / "abo-listings.tar"
    images_tar = data_dir / "abo-images-small.tar"

    listings_extracted = data_dir / "abo-listings"
    if not listings_extracted.exists() and (data_dir / "listings").exists():
        listings_extracted = data_dir

    images_extracted = data_dir / "abo-images-small"
    if not images_extracted.exists() and (data_dir / "small").exists():
        images_extracted = data_dir

    has_images = (
        images_extracted.exists()
        or (data_dir / "small").exists()
        or (data_dir / "metadata" / "images.csv").exists()
    )
    has_listings = (
        listings_extracted.exists()
        or (data_dir / "listings").exists()
        or (data_dir / "abo-listings").exists()
    )

    if not args.skip_download and not has_listings:
        if not listings_extracted.exists():
            download_file(ABO_LISTINGS_URL, listings_tar)
            extract_tar(listings_tar, listings_extracted)
    else:
        print(f"Using listings at: {listings_extracted}", flush=True)

    if not args.skip_download and not has_images:
        if not images_extracted.exists():
            download_file(ABO_IMAGES_SMALL_URL, images_tar)
            extract_tar(images_tar, images_extracted)
    else:
        print(f"Using images at: {images_extracted}", flush=True)

    loader = ABOCatalogLoader(
        images_root=images_extracted,
        listings_root=listings_extracted,
        max_skus=args.max_skus,
        min_images_per_sku=args.min_images,
        seed=args.seed,
    )

    manifest_df = loader.load_manifest()
    train_df, val_df, test_df = split_by_sku(manifest_df, seed=args.seed)
    save_manifests_and_stats(train_df, val_df, test_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()


