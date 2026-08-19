"""Augmentation Engine for FusionMatch Multimodal Pipeline.

Implements realistic e-commerce listing perturbations for both visual and textual modalities:
- Image: Geometric transforms, color jitter, JPEG compression artifacts, Gaussian noise, random erasing.
- Text: Typo injection, token/brand dropout, text truncation, whitespace noise.
"""

from __future__ import annotations

import io
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class TextAugmenter:
    """Text perturbation engine simulating noisy multi-seller e-commerce listings."""

    KEYBOARD_ADJACENCY: Dict[str, str] = {
        "q": "wased", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh",
        "y": "tughj", "u": "yihjk", "i": "uojkl", "o": "ipkl", "p": "ol",
        "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
        "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
        "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
        "n": "bhjm", "m": "njk"
    }

    def __init__(
        self,
        typo_prob: float = 0.3,
        char_swap_prob: float = 0.2,
        word_dropout_prob: float = 0.2,
        truncation_prob: float = 0.15,
        empty_text_prob: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.typo_prob = typo_prob
        self.char_swap_prob = char_swap_prob
        self.word_dropout_prob = word_dropout_prob
        self.truncation_prob = truncation_prob
        self.empty_text_prob = empty_text_prob
        self.rng = random.Random(seed)

    def inject_typo(self, text: str) -> str:
        """Substitutes a character with an adjacent keyboard key."""
        if not text or len(text) < 3:
            return text
        chars = list(text)
        idx = self.rng.randint(0, len(chars) - 1)
        ch = chars[idx].lower()
        if ch in self.KEYBOARD_ADJACENCY:
            repl = self.rng.choice(self.KEYBOARD_ADJACENCY[ch])
            chars[idx] = repl.upper() if chars[idx].isupper() else repl
        return "".join(chars)

    def swap_adjacent_chars(self, text: str) -> str:
        """Swaps two adjacent characters."""
        if not text or len(text) < 4:
            return text
        chars = list(text)
        idx = self.rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def drop_words(self, text: str, brand: Optional[str] = None) -> str:
        """Drops words or brand token to simulate omitted title details."""
        words = text.split()
        if len(words) <= 2:
            return text

        # If brand provided, occasionally drop brand name
        if brand and self.rng.random() < 0.4:
            words = [w for w in words if w.lower() != brand.lower()]

        # Random word dropout
        retained = [w for w in words if self.rng.random() > 0.15]
        if not retained:
            return text
        return " ".join(retained)

    def truncate_text(self, text: str) -> str:
        """Simulates truncated titles common on aggregator feeds."""
        words = text.split()
        if len(words) <= 3:
            return text
        keep_len = self.rng.randint(2, max(2, len(words) - 1))
        return " ".join(words[:keep_len])

    def augment(self, text: str, brand: Optional[str] = None) -> str:
        """Applies stochastic text perturbation pipeline."""
        if not text or not text.strip():
            return ""

        # Test quality-gate resilience with occasional empty string
        if self.rng.random() < self.empty_text_prob:
            return ""

        result = text

        if self.rng.random() < self.word_dropout_prob:
            result = self.drop_words(result, brand)

        if self.rng.random() < self.truncation_prob:
            result = self.truncate_text(result)

        if self.rng.random() < self.char_swap_prob:
            result = self.swap_adjacent_chars(result)

        if self.rng.random() < self.typo_prob:
            result = self.inject_typo(result)

        # Normalize redundant whitespace
        result = re.sub(r"\s+", " ", result).strip()
        return result if result else text


class ImageAugmenter:
    """Image perturbation engine using PIL & NumPy (compatible with both Albumentations & torchvision)."""

    def __init__(
        self,
        img_size: int = 224,
        color_jitter_prob: float = 0.5,
        jpeg_compression_prob: float = 0.3,
        gaussian_noise_prob: float = 0.25,
        cutout_prob: float = 0.3,
        horizontal_flip_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.img_size = img_size
        self.color_jitter_prob = color_jitter_prob
        self.jpeg_compression_prob = jpeg_compression_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.cutout_prob = cutout_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def apply_color_jitter(self, img: Image.Image) -> Image.Image:
        """Applies subtle brightness and contrast adjustments."""
        from PIL import ImageEnhance
        
        # Brightness factor 0.8 to 1.2
        b_factor = self.rng.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(b_factor)

        # Contrast factor 0.8 to 1.2
        c_factor = self.rng.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(c_factor)

        # Color / Saturation factor 0.8 to 1.2
        s_factor = self.rng.uniform(0.8, 1.2)
        img = ImageEnhance.Color(img).enhance(s_factor)

        return img

    def apply_jpeg_compression(self, img: Image.Image, quality_range: Tuple[int, int] = (40, 80)) -> Image.Image:
        """Simulates lossy JPEG re-encoding artifacts common in web scrapes."""
        quality = self.rng.randint(quality_range[0], quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def apply_gaussian_noise(self, img: Image.Image, sigma: float = 12.0) -> Image.Image:
        """Adds zero-mean Gaussian noise to image array."""
        arr = np.array(img, dtype=np.float32)
        noise = self.np_rng.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def apply_cutout(self, img: Image.Image, max_box_ratio: float = 0.25) -> Image.Image:
        """Simulates sticker/watermark/banner occlusion via random cutout box."""
        w, h = img.size
        box_w = int(w * self.rng.uniform(0.1, max_box_ratio))
        box_h = int(h * self.rng.uniform(0.1, max_box_ratio))
        x1 = self.rng.randint(0, max(0, w - box_w))
        y1 = self.rng.randint(0, max(0, h - box_h))
        
        arr = np.array(img).copy()
        # Fill with either gray (128) or white/black
        fill_color = self.rng.choice([0, 128, 240])
        arr[y1:y1 + box_h, x1:x1 + box_w, :] = fill_color
        return Image.fromarray(arr)

    def augment(self, img: Image.Image | np.ndarray) -> Image.Image:
        """Applies stochastic image perturbation pipeline."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        # 1. Horizontal Flip
        if self.rng.random() < self.horizontal_flip_prob:
            img = ImageOps.mirror(img)

        # 2. Color Jitter
        if self.rng.random() < self.color_jitter_prob:
            img = self.apply_color_jitter(img)

        # 3. JPEG Compression
        if self.rng.random() < self.jpeg_compression_prob:
            img = self.apply_jpeg_compression(img)

        # 4. Gaussian Noise
        if self.rng.random() < self.gaussian_noise_prob:
            img = self.apply_gaussian_noise(img)

        # 5. Cutout / Occlusion
        if self.rng.random() < self.cutout_prob:
            img = self.apply_cutout(img)

        # Resize to standard size (e.g. 256x256)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        return img


def get_image_augmentations(split: str = "train", img_size: int = 256) -> ImageAugmenter:
    """Returns image augmentation handler configured for train or eval."""
    if split == "train":
        return ImageAugmenter(img_size=img_size)
    return ImageAugmenter(
        img_size=img_size,
        color_jitter_prob=0.0,
        jpeg_compression_prob=0.0,
        gaussian_noise_prob=0.0,
        cutout_prob=0.0,
        horizontal_flip_prob=0.0,
    )


def get_text_augmentations(split: str = "train") -> TextAugmenter:
    """Returns text augmentation handler configured for train or eval."""
    if split == "train":
        return TextAugmenter()
    return TextAugmenter(
        typo_prob=0.0,
        char_swap_prob=0.0,
        word_dropout_prob=0.0,
        truncation_prob=0.0,
        empty_text_prob=0.0,
    )


def apply_multimodal_augmentations(
    image: Image.Image | np.ndarray,
    text: str,
    brand: Optional[str] = None,
    split: str = "train",
    img_size: int = 256,
) -> Tuple[Image.Image, str]:
    """Applies joint visual and textual augmentation."""
    img_aug = get_image_augmentations(split=split, img_size=img_size)
    txt_aug = get_text_augmentations(split=split)

    aug_image = img_aug.augment(image)
    aug_text = txt_aug.augment(text, brand=brand)
    return aug_image, aug_text
