"""Quality proxy estimators for images and text in FusionMatch."""

from typing import List, Union, Optional
import cv2
import numpy as np
from PIL import Image
import torch


@torch.no_grad()
def compute_single_image_quality(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> float:
    """Computes blur variance + resolution score for a single image.
    
    Returns:
        float: Quality proxy score in range [0.0, 1.0].
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor (C, H, W) or (1, C, H, W) to numpy uint8
        t = image.detach().cpu().float()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.shape[0] in [1, 3]:  # CHW -> HWC
            t = t.permute(1, 2, 0)
        # Check if normalized in [-1, 1] and rescale to [0, 1]
        if t.min() < 0.0:
            t = t * 0.5 + 0.5
        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.squeeze(-1)
    elif isinstance(image, Image.Image):
        gray = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[0] == 3:
            gray = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray = image
    else:
        return 0.5

    # Laplacian variance as sharpness proxy
    try:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = min(lap_var / 500.0, 1.0)
    except Exception:
        blur_score = 0.5

    # Resolution proxy normalized to 256x256
    h, w = gray.shape[:2]
    res_score = min((h * w) / (256.0 * 256.0), 1.0)

    # Weighted blend (70% sharpness / 30% resolution)
    quality = float(0.7 * blur_score + 0.3 * res_score)
    return float(np.clip(quality, 0.0, 1.0))


@torch.no_grad()
def image_quality_score(
    images: Union[List[Image.Image], List[np.ndarray], torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """Computes batch image quality proxy scores.
    
    Args:
        images: List of PIL Images, list of numpy arrays, or (B, C, H, W) tensor.
        
    Returns:
        torch.Tensor of shape (B,) with float values in [0.0, 1.0].
    """
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        # Batch tensor (B, C, H, W)
        scores = [compute_single_image_quality(images[i]) for i in range(images.size(0))]
    else:
        scores = [compute_single_image_quality(img) for img in images]

    return torch.tensor(scores, dtype=torch.float32)


@torch.no_grad()
def compute_single_text_quality(text: Optional[str], tokenizer=None) -> float:
    """Computes information density proxy score for a single string.
    
    Returns:
        float: Quality proxy score in range [0.0, 1.0].
    """
    if not text or not str(text).strip():
        return 0.0

    t_str = str(text).strip()
    if tokenizer is not None and hasattr(tokenizer, "tokenize"):
        try:
            tokens = tokenizer.tokenize(t_str)
            n_tokens = len(tokens)
        except Exception:
            n_tokens = len(t_str.split())
    else:
        n_tokens = len(t_str.split())

    # Saturates at ~20 tokens
    score = min(n_tokens / 20.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


@torch.no_grad()
def text_quality_score(
    texts: Union[List[str], List[Optional[str]]],
    tokenizer=None
) -> torch.Tensor:
    """Computes batch text quality proxy scores.
    
    Args:
        texts: List of text strings.
        tokenizer: Optional Hugging Face tokenizer instance.
        
    Returns:
        torch.Tensor of shape (B,) with float values in [0.0, 1.0].
    """
    scores = [compute_single_text_quality(t, tokenizer=tokenizer) for t in texts]
    return torch.tensor(scores, dtype=torch.float32)
