"""Checkpoint saving and loading utilities."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch


def save_checkpoint(
    state_dict: Dict[str, Any],
    save_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves a model checkpoint dictionary with parent directory creation."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": state_dict}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, str(path))
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads a saved checkpoint dictionary."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload, "metadata": {}}
