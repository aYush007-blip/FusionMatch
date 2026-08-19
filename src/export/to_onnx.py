"""ONNX Model Export Engine for FusionMatch."""

from typing import Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
import onnx
from ..models.fusion_match_model import FusionMatchModel
from ..utils.io import load_checkpoint


def export_to_onnx(
    model_or_ckpt: Union[nn.Module, str, Path],
    output_path: Union[str, Path],
    model_id: str = "google/siglip-base-patch16-224",
    embed_dim: int = 256,
    opset_version: int = 17,
    use_mock: bool = False,
) -> Path:
    """Exports PyTorch FusionMatchModel to ONNX format with dynamic batching.
    
    Args:
        model_or_ckpt: Pre-instantiated PyTorch module or path to .pt checkpoint
        output_path: Target destination path for .onnx file
        model_id: SigLIP backbone identifier
        embed_dim: Output unit-hypersphere projection dimension
        opset_version: ONNX operator set version (>=17)
        use_mock: If True, uses mock encoder for fast testing/export
        
    Returns:
        Path to validated ONNX model file
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_or_ckpt, (str, Path)):
        model = FusionMatchModel(model_id=model_id, embed_dim=embed_dim, use_mock=use_mock)
        ckpt = load_checkpoint(model_or_ckpt, device="cpu")
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = model_or_ckpt

    model.cpu()
    model.eval()

    # Dummy inputs for graph tracing
    dummy_pixels = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_input_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 32, dtype=torch.long)
    dummy_qv = torch.tensor([0.85], dtype=torch.float32)
    dummy_qt = torch.tensor([0.75], dtype=torch.float32)

    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "q_v": {0: "batch_size"},
        "q_t": {0: "batch_size"},
        "embedding": {0: "batch_size"},
        "gates": {0: "batch_size"},
    }

    print(f"Exporting FusionMatch model to ONNX: {out_file} (opset={opset_version})...")
    torch.onnx.export(
        model,
        (dummy_pixels, dummy_input_ids, dummy_attention_mask, dummy_qv, dummy_qt),
        str(out_file),
        input_names=["pixel_values", "input_ids", "attention_mask", "q_v", "q_t"],
        output_names=["embedding", "gates"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"ONNX export successful: {out_file} ({file_size_mb:.2f} MB)")
    return out_file
