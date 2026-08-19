"""Dynamic INT8 Quantization for ONNX Runtime CPU Inference."""

from typing import Union
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_dynamic_int8(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Union[str, Path],
) -> Path:
    """Applies dynamic INT8 weight quantization to an ONNX model for CPU acceleration.
    
    Args:
        input_onnx_path: Path to float32 ONNX model
        output_onnx_path: Destination path for quantized int8 ONNX model
        
    Returns:
        Path to quantized ONNX model
    """
    in_path = Path(input_onnx_path)
    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing ONNX model to INT8: {in_path} -> {out_path}...")
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )

    in_size_mb = in_path.stat().st_size / (1024 * 1024)
    out_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Quantization complete: {in_size_mb:.2f} MB -> {out_size_mb:.2f} MB ({in_size_mb/out_size_mb:.1f}x compression)")

    # Verify session initialization
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    assert len(sess.get_inputs()) >= 5
    return out_path
