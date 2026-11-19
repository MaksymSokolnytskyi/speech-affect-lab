"""Dynamic int8 quantisation of an ONNX model.

Dynamic means the weights of linear layers (MatMul) are quantised to int8 ahead of time,
while activations are quantised on the fly for each input. No calibration data is needed.
The convolutions of the CNN feature encoder stay float32, which is why the speed-up is modest.
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def quantize(fp32_path: Path, out: Path) -> Path:
    pre = out.with_name(out.stem + "_pre.onnx")
    try:
        # Pre-processing (shape inference, graph simplification) is recommended before
        # quantisation.
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        quantize_dynamic(
            str(pre), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
        )
    finally:
        pre.unlink(missing_ok=True)
    return out
