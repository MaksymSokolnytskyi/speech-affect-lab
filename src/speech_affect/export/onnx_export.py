"""Export of a truncated encoder to ONNX.

`TruncatedEncoder` = (input normalisation, if the feature extractor does it) -> CNN -> the
first k transformer layers -> mean over time. The output is a `[1, hidden_size]` embedding,
exactly the one cached in stage 1 for layer k (`hidden_states[k]`).

Why only the encoder and not encoder + classifier: the classifier is linear and tiny, and
quantisation only changes the embedding. This lets int8 be evaluated fairly: the same
out-of-fold probe trained on fp32 embeddings is applied to int8 embeddings.

Truncation: in the HF implementation `hidden_states[i]` is the state AFTER i layers (0 is
before the first layer). So we keep `layers[:k]` and take the output of the last of them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

from speech_affect.config import get_config

log = logging.getLogger(__name__)

INPUT_NAME = "waveform"
OUTPUT_NAME = "clip_embedding"


class TruncatedEncoder(nn.Module):
    def __init__(
        self, hf_id: str, revision: str | None, n_layers: int | None, normalize_input: bool
    ):
        super().__init__()
        model = AutoModel.from_pretrained(hf_id, revision=revision, attn_implementation="eager")
        n_total = model.config.num_hidden_layers
        n_layers = n_total if n_layers is None else n_layers
        if not 1 <= n_layers <= n_total:
            raise ValueError(f"n_layers must be in 1..{n_total}, got {n_layers}")
        model.encoder.layers = model.encoder.layers[:n_layers]
        model.config.num_hidden_layers = n_layers
        self.model = model.eval()
        self.normalize_input = normalize_input
        self.n_layers = n_layers
        self.hidden_size = int(model.config.hidden_size)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values
        if self.normalize_input:
            # Same as zero_mean_unit_var_norm in the HF feature extractor.
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-7)
        hidden = self.model(x).last_hidden_state  # [1, T, D]
        return hidden.mean(dim=1)


def build(model_name: str, n_layers: int | None = None) -> TruncatedEncoder:
    """Encoder `model_name` truncated to its first `n_layers` layers (None = all layers)."""
    enc = get_config().encoders
    hf_id, revision = enc.models[model_name], enc.revisions.get(model_name)
    fe = AutoFeatureExtractor.from_pretrained(hf_id, revision=revision)
    return TruncatedEncoder(hf_id, revision, n_layers, normalize_input=bool(fe.do_normalize))


def export(module: TruncatedEncoder, out: Path, sample_rate: int | None = None) -> Path:
    """Export with a dynamic input length."""
    cfg = get_config()
    sr = sample_rate or cfg.audio.sample_rate
    out.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3 * sr) * 0.1
    time_dim = torch.export.Dim(
        "time",
        min=int(cfg.cremad.min_duration_s * sr),
        max=int(cfg.export.max_input_seconds * sr),
    )
    try:
        torch.onnx.export(
            module,
            (example,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamo=True,
            dynamic_shapes={"input_values": {1: time_dim}},
            optimize=True,
            external_data=False,  # weights inside the .onnx (model < 2 GB): simpler to quantise
        )
    except Exception as e:  # fallback: the legacy TorchScript-based exporter
        log.warning("dynamo export failed (%s), falling back to TorchScript", type(e).__name__)
        torch.onnx.export(
            module,
            (example,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {1: "time"}},
            opset_version=17,
            dynamo=False,
        )
    return out


@torch.inference_mode()
def torch_embedding(module: TruncatedEncoder, wave: np.ndarray) -> np.ndarray:
    return module(torch.from_numpy(wave[None].astype(np.float32))).numpy()
