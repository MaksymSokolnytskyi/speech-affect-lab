"""Frozen audio encoders: waveform -> one embedding per layer.

What happens in `embed`:
1. The model's feature extractor prepares the input (e.g. normalises the signal to zero mean
   and unit variance, if the model was trained that way).
2. The model runs the audio through a CNN (20 ms frames) and 12 transformer layers.
   `output_hidden_states=True` returns 13 tensors `[1, T, 768]`: the input to the first layer
   plus the output of every layer.
3. Each is averaged over time (mean pooling) -> `[13, 768]`.

Clips are processed one at a time: in the base models the first convolution is followed by
a GroupNorm over the whole time axis, so padding within a batch would change the result.

Weights are loaded at the commit pinned in `EncoderConfig.revisions`, so results stay
reproducible if the model repository on the Hub changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from speech_affect.config import get_config


@dataclass
class Encoder:
    name: str
    hf_id: str
    revision: str | None
    feature_extractor: object
    model: torch.nn.Module
    device: torch.device

    @torch.inference_mode()
    def embed(self, wave: np.ndarray, sample_rate: int) -> np.ndarray:
        """Raw mono float signal -> time-averaged state of every layer, `[n_layers, dim]`."""
        inputs = self.feature_extractor(wave, sampling_rate=sample_rate, return_tensors="pt")
        x = inputs["input_values"].to(self.device)
        out = self.model(x, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # [L, 1, T, D]
        return hs.mean(dim=2).squeeze(1).float().cpu().numpy()


def load_encoder(name: str, device: str | None = None) -> Encoder:
    enc = get_config().encoders
    hf_id, revision = enc.models[name], enc.revisions.get(name)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fe = AutoFeatureExtractor.from_pretrained(hf_id, revision=revision)
    model = AutoModel.from_pretrained(hf_id, revision=revision).to(dev).eval()
    return Encoder(name, hf_id, revision, fe, model, dev)
