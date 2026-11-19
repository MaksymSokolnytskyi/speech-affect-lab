from __future__ import annotations

import warnings

import numpy as np
import pytest

from speech_affect.config import get_config


@pytest.mark.slow
@pytest.mark.parametrize("model", ["wavlm", "hubert"])
def test_onnx_matches_torch(model, tmp_path):
    """ONNX fp32 matches PyTorch within the stage 3 criterion on inputs of different lengths."""
    import onnxruntime as ort

    from speech_affect.export.onnx_export import INPUT_NAME, build, export, torch_embedding

    warnings.filterwarnings("ignore")
    module = build(model, 2)
    path = export(module, tmp_path / f"{model}.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for n in (12_000, 37_000, 64_000):
        w = rng.normal(0, 0.1, n).astype(np.float32)
        diff = np.abs(sess.run(None, {INPUT_NAME: w[None]})[0] - torch_embedding(module, w)).max()
        assert diff <= get_config().export.equivalence_atol


@pytest.mark.slow
def test_truncated_equals_full_model_hidden_state():
    """A model truncated to k layers yields exactly hidden_states[k] of the full model."""
    from speech_affect.export.onnx_export import build, torch_embedding
    from speech_affect.features.encoders import load_encoder

    w = np.random.default_rng(1).normal(0, 0.1, 30_000).astype(np.float32)
    full = load_encoder("hubert", device="cpu").embed(w, get_config().audio.sample_rate)
    trunc = torch_embedding(build("hubert", 4), w)[0]
    np.testing.assert_allclose(trunc, full[4], atol=1e-5)
