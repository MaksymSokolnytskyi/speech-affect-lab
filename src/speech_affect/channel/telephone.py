"""Simulation of a classic telephone channel (PSTN, G.711 mu-law).

Chain for a 16 kHz signal:
1. Band-pass 300-3400 Hz (Butterworth, zero phase): what a telephone line lets through.
2. Resample 16 -> 8 kHz (`resample_poly` applies its own anti-aliasing filter).
3. mu-law: logarithmic compression, quantisation to 8 bits (256 levels), expansion.
   This is the continuous mu-law formula; real G.711 uses a piecewise-linear approximation
   of it, and the difference is small.
4. Back to 8 -> 16 kHz, because the encoders were trained at 16 kHz. Nothing above 4 kHz is
   left in the signal.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

MU = 255
BAND_HZ = (300.0, 3400.0)
FILTER_ORDER = 4
PHONE_SR = 8000


@lru_cache(maxsize=4)
def _bandpass(sr: int) -> np.ndarray:
    return butter(FILTER_ORDER, BAND_HZ, btype="bandpass", fs=sr, output="sos")


def mu_law_encode(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """[-1, 1] -> integer codes 0 ... 2**bits - 1."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    levels = 2**bits - 1
    return np.round((y + 1) / 2 * levels).astype(np.int32)


def mu_law_decode(q: np.ndarray, bits: int = 8) -> np.ndarray:
    levels = 2**bits - 1
    y = q.astype(np.float64) / levels * 2 - 1
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(MU)) / MU).astype(np.float32)


def telephone(wave: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """Clean signal at `sr` -> "passed through a phone line", returned at `sr`, same length."""
    x = sosfiltfilt(_bandpass(sr), wave.astype(np.float64))
    x = resample_poly(x, PHONE_SR, sr)
    # No peak normalisation before mu-law: the signal is already in [-1, 1]; the clip inside
    # mu_law_encode only guards against rare overshoots after the filter.
    x = mu_law_decode(mu_law_encode(x))
    x = resample_poly(x, sr, PHONE_SR)
    n = len(wave)
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, len(x))] = x[:n]
    return out
