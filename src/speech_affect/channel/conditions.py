"""Registry of recording conditions: how to degrade a clean clip before the encoder.

Each condition is a function `(wave, clip_id) -> wave` at 16 kHz. `clip_id` makes the random
choices (which noise segment, which augmentation) deterministic per clip.

Test conditions draw noise only from the test pool of noise files, augmentation only from the
train pool. Noise is added BEFORE the telephone channel: the speaker talks in a noisy room,
and then the sound travels over the phone line.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from speech_affect.channel.noise import mix_at_snr, noise_segment, rng_for
from speech_affect.channel.telephone import telephone
from speech_affect.config import get_config

Condition = Callable[[np.ndarray, str], np.ndarray]


def clean(wave: np.ndarray, clip_id: str) -> np.ndarray:
    return wave


def tel(wave: np.ndarray, clip_id: str) -> np.ndarray:
    return telephone(wave)


def _tel_noise(snr_db: float, pool: str) -> Condition:
    def f(wave: np.ndarray, clip_id: str) -> np.ndarray:
        noisy = mix_at_snr(wave, noise_segment(len(wave), clip_id, pool), snr_db)
        return telephone(noisy)

    f.__name__ = f"tel_snr{snr_db}_{pool}"
    return f


def aug_train(wave: np.ndarray, clip_id: str) -> np.ndarray:
    """Training augmentation: for each clip a random (but deterministic) condition from
    {clean, telephone, telephone + train-pool noise at each SNR}."""
    options: list[Condition] = [clean, tel] + [
        _tel_noise(s, "train") for s in get_config().noise.snr_db
    ]
    choice = rng_for(clip_id, "aug").integers(len(options))
    return options[choice](wave, clip_id)


TEST_CONDITIONS = ["clean", "tel"] + [f"tel_snr{s}" for s in get_config().noise.snr_db]

CONDITIONS: dict[str, Condition] = {
    "clean": clean,
    "tel": tel,
    **{f"tel_snr{s}": _tel_noise(s, "test") for s in get_config().noise.snr_db},
    "aug_train": aug_train,
}


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise KeyError(f"unknown condition {name!r}; available: {sorted(CONDITIONS)}") from None
