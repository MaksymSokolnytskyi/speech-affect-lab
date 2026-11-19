"""Single source of project constants.

All paths, seeds, model IDs and validation parameters live here. Modules and experiments get
them through `get_config()` and carry no "magic numbers" of their own.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field

ROOT_ENV_VAR = "SPEECH_AFFECT_ROOT"


def _default_root() -> Path:
    """Project root: `$SPEECH_AFFECT_ROOT` if set, else the source checkout, else the cwd.

    `parents[2]` of this file is the repository root only for an editable install (`src/`
    layout); for a regular install it points into site-packages, so we fall back to the cwd.
    """
    if env := os.environ.get(ROOT_ENV_VAR):
        return Path(env).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists():
        return checkout
    return Path.cwd().resolve()


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Frozen):
    root: Path = Field(default_factory=_default_root)

    @computed_field
    @property
    def data(self) -> Path:
        return self.root / "data"

    @computed_field
    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @computed_field
    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @computed_field
    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @computed_field
    @property
    def outputs(self) -> Path:
        return self.root / "outputs"


class AudioConfig(_Frozen):
    sample_rate: int = 16_000
    """Rate at which WavLM and HuBERT were pre-trained; CREMA-D is stored at this rate too."""


class DownloadConfig(_Frozen):
    workers: int = 8
    retries: int = 4
    timeout_s: float = 120.0


class CremaDConfig(_Frozen):
    repo_url: str = "https://github.com/CheyneyComputerScience/CREMA-D.git"
    media_url: str = (
        "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/{path}"
    )
    repo_dirname: str = "crema-d-repo"
    """Clone holding git-lfs pointers (no audio): the source of the official SHA-256 hashes."""
    audio_dirname: str = "crema-d-audio"
    """Where the downloaded WAV files go."""

    # Official numbers from the dataset README (checked 2026-11-16, see docs/DATA.md).
    n_clips: int = 7442
    n_speakers: int = 91
    n_male: int = 48
    n_female: int = 43
    n_sentences: int = 12
    emotions: tuple[str, ...] = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
    intensities: tuple[str, ...] = ("LO", "MD", "HI", "XX")

    min_duration_s: float = 0.5
    """Clips shorter than this are treated as corrupt and excluded explicitly (with a log)."""


class EncoderConfig(_Frozen):
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "wavlm": "microsoft/wavlm-base-plus",
            "hubert": "facebook/hubert-base-ls960",
        }
    )
    revisions: dict[str, str] = Field(
        default_factory=lambda: {
            "wavlm": "4c66d4806a428f2e922ccfa1a962776e232d487b",
            "hubert": "dba3bb02fda4248b6e082697eee756de8fe8aa8a",
        }
    )
    """Pinned Hugging Face Hub commits: the weights every cached result was computed with."""
    n_hidden_states: int = 13
    """Input to the first transformer layer + the outputs of 12 layers."""
    hidden_size: int = 768

    @property
    def n_layers(self) -> int:
        """Number of transformer layers (hidden state 0 is the transformer input)."""
        return self.n_hidden_states - 1


class NoiseConfig(_Frozen):
    hf_repo: str = "FluidInference/musan"
    """Unofficial MUSAN mirror on HF: lets us fetch only noise/ (~585 MB) instead of the 11 GB
    OpenSLR archive. File hashes are checked against the mirror's git-lfs metadata."""
    subdirs: tuple[str, ...] = ("free-sound", "sound-bible")
    test_share_pct: int = 30
    """Share of noise files (by filename hash) that go to the test pool."""
    snr_db: tuple[int, ...] = (20, 10, 5)


class ExportConfig(_Frozen):
    bench_seconds: float = 3.0
    bench_warmup: int = 20
    bench_runs: int = 200
    bench_threads: tuple[int, ...] = (1, 4)
    equivalence_atol: float = 1e-4
    """Stage 3 criterion: max absolute difference between ONNX fp32 and PyTorch."""
    n_equivalence_clips: int = 20
    embed_threads: int = 4
    """ONNX Runtime threads for embedding every clip (fp32 vs int8 quality check)."""
    max_input_seconds: float = 30.0
    """Upper bound of the dynamic time axis of the exported model."""


class EvalConfig(_Frozen):
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    logreg_C_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    logreg_max_iter: int = 2000


class Config(_Frozen):
    seed: int = 20261116
    paths: PathsConfig = Field(default_factory=PathsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    cremad: CremaDConfig = Field(default_factory=CremaDConfig)
    encoders: EncoderConfig = Field(default_factory=EncoderConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @property
    def cremad_repo_dir(self) -> Path:
        return self.paths.raw / self.cremad.repo_dirname

    @property
    def cremad_audio_dir(self) -> Path:
        return self.paths.raw / self.cremad.audio_dirname

    @property
    def cremad_manifest_path(self) -> Path:
        return self.paths.manifests / "cremad.parquet"


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
