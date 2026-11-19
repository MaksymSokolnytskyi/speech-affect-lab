from __future__ import annotations

from speech_affect.config import ROOT_ENV_VAR, Config


def test_root_can_be_overridden_and_paths_follow_it(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    cfg = Config()
    assert cfg.paths.root == tmp_path.resolve()
    assert cfg.paths.outputs == tmp_path.resolve() / "outputs"
    assert cfg.cremad_manifest_path.is_relative_to(tmp_path.resolve() / "data")


def test_every_encoder_has_a_pinned_revision():
    enc = Config().encoders
    assert set(enc.revisions) == set(enc.models)
    assert all(len(rev) == 40 for rev in enc.revisions.values())
