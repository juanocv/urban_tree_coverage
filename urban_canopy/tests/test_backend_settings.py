"""CLI and API backend construction share one validated settings object."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from urban_canopy.models.backend_settings import (
    BackendSettings,
    backend_provenance,
    build_segmenter_from_settings,
)
from urban_canopy.models.taxonomy import ADE20K

BACKEND_ENV_KEYS = (
    "UC_SEG_BACKEND",
    "UC_SEG_MODEL",
    "UC_SEG_TASK",
    "UC_DEVICE",
    "UC_TAXONOMY",
    "UC_D2_CONFIG",
    "UC_D2_WEIGHTS",
    "UC_D2_SCORE_THRESH",
    "UC_DEEPLAB_CKPT",
    "UC_DEEPLAB_REPO",
    "UC_DEEPLAB_MODEL",
    "UC_TRUST_CHECKPOINT",
)
ENV_EXAMPLE = Path(__file__).parents[2] / ".env.example"


@pytest.fixture(autouse=True)
def clean_backend_environment(monkeypatch, tmp_path):
    """
    No ambient configuration: neither process variables nor a ``.env`` file.

    Clearing the variables is not enough on its own. pydantic-settings reads
    ``env_file=".env"`` itself, relative to the working directory, so a test run
    from a checkout that has one silently inherits it -- which is exactly how
    ``UC_TRUST_CHECKPOINT=1`` in a developer's own ``.env`` used to fail the
    defaults test on their machine and nowhere else. Starting in an empty
    directory makes "no environment" true. Tests that want a specific ``.env``
    write one and chdir to it themselves; the later chdir wins.
    """
    for key in BACKEND_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


class _Segmenter:
    backend_name = "stub"
    class_space = "ade20k"
    taxonomy = ADE20K
    device = "cpu"


def test_backend_settings_accept_the_full_env_example(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = BackendSettings()

    assert settings.backend == "oneformer"
    assert settings.model_name is None
    assert settings.taxonomy_path is None
    assert settings.deeplab_checkpoint is None


def test_backend_settings_read_deeplab_environment(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"weights")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UC_SEG_BACKEND", "deeplab")
    monkeypatch.setenv("UC_DEEPLAB_CKPT", str(checkpoint))
    monkeypatch.setenv("UC_DEEPLAB_REPO", str(repo))
    monkeypatch.setenv("UC_DEEPLAB_MODEL", "deeplabv3plus_mobilenet")

    settings = BackendSettings()
    captured = {}

    def fake_builder(backend, **kwargs):
        captured.update(backend=backend, **kwargs)
        return object()

    build_segmenter_from_settings(settings, builder=fake_builder)

    assert captured["backend"] == "deeplab"
    assert captured["ckpt_path"] == str(checkpoint)
    assert captured["repo_path"] == str(repo)
    assert captured["model_name"] == "deeplabv3plus_mobilenet"
    assert captured["taxonomy"].class_space == "cityscapes"


def test_backend_specific_files_are_validated_before_construction(tmp_path):
    called = False

    def fake_builder(*args, **kwargs):
        nonlocal called
        called = True

    settings = BackendSettings(
        _env_file=None,
        backend="deeplab",
        device="cpu",
        deeplab_checkpoint=tmp_path / "missing.pth",
    )
    with pytest.raises(FileNotFoundError, match="UC_DEEPLAB_CKPT"):
        build_segmenter_from_settings(settings, builder=fake_builder)
    assert called is False


def test_backend_and_device_literals_are_enforced_at_runtime():
    with pytest.raises(ValidationError):
        BackendSettings(_env_file=None, backend="unknown")
    with pytest.raises(ValidationError):
        BackendSettings(_env_file=None, device="tpu")


def test_provenance_hashes_a_configured_local_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint bytes")
    settings = BackendSettings(
        _env_file=None,
        backend="deeplab",
        device="cpu",
        deeplab_checkpoint=checkpoint,
        deeplab_model="deeplabv3plus_mobilenet",
    )

    payload = backend_provenance(_Segmenter(), settings)

    assert payload["checkpoint"] == "model.pth"
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()
    assert payload["taxonomy"]["class_space"] == "ade20k"
    assert payload["taxonomy_source"] == "built-in"


# ------------------------------------------------- CLI / environment merge ---
# Regression: these four flags used to carry an argparse default, which
# from_cli_args wrote over the environment-backed value unconditionally. The
# UC_* settings were shipped in .env.example and documented, yet had no effect
# on any CLI run.
def _analyse_args(*argv):
    from urban_canopy.cli._argparse import build_parser

    return build_parser().parse_args(["analyse", "--image", "x.jpg", *argv])


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("UC_SEG_BACKEND", "deeplab", "backend", "deeplab"),
        ("UC_SEG_TASK", "panoptic", "oneformer_task", "panoptic"),
        ("UC_D2_SCORE_THRESH", "0.9", "d2_score_threshold", 0.9),
        ("UC_TRUST_CHECKPOINT", "1", "trust_checkpoint", True),
    ],
)
def test_environment_survives_when_the_flag_is_absent(
    monkeypatch, variable, value, field, expected
):
    monkeypatch.setenv(variable, value)
    settings = BackendSettings.from_cli_args(_analyse_args(), device="cpu")
    assert getattr(settings, field) == expected


@pytest.mark.parametrize(
    ("variable", "value", "argv", "field", "expected"),
    [
        ("UC_SEG_BACKEND", "deeplab", ("--seg", "mask2former"), "backend", "mask2former"),
        ("UC_SEG_TASK", "panoptic", ("--seg-task", "semantic"), "oneformer_task", "semantic"),
        (
            "UC_D2_SCORE_THRESH",
            "0.9",
            ("--d2-score-thresh", "0.25"),
            "d2_score_threshold",
            0.25,
        ),
    ],
)
def test_flag_still_overrides_the_environment(monkeypatch, variable, value, argv, field, expected):
    monkeypatch.setenv(variable, value)
    settings = BackendSettings.from_cli_args(_analyse_args(*argv), device="cpu")
    assert getattr(settings, field) == expected


def test_defaults_hold_with_neither_flag_nor_environment(monkeypatch):
    for key in BACKEND_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("UC_TRUST_CHECKPOINT", raising=False)

    settings = BackendSettings.from_cli_args(_analyse_args(), device="cpu")
    assert settings.backend == "oneformer"
    assert settings.oneformer_task == "semantic"
    assert settings.d2_score_threshold == pytest.approx(0.50)
    assert settings.trust_checkpoint is False


def test_trust_checkpoint_flag_enables_pickle_without_the_variable(monkeypatch):
    monkeypatch.delenv("UC_TRUST_CHECKPOINT", raising=False)
    settings = BackendSettings.from_cli_args(_analyse_args("--trust-checkpoint"), device="cpu")
    assert settings.trust_checkpoint is True
