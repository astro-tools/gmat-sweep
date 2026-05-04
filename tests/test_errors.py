"""Tests for gmat_sweep.errors — typed exception hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    RunFailed,
    SweepConfigError,
)


def test_root_class_subclasses_exception() -> None:
    assert issubclass(GmatSweepError, Exception)


@pytest.mark.parametrize(
    "cls",
    [SweepConfigError, RunFailed, BackendError, ManifestCorruptError],
)
def test_every_leaf_is_a_gmat_sweep_error(cls: type[GmatSweepError]) -> None:
    assert issubclass(cls, GmatSweepError)


@pytest.mark.parametrize("cls", [SweepConfigError, BackendError])
def test_plain_message_classes_carry_only_a_message(cls: type[GmatSweepError]) -> None:
    exc = cls("nope")
    assert isinstance(exc, GmatSweepError)
    assert str(exc) == "nope"


def test_run_failed_carries_run_id_and_stderr() -> None:
    exc = RunFailed("worker died", run_id=7, stderr="Traceback (most recent call last)…")
    assert exc.run_id == 7
    assert exc.stderr == "Traceback (most recent call last)…"
    assert str(exc) == "worker died"


def test_run_failed_stderr_defaults_to_none() -> None:
    exc = RunFailed("worker died", run_id=0)
    assert exc.stderr is None


def test_manifest_corrupt_carries_path() -> None:
    p = Path("/tmp/manifest.json")
    exc = ManifestCorruptError("invalid json at byte 42", p)
    assert exc.path == p
    assert str(exc) == "invalid json at byte 42"
