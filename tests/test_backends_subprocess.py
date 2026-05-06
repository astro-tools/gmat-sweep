"""Tests for gmat_sweep.backends._subprocess.run_spec_in_subprocess.

Mocks subprocess.run so no real GMAT is involved. The integration test in
test_worker_entrypoint_integration.py exercises the real subprocess hop.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from gmat_sweep.backends import _subprocess
from gmat_sweep.spec import RunOutcome, RunSpec


def _make_spec(*, output_dir: Path, run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=Path("/missions/m.script"),
        overrides={},
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


@pytest.fixture
def fake_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Default fake: write a synthetic ok outcome and return exit 0.

    Captures argv into the returned list so tests can assert on the
    invocation. Tests that need a different scenario monkeypatch
    subprocess.run themselves.
    """
    captured: list[list[str]] = []

    def _run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        spec_idx = argv.index("--spec") + 1
        outcome_idx = argv.index("--outcome") + 1
        spec = RunSpec.from_dict(json.loads(Path(argv[spec_idx]).read_text()))
        now = datetime.now(timezone.utc)
        outcome = RunOutcome.ok(
            run_id=spec.run_id,
            output_paths={"report__R": spec.output_dir / "report__R.parquet"},
            started_at=now,
            ended_at=now,
        )
        Path(argv[outcome_idx]).write_text(json.dumps(outcome.to_dict()))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    return captured


def _capture_tempdir(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Wrap tempfile.TemporaryDirectory so the test can recover the chosen path.

    The helper uses the default tempfile context manager; this wrapper records
    the directory so the test can verify cleanup after the call returns or
    raises.
    """
    seen: list[Path] = []
    real_tempdir = tempfile.TemporaryDirectory

    class _Recorder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = real_tempdir(*args, **kwargs)
            seen.append(Path(self._inner.name))

        def __enter__(self) -> str:
            return cast(str, self._inner.__enter__())

        def __exit__(self, *exc_info: Any) -> None:
            self._inner.__exit__(*exc_info)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _Recorder)
    return seen


# ---- happy path ----------------------------------------------------------


def test_round_trips_spec_and_returns_ok_outcome(
    tmp_path: Path, fake_subprocess_run: list[list[str]]
) -> None:
    spec = _make_spec(output_dir=tmp_path / "run-0", run_id=7)
    outcome = _subprocess.run_spec_in_subprocess(spec)

    assert outcome.status == "ok"
    assert outcome.run_id == 7
    assert outcome.output_paths == {"report__R": tmp_path / "run-0" / "report__R.parquet"}


def test_invokes_python_dash_m_with_spec_and_outcome_paths(
    tmp_path: Path, fake_subprocess_run: list[list[str]]
) -> None:
    _subprocess.run_spec_in_subprocess(_make_spec(output_dir=tmp_path / "run-0"))

    assert len(fake_subprocess_run) == 1
    argv = fake_subprocess_run[0]
    assert argv[1:4] == ["-m", "gmat_sweep._worker_entrypoint", "--spec"]
    assert "--outcome" in argv


def test_python_override_changes_interpreter(
    tmp_path: Path, fake_subprocess_run: list[list[str]]
) -> None:
    _subprocess.run_spec_in_subprocess(
        _make_spec(output_dir=tmp_path / "run-0"),
        python="/opt/python/bin/python",
    )

    assert fake_subprocess_run[0][0] == "/opt/python/bin/python"


def test_spec_round_trips_through_temp_file(
    tmp_path: Path, fake_subprocess_run: list[list[str]]
) -> None:
    """Bit-equality across to_dict → JSON file → from_dict in the fake's _run."""
    spec = _make_spec(output_dir=tmp_path / "run-0", run_id=42)

    outcome = _subprocess.run_spec_in_subprocess(spec)
    assert outcome.run_id == 42


# ---- transport failures --------------------------------------------------


def test_non_zero_exit_returns_failed_outcome_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, 3, stdout="", stderr="cannot read spec /tmp/x.json"
        )

    monkeypatch.setattr(subprocess, "run", _run)

    outcome = _subprocess.run_spec_in_subprocess(
        _make_spec(output_dir=tmp_path / "run-0", run_id=5)
    )

    assert outcome.status == "failed"
    assert outcome.run_id == 5
    assert outcome.stderr is not None
    assert "exited with status 3" in outcome.stderr
    assert "cannot read spec" in outcome.stderr


def test_timeout_returns_failed_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=5.0, stderr="partial stderr")

    monkeypatch.setattr(subprocess, "run", _run)

    outcome = _subprocess.run_spec_in_subprocess(
        _make_spec(output_dir=tmp_path / "run-0", run_id=11),
        timeout=5.0,
    )

    assert outcome.status == "failed"
    assert outcome.run_id == 11
    assert outcome.stderr is not None
    assert "timed out after 5.0s" in outcome.stderr
    assert "partial stderr" in outcome.stderr


def test_oserror_on_spawn_returns_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _run(_argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("[Errno 2] No such file or directory: '/no/such/python'")

    monkeypatch.setattr(subprocess, "run", _run)

    outcome = _subprocess.run_spec_in_subprocess(
        _make_spec(output_dir=tmp_path / "run-0", run_id=1),
        python="/no/such/python",
    )

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "could not be spawned" in outcome.stderr


# ---- tempdir cleanup -----------------------------------------------------


def test_tempdir_cleaned_on_success(
    tmp_path: Path, fake_subprocess_run: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_tempdir(monkeypatch)
    _subprocess.run_spec_in_subprocess(_make_spec(output_dir=tmp_path / "run-0"))

    assert len(seen) == 1
    assert not seen[0].exists()


def test_tempdir_cleaned_on_unexpected_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_tempdir(monkeypatch)

    def _run(_argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("unexpected — neither OSError nor TimeoutExpired")

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(RuntimeError):
        _subprocess.run_spec_in_subprocess(_make_spec(output_dir=tmp_path / "run-0"))

    assert len(seen) == 1
    assert not seen[0].exists()
