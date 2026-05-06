"""Tests for gmat_sweep._worker_entrypoint — CLI parsing, JSON round-trip, exit codes.

The module is invoked via ``python -m gmat_sweep._worker_entrypoint``; these
unit tests call ``main(argv)`` in-process so the existing
``fake_gmat_run`` fixture (sys.modules monkeypatch) reaches ``run_one``.
The real subprocess hop is exercised by the integration test in
``test_worker_entrypoint_integration.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gmat_sweep import _worker_entrypoint
from gmat_sweep.spec import RunOutcome
from tests.conftest import FakeGmatRun


def _make_spec_dict(*, output_dir: Path, run_id: int = 0) -> dict[str, Any]:
    return {
        "script_path": "/missions/m.script",
        "overrides": {},
        "output_dir": str(output_dir),
        "run_id": run_id,
        "seed": None,
        "run_options": {},
    }


def _write_spec(path: Path, *, output_dir: Path, run_id: int = 0) -> None:
    path.write_text(json.dumps(_make_spec_dict(output_dir=output_dir, run_id=run_id)))


# ---- happy path ----------------------------------------------------------


def test_main_writes_outcome_and_returns_zero(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    spec_path = tmp_path / "spec.json"
    outcome_path = tmp_path / "outcome.json"
    _write_spec(spec_path, output_dir=tmp_path / "run-0", run_id=0)

    rc = _worker_entrypoint.main(["--spec", str(spec_path), "--outcome", str(outcome_path)])

    assert rc == 0
    outcome = RunOutcome.from_dict(json.loads(outcome_path.read_text()))
    assert outcome.status == "ok"
    assert outcome.run_id == 0


def test_main_failed_run_still_returns_zero(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """A failed run is still a successful entrypoint invocation — exit 0, status='failed'."""
    fake_gmat_run.install_failing_loader(FileNotFoundError("/missions/m.script does not exist"))

    spec_path = tmp_path / "spec.json"
    outcome_path = tmp_path / "outcome.json"
    _write_spec(spec_path, output_dir=tmp_path / "run-0")

    rc = _worker_entrypoint.main(["--spec", str(spec_path), "--outcome", str(outcome_path)])

    assert rc == 0
    outcome = RunOutcome.from_dict(json.loads(outcome_path.read_text()))
    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "m.script" in outcome.stderr


# ---- transport failures --------------------------------------------------


def test_missing_spec_argument_exits_two(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as ei:
        _worker_entrypoint.main(["--outcome", str(tmp_path / "outcome.json")])
    assert ei.value.code == 2


def test_missing_outcome_argument_exits_two(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as ei:
        _worker_entrypoint.main(["--spec", str(tmp_path / "spec.json")])
    assert ei.value.code == 2


def test_unreadable_spec_file_returns_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _worker_entrypoint.main(
        [
            "--spec",
            str(tmp_path / "missing.json"),
            "--outcome",
            str(tmp_path / "outcome.json"),
        ]
    )

    assert rc == 3
    captured = capsys.readouterr()
    assert "cannot read spec" in captured.err
    assert "missing.json" in captured.err


def test_malformed_spec_json_returns_three(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{not valid json")

    rc = _worker_entrypoint.main(
        ["--spec", str(spec_path), "--outcome", str(tmp_path / "outcome.json")]
    )

    assert rc == 3


def test_spec_missing_required_field_returns_three(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"script_path": "/missions/m.script"}))

    rc = _worker_entrypoint.main(
        ["--spec", str(spec_path), "--outcome", str(tmp_path / "outcome.json")]
    )

    assert rc == 3


def test_spec_with_wrong_type_returns_three(tmp_path: Path) -> None:
    """Non-int run_id, etc. — TypeError/ValueError from RunSpec.from_dict."""
    spec_path = tmp_path / "spec.json"
    bad = _make_spec_dict(output_dir=tmp_path / "run-0")
    bad["run_id"] = "not-an-int"
    spec_path.write_text(json.dumps(bad))

    rc = _worker_entrypoint.main(
        ["--spec", str(spec_path), "--outcome", str(tmp_path / "outcome.json")]
    )

    assert rc == 3


def test_unwriteable_outcome_path_returns_four(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, output_dir=tmp_path / "run-0")

    # Pre-create the outcome path as a *directory* so write_text fails with
    # IsADirectoryError (a subclass of OSError).
    outcome_path = tmp_path / "outcome.json"
    outcome_path.mkdir()

    rc = _worker_entrypoint.main(["--spec", str(spec_path), "--outcome", str(outcome_path)])

    assert rc == 4
    captured = capsys.readouterr()
    assert "cannot write outcome" in captured.err
