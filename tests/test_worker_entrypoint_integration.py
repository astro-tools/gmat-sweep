"""Integration tests for _worker_entrypoint and _subprocess against real GMAT."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.spec import RunOutcome, RunSpec


def _make_spec(*, script_path: Path, output_dir: Path, run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=script_path,
        overrides={},
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


@pytest.mark.integration
def test_entrypoint_against_stock_script_produces_ok_outcome(
    leo_basic_script: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run-0"
    output_dir.mkdir()

    spec_path = tmp_path / "spec.json"
    outcome_path = tmp_path / "outcome.json"
    spec = _make_spec(script_path=leo_basic_script, output_dir=output_dir)
    spec_path.write_text(json.dumps(spec.to_dict()))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gmat_sweep._worker_entrypoint",
            "--spec",
            str(spec_path),
            "--outcome",
            str(outcome_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    outcome = RunOutcome.from_dict(json.loads(outcome_path.read_text()))
    assert outcome.status == "ok"
    assert outcome.output_paths
    for path in outcome.output_paths.values():
        assert path.exists()
        assert path.suffix == ".parquet"


@pytest.mark.integration
def test_entrypoint_against_broken_spec_writes_failed_outcome_with_zero_exit(
    tmp_path: Path,
) -> None:
    """Failure-as-row at the entrypoint boundary: failed run, exit 0."""
    output_dir = tmp_path / "run-0"
    output_dir.mkdir()

    spec_path = tmp_path / "spec.json"
    outcome_path = tmp_path / "outcome.json"
    spec = _make_spec(
        script_path=tmp_path / "does_not_exist.script",
        output_dir=output_dir,
    )
    spec_path.write_text(json.dumps(spec.to_dict()))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gmat_sweep._worker_entrypoint",
            "--spec",
            str(spec_path),
            "--outcome",
            str(outcome_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    outcome = RunOutcome.from_dict(json.loads(outcome_path.read_text()))
    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert outcome.stderr.strip() != ""


@pytest.mark.integration
def test_two_back_to_back_invocations_both_succeed(leo_basic_script: Path, tmp_path: Path) -> None:
    """Proves the parent never bootstraps gmatpy — both children get a fresh interpreter."""
    (tmp_path / "run-0").mkdir()
    (tmp_path / "run-1").mkdir()
    spec_a = _make_spec(script_path=leo_basic_script, output_dir=tmp_path / "run-0", run_id=0)
    spec_b = _make_spec(script_path=leo_basic_script, output_dir=tmp_path / "run-1", run_id=1)

    outcome_a = run_spec_in_subprocess(spec_a)
    outcome_b = run_spec_in_subprocess(spec_b)

    assert outcome_a.status == "ok", outcome_a.stderr
    assert outcome_b.status == "ok", outcome_b.stderr
    assert outcome_a.run_id == 0
    assert outcome_b.run_id == 1
