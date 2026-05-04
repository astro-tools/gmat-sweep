"""Tests for gmat_sweep.spec — RunSpec / SweepSpec / RunOutcome and JSON round-trip."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gmat_sweep.spec import RunOutcome, RunSpec, SweepSpec


def _utc(year: int, month: int, day: int, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(year, month, day, h, m, s, tzinfo=timezone.utc)


def _make_run_spec(run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=Path("/missions/flyby.script"),
        overrides={"Sat.SMA": 7000.0, "Sat.ECC": 0.001},
        output_dir=Path(f"/sweep-out/run-{run_id}"),
        run_id=run_id,
        seed=42,
        run_options={"overwrite": True},
    )


def _make_sweep_spec(n: int = 3) -> SweepSpec:
    return SweepSpec(
        mission_script_path=Path("/missions/flyby.script"),
        runs=tuple(_make_run_spec(i) for i in range(n)),
        backend="joblib",
        backend_kwargs={"n_jobs": 4},
        output_dir=Path("/sweep-out"),
        manifest_path=Path("/sweep-out/manifest.json"),
        sweep_seed=1729,
    )


# ---- RunSpec --------------------------------------------------------------


def test_run_spec_is_frozen() -> None:
    spec = _make_run_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.run_id = 99  # type: ignore[misc]


def test_run_spec_round_trips_through_json() -> None:
    original = _make_run_spec(run_id=5)
    serialised = json.dumps(original.to_dict(), sort_keys=True)
    restored = RunSpec.from_dict(json.loads(serialised))
    assert restored == original
    # Bit-equal across a re-serialise.
    assert json.dumps(restored.to_dict(), sort_keys=True) == serialised


def test_run_spec_seed_none_round_trips() -> None:
    spec = dataclasses.replace(_make_run_spec(), seed=None)
    restored = RunSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.seed is None
    assert restored == spec


# ---- SweepSpec ------------------------------------------------------------


def test_sweep_spec_round_trips_with_runs() -> None:
    original = _make_sweep_spec(n=5)
    serialised = json.dumps(original.to_dict(), sort_keys=True)
    restored = SweepSpec.from_dict(json.loads(serialised))
    assert restored == original
    assert json.dumps(restored.to_dict(), sort_keys=True) == serialised
    # The contract the manifest and resume flow depend on: run_id == index.
    assert tuple(r.run_id for r in restored.runs) == tuple(range(5))


def test_sweep_spec_default_seed_is_none_and_round_trips() -> None:
    spec = SweepSpec(
        mission_script_path=Path("/m.script"),
        runs=(),
        backend="joblib",
        backend_kwargs={},
        output_dir=Path("/o"),
        manifest_path=Path("/o/manifest.json"),
    )
    assert spec.sweep_seed is None
    restored = SweepSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec


# ---- RunOutcome -----------------------------------------------------------


def test_run_outcome_ok_helper_sets_status_and_clears_stderr() -> None:
    started = _utc(2026, 5, 4, 0, 0, 0)
    ended = _utc(2026, 5, 4, 0, 0, 12)
    out = RunOutcome.ok(
        run_id=2,
        output_paths={"ReportFile1": Path("/o/run-2/r1.parquet")},
        started_at=started,
        ended_at=ended,
    )
    assert out.status == "ok"
    assert out.stderr is None
    assert out.duration_s == 12.0
    assert out.output_paths == {"ReportFile1": Path("/o/run-2/r1.parquet")}


def test_run_outcome_failed_helper_captures_stderr_and_empty_outputs() -> None:
    started = _utc(2026, 5, 4, 0, 0, 0)
    ended = _utc(2026, 5, 4, 0, 0, 5)
    out = RunOutcome.failed(
        run_id=4,
        stderr="GMAT exploded",
        started_at=started,
        ended_at=ended,
    )
    assert out.status == "failed"
    assert out.stderr == "GMAT exploded"
    assert out.duration_s == 5.0
    assert out.output_paths == {}


def test_run_outcome_ok_round_trips_paths_and_timestamps() -> None:
    started = _utc(2026, 5, 4, 0, 0, 0)
    ended = _utc(2026, 5, 4, 0, 1, 30)
    original = RunOutcome.ok(
        run_id=11,
        output_paths={
            "ReportFile1": Path("/o/r1.parquet"),
            "EphemerisFile1": Path("/o/eph.parquet"),
        },
        started_at=started,
        ended_at=ended,
    )
    restored = RunOutcome.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.duration_s == 90.0


def test_run_outcome_failed_round_trips_with_stderr() -> None:
    started = _utc(2026, 5, 4, 0, 0, 0)
    ended = _utc(2026, 5, 4, 0, 0, 5)
    original = RunOutcome.failed(
        run_id=4,
        stderr="GMAT exploded",
        started_at=started,
        ended_at=ended,
    )
    restored = RunOutcome.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.stderr == "GMAT exploded"


def test_run_outcome_skipped_constructed_directly_round_trips() -> None:
    moment = _utc(2026, 5, 4, 0, 0, 0)
    skipped = RunOutcome(
        run_id=8,
        status="skipped",
        output_paths={},
        duration_s=0.0,
        stderr=None,
        started_at=moment,
        ended_at=moment,
    )
    restored = RunOutcome.from_dict(json.loads(json.dumps(skipped.to_dict())))
    assert restored == skipped
    assert restored.status == "skipped"
