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
        duration_s=12.0,
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
        duration_s=5.0,
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
        duration_s=90.0,
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
        duration_s=5.0,
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


def test_run_outcome_from_dict_rejects_unknown_status() -> None:
    moment = _utc(2026, 5, 4, 0, 0, 0)
    payload = {
        "run_id": 1,
        "status": "banana",
        "output_paths": {},
        "duration_s": 0.0,
        "stderr": None,
        "started_at": moment.isoformat(),
        "ended_at": moment.isoformat(),
    }
    with pytest.raises(ValueError, match="banana"):
        RunOutcome.from_dict(payload)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("script_path", 123),
        ("output_dir", None),
        ("run_id", "0"),
        ("run_id", True),  # bool is an int subclass — reject explicitly
        ("seed", "42"),
        ("overrides", ["Sat.SMA", 7000.0]),
        ("run_options", "overwrite"),
    ],
)
def test_run_spec_from_dict_rejects_malformed_field(field: str, bad_value: object) -> None:
    payload: dict[str, object] = {
        "script_path": "/missions/flyby.script",
        "overrides": {"Sat.SMA": 7000.0},
        "output_dir": "/sweep-out/run-0",
        "run_id": 0,
        "seed": 42,
        "run_options": {},
    }
    payload[field] = bad_value
    with pytest.raises(ValueError, match=field):
        RunSpec.from_dict(payload)


def test_run_outcome_duration_unaffected_by_wall_clock_step() -> None:
    """Wall-clock bookends can disagree with duration_s when NTP corrects mid-run.

    The fix is to source duration from time.monotonic() inside the worker; this
    test asserts the helpers accept (and preserve) a duration that contradicts
    `ended_at - started_at`.
    """
    started = _utc(2026, 5, 4, 12, 0, 0)
    # Simulate the system clock being stepped backwards by 5 seconds mid-run:
    # ended_at is *before* started_at, but monotonic-derived duration is 3.0 s.
    ended = _utc(2026, 5, 4, 11, 59, 55)
    monotonic_duration = 3.0

    out = RunOutcome.ok(
        run_id=0,
        output_paths={},
        started_at=started,
        ended_at=ended,
        duration_s=monotonic_duration,
    )
    assert out.duration_s == monotonic_duration
    # Sanity: the wall-clock delta is negative; the helper trusted the
    # caller's monotonic value rather than re-deriving.
    assert (ended - started).total_seconds() < 0
