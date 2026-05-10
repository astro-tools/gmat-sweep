"""Failure-mode coverage: a single bad run does not abort the parent sweep.

The four scenarios #11 calls out:

- **Invalid override** — applies a value GMAT rejects (an unknown dotted path).
  Real-GMAT path; the worker catches and labels.
- **Divergent solver** — exercised via a custom :class:`Pool` that hands the
  driver a synthetic :class:`RunOutcome.failed` carrying a representative
  "DC: did not converge" stderr. Standing up a deterministically-divergent
  Differential Corrector fixture in GMAT script is fiddly and adds wall-clock
  cost without changing what's being asserted: that the *driver / aggregator*
  surfaces a failed solver run consistently.
- **Bad script path** — the worker can't load the script (e.g. file deleted
  between sweep construction and worker dispatch). Simulated via the same
  synthetic pool: the v0.1 driver computes the script's canonical SHA up front
  in :meth:`Sweep._build_manifest`, so a script that's missing at
  *sweep-construction* time fails the parent sweep eagerly — that's the
  intended driver-side guardrail. The integration scenario the issue cares
  about is the *worker-side* "I can't find the script" path, where the
  per-run :class:`FileNotFoundError` becomes a failed outcome and the parent
  sweep completes. The real-GMAT side of this is already covered in
  ``test_worker.py::test_run_one_failed_when_load_raises``.
- **Simulated worker OOM** — also via a custom :class:`Pool` returning a
  synthetic ``MemoryError`` traceback. The issue tags this scenario "simulated"
  because reliably triggering an OOM in a worker subprocess is platform- and
  ulimit-dependent.

Every scenario asserts the same three things:

1. The aggregated DataFrame has a row for the failed run with ``__status`` set
   to ``"failed"``.
2. The manifest entry for that run carries non-empty ``stderr``.
3. The parent ``sweep()`` call completes without raising.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import gmat_sweep
from gmat_sweep.aggregate import lazy_multiindex
from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.grids import expand_grid_to_run_specs
from gmat_sweep.manifest import Manifest
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.sweep import Sweep

# ---- shared assertions --------------------------------------------------


def _failed_run_ids(df: pd.DataFrame) -> list[int]:
    failed_rows = df.loc[df["__status"] == "failed"]
    return sorted(failed_rows.index.get_level_values("run_id").unique().tolist())


def _assert_failed_run_has_stderr(manifest: Manifest, run_id: int) -> str:
    entry = next(e for e in manifest.entries if e.run_id == run_id)
    assert entry.status == "failed"
    assert entry.stderr is not None
    assert entry.stderr.strip() != ""
    return entry.stderr


# ---- scenario 1: invalid override (real GMAT) --------------------------


@pytest.mark.integration
def test_invalid_override_yields_failed_status_and_completes_sweep(
    leo_basic_script: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    df = gmat_sweep.sweep(
        leo_basic_script,
        grid={"Sat.NotARealField": [1.0, 2.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    assert _failed_run_ids(df) == [0, 1]
    manifest = Manifest.load(out / "manifest.jsonl")
    for run_id in (0, 1):
        stderr = _assert_failed_run_has_stderr(manifest, run_id)
        assert "NotARealField" in stderr or "field" in stderr.lower()


@pytest.mark.integration
def test_one_invalid_override_does_not_abort_other_runs(
    leo_basic_script: Path, tmp_path: Path
) -> None:
    """Mix one bad override with two good ones — the parent sweep still ships a frame."""
    # SMA=0 is a deterministic invariant violation; the propagator bails and
    # the worker captures the resulting GmatRunError into the failed outcome.
    out = tmp_path / "out"
    df = gmat_sweep.sweep(
        leo_basic_script,
        grid={"Sat.SMA": [7000.0, 0.0, 7100.0]},
        backend=LocalJoblibPool(max_workers=2),
        out=out,
    )

    statuses_by_run = df["__status"].groupby(level="run_id").first().to_dict()
    assert statuses_by_run[0] == "ok"
    assert statuses_by_run[1] == "failed"
    assert statuses_by_run[2] == "ok"

    manifest = Manifest.load(out / "manifest.jsonl")
    _assert_failed_run_has_stderr(manifest, run_id=1)


# ---- scenario 2: bad script path (simulated) --------------------------


def test_bad_script_path_yields_failed_status_in_aggregated_frame(
    tmp_path: Path,
) -> None:
    """A worker that can't load the script surfaces as a failed run; sweep completes."""
    missing_path = tmp_path / "missing.script"
    bad_path_stderr = (
        "Traceback (most recent call last):\n"
        '  File ".../gmat_sweep/worker.py", line ..., in run_one\n'
        "    mission = gmat_run.Mission.load(spec.script_path)\n"
        f"FileNotFoundError: [Errno 2] No such file or directory: {str(missing_path)!r}"
    )
    df, manifest = _drive_synthetic_sweep(
        grid={"x": [1, 2]},
        output_dir=tmp_path / "out",
        script_path=tmp_path / "stub.script",
        factory={
            0: lambda spec: _failed_outcome(spec, bad_path_stderr),
            1: lambda spec: _failed_outcome(spec, bad_path_stderr),
        },
    )

    assert _failed_run_ids(df) == [0, 1]
    stderr = _assert_failed_run_has_stderr(manifest, run_id=0)
    assert "missing.script" in stderr
    assert "FileNotFoundError" in stderr


# ---- scenario 3 + 4 plumbing: synthetic-failure pool --------------------


class _SyntheticFailingPool(Pool):
    """Hand the driver a stream of pre-built outcomes — no subprocess, no GMAT.

    Used for the divergent-solver and worker-OOM scenarios where what we want
    to assert is the driver / aggregator path, not the worker's exception
    handling (already covered in test_worker.py).
    """

    def __init__(self, outcome_factory: dict[int, Any]) -> None:
        self._submitted: list[RunSpec] = []
        self._factory = outcome_factory

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        self._submitted.append(spec)
        return Future()

    def as_completed(self, _futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        for spec in self._submitted:
            yield self._factory[spec.run_id](spec)

    def close(self) -> None:
        pass


def _ok_outcome(spec: RunSpec, output_paths: dict[str, Path]) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.ok(
        run_id=spec.run_id, output_paths=output_paths, started_at=now, ended_at=now
    )


def _failed_outcome(spec: RunSpec, stderr: str) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.failed(run_id=spec.run_id, stderr=stderr, started_at=now, ended_at=now)


def _drive_synthetic_sweep(
    *,
    grid: dict[str, list[Any]],
    output_dir: Path,
    script_path: Path,
    factory: dict[int, Any],
) -> tuple[pd.DataFrame, Manifest]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sweep._build_manifest hashes the script file at sweep-construction time;
    # the synthetic-pool tests don't need real GMAT but they do need a readable
    # file to satisfy that hash. A trivial stub is enough.
    script_path.parent.mkdir(parents=True, exist_ok=True)
    if not script_path.exists():
        script_path.write_text("% stub script for synthetic-pool tests\n", encoding="utf-8")
    runs = expand_grid_to_run_specs(grid, script_path, output_dir)
    pool = _SyntheticFailingPool(factory)
    sweep = Sweep(
        runs=runs,
        backend=pool,
        manifest_path=output_dir / "manifest.jsonl",
        output_dir=output_dir,
        script_path=script_path,
        parameter_spec=grid,
        progress=False,
    ).run()
    df = lazy_multiindex(sweep.to_manifest(), output_dir)
    return df, sweep.to_manifest()


# ---- scenario 3: divergent solver (simulated) --------------------------


def test_divergent_solver_run_surfaces_as_failed_in_aggregated_frame(
    tmp_path: Path,
) -> None:
    diverged_stderr = (
        "Traceback (most recent call last):\n"
        '  File "...", line ..., in run_one\n'
        '    raise GmatRunError("DC1: did not converge after 25 iterations")\n'
        "gmat_run.errors.GmatRunError: DC1: did not converge after 25 iterations"
    )
    df, manifest = _drive_synthetic_sweep(
        grid={"x": [1, 2]},
        output_dir=tmp_path / "out",
        script_path=tmp_path / "stub.script",
        factory={
            0: lambda spec: _failed_outcome(spec, diverged_stderr),
            1: lambda spec: _failed_outcome(spec, diverged_stderr),
        },
    )

    assert _failed_run_ids(df) == [0, 1]
    stderr = _assert_failed_run_has_stderr(manifest, run_id=0)
    assert "did not converge" in stderr


# ---- scenario 4: simulated worker OOM ----------------------------------


def test_worker_oom_run_surfaces_as_failed_and_other_runs_succeed(
    tmp_path: Path,
) -> None:
    """OOM on run 1, OK on runs 0 and 2 — sweep returns a frame with all three."""
    # A representative MemoryError traceback the loky / pickle layer might surface.
    oom_stderr = (
        "MemoryError\n"
        "Traceback (most recent call last):\n"
        '  File ".../joblib/externals/loky/process_executor.py", line ..., in _process_worker\n'
        "    r = call_item.fn(*call_item.args, **call_item.kwargs)\n"
        "MemoryError"
    )

    # Make runs 0 and 2 succeed by emitting tiny per-run Parquets the
    # aggregator can read; run 1 fails synthetically.
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    ok_paths: dict[int, dict[str, Path]] = {}
    for rid in (0, 2):
        run_dir = output_dir / f"run-{rid}"
        run_dir.mkdir()
        path = run_dir / "R.parquet"
        pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T00:00:30"]),
                "x": [float(rid), float(rid) + 0.5],
            }
        ).to_parquet(path)
        ok_paths[rid] = {"R": path}

    df, manifest = _drive_synthetic_sweep(
        grid={"x": [1, 2, 3]},
        output_dir=output_dir,
        script_path=tmp_path / "stub.script",
        factory={
            0: lambda spec: _ok_outcome(spec, ok_paths[0]),
            1: lambda spec: _failed_outcome(spec, oom_stderr),
            2: lambda spec: _ok_outcome(spec, ok_paths[2]),
        },
    )

    assert _failed_run_ids(df) == [1]
    stderr = _assert_failed_run_has_stderr(manifest, run_id=1)
    assert "MemoryError" in stderr

    statuses_by_run = df["__status"].groupby(level="run_id").first().to_dict()
    assert statuses_by_run == {0: "ok", 1: "failed", 2: "ok"}
