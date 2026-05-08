"""Tests for gmat_sweep.backends.debug.DebugPool — in-process, single-run dispatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.debug import DebugPool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunSpec
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeResults


def _make_spec(*, output_dir: Path, run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=Path("/missions/m.script"),
        overrides={},
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT script\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _make_runs(script: Path, output_dir: Path, n: int) -> list[RunSpec]:
    return [
        RunSpec(
            script_path=script,
            overrides={},
            output_dir=output_dir / f"run-{i}",
            run_id=i,
            seed=None,
            run_options={},
        )
        for i in range(n)
    ]


def test_debugpool_is_pool_subclass_with_debug_sentinel() -> None:
    assert issubclass(DebugPool, Pool)
    assert DebugPool.subprocess_isolated == "debug"


def test_debugpool_construction_requires_explicit_optin() -> None:
    with pytest.raises(BackendError) as ei:
        DebugPool()
    assert "allow_unisolated_pool" in str(ei.value)


def test_debugpool_constructs_when_optin_passed() -> None:
    pool = DebugPool(allow_unisolated_pool=True)
    pool.close()


def test_debugpool_runs_specs_in_driver_process(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    # Runs in-process iff the worker observes the driver's pid.
    observed_pids: list[int] = []

    def _record_pid(**_: Any) -> FakeResults:
        observed_pids.append(os.getpid())
        return FakeResults()

    fake_gmat_run.install_loader(run_hook=_record_pid)

    pool = DebugPool(allow_unisolated_pool=True)
    spec = _make_spec(output_dir=tmp_path / "run_0")
    f = pool.submit(spec)
    outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].status == "ok"
    assert f.done()
    assert f.result() is outcomes[0]
    assert observed_pids == [os.getpid()]


def test_debugpool_user_breakpoint_drops_into_driver_debugger(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reason DebugPool exists: a breakpoint() reached during the run
    # must hit the driver's sys.breakpointhook. Patch the hook to a recorder
    # and have the fake mission's run() call breakpoint() while the pool is
    # active; assert the recorder fired in this process.
    breakpoint_hits: list[int] = []

    def _record_breakpoint(*_args: Any, **_kwargs: Any) -> None:
        breakpoint_hits.append(os.getpid())

    monkeypatch.setattr(sys, "breakpointhook", _record_breakpoint)

    def _trigger_breakpoint(**_: Any) -> FakeResults:
        breakpoint()
        return FakeResults()

    fake_gmat_run.install_loader(run_hook=_trigger_breakpoint)

    pool = DebugPool(allow_unisolated_pool=True)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))

    assert breakpoint_hits == [os.getpid()]


def test_debugpool_close_is_idempotent() -> None:
    pool = DebugPool(allow_unisolated_pool=True)
    pool.close()
    pool.close()


def test_debugpool_submit_after_close_raises(tmp_path: Path) -> None:
    pool = DebugPool(allow_unisolated_pool=True)
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_debugpool_as_completed_rejects_unknown_future(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    pool = DebugPool(allow_unisolated_pool=True)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))  # Drained — no longer pending.

    with pytest.raises(BackendError):
        list(pool.as_completed([f]))


def test_debugpool_lazy_export_from_backends_package() -> None:
    # The pool is registered on the backends package so
    # `from gmat_sweep.backends import DebugPool` works without forcing an
    # eager import on plain `from gmat_sweep.backends import LocalJoblibPool`.
    from gmat_sweep import backends

    assert backends.DebugPool is DebugPool


# ---- Sweep-level gates ----------------------------------------------------


def test_sweep_rejects_debugpool_without_acknowledgement(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    runs = _make_runs(script, tmp_path / "out", n=1)
    pool = DebugPool(allow_unisolated_pool=True)
    try:
        with pytest.raises(BackendError) as ei:
            Sweep(
                runs=runs,
                backend=pool,
                manifest_path=tmp_path / "out" / "manifest.jsonl",
                output_dir=tmp_path / "out",
                script_path=script,
                parameter_spec={"_kind": "explicit", "columns": [], "rows": [[]]},
                progress=False,
            )
        assert "allow_unisolated_pool" in str(ei.value)
    finally:
        pool.close()


def test_sweep_runs_one_spec_through_debugpool(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with DebugPool(allow_unisolated_pool=True) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"_kind": "explicit", "columns": [], "rows": [[]]},
            progress=False,
            allow_unisolated_pool=True,
        )
        sweep.run()

    manifest = sweep.to_manifest()
    assert manifest.run_count == 1
    assert len(manifest.entries) == 1
    assert manifest.entries[0].run_id == 0


def test_sweep_run_rejects_multi_spec_through_debugpool(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=2)

    with DebugPool(allow_unisolated_pool=True) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"_kind": "explicit", "columns": [], "rows": [[], []]},
            progress=False,
            allow_unisolated_pool=True,
        )
        with pytest.raises(BackendError) as ei:
            sweep.run()
        assert "DebugPool" in str(ei.value)
        assert "2" in str(ei.value)
