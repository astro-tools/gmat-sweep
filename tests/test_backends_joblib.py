"""Tests for gmat_sweep.backends.joblib.LocalJoblibPool — submit/as_completed semantics."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from tests.conftest import FakeGmatRun


def _make_spec(*, output_dir: Path, run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=Path("/missions/m.script"),
        overrides={},
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


# Tests run with workers=1 so joblib dispatches in-process; the existing
# fake_gmat_run fixture (driver-side monkeypatch of sys.modules['gmat_run'])
# would not reach loky subprocesses. The full subprocess fan-out path is
# exercised by integration tests against real GMAT (deferred to #11).
@pytest.fixture
def pool() -> Iterator[LocalJoblibPool]:
    p = LocalJoblibPool(workers=1)
    yield p
    p.close()


def test_localjoblibpool_is_pool_subclass() -> None:
    assert issubclass(LocalJoblibPool, Pool)
    assert LocalJoblibPool.subprocess_isolated is True


@pytest.mark.parametrize("workers", [0, -2, -100])
def test_invalid_workers_raises(workers: int) -> None:
    with pytest.raises(BackendError):
        LocalJoblibPool(workers=workers)


def test_submit_returns_pending_future(pool: LocalJoblibPool, tmp_path: Path) -> None:
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()


def test_as_completed_yields_outcome_and_sets_future_result(
    pool: LocalJoblibPool, tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 0
    assert outcomes[0].status == "ok"
    assert f.done()
    assert f.result() is outcomes[0]


def test_as_completed_drains_multiple_specs_and_writes_per_run_logs(
    pool: LocalJoblibPool, tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(4)
    ]

    outcomes = list(pool.as_completed(futures))

    assert {o.run_id for o in outcomes} == {0, 1, 2, 3}
    for f in futures:
        assert f.done()
    # Each worker writes <output_dir>/run_<run_id>/worker.log per the issue's
    # acceptance bullet — confirm the path comes out where the orchestrator
    # routed spec.output_dir.
    for i in range(4):
        assert (tmp_path / f"run_{i}" / "worker.log").exists()


def test_as_completed_supports_multiple_drain_batches(
    pool: LocalJoblibPool, tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    f0 = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f0]))

    f1 = pool.submit(_make_spec(output_dir=tmp_path / "run_1", run_id=1))
    list(pool.as_completed([f1]))

    assert f0.done() and f1.done()
    assert f0.result().run_id == 0
    assert f1.result().run_id == 1


def test_as_completed_rejects_unknown_future(
    pool: LocalJoblibPool, tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))  # Drained — no longer pending.

    with pytest.raises(BackendError):
        list(pool.as_completed([f]))


def test_close_is_idempotent() -> None:
    pool = LocalJoblibPool(workers=1)
    pool.close()
    pool.close()


def test_submit_after_close_raises(tmp_path: Path) -> None:
    pool = LocalJoblibPool(workers=1)
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises() -> None:
    pool = LocalJoblibPool(workers=1)
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_pool_as_context_manager_closes_on_exit(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    with LocalJoblibPool(workers=1) as pool:
        f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
        list(pool.as_completed([f]))
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_1"))


def test_close_cancels_pending_futures(tmp_path: Path) -> None:
    pool = LocalJoblibPool(workers=1)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


def _ok_outcome(run_id: int) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.ok(run_id=run_id, output_paths={}, started_at=now, ended_at=now)


def test_default_dispatches_run_one_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default ``reuse_gmat_context=True`` calls ``run_one`` per task — fast path."""
    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_run_spec_in_subprocess(spec: RunSpec) -> RunOutcome:
        calls.append(("run_spec_in_subprocess", spec.run_id))
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.joblib.run_one", _fake_run_one)
    monkeypatch.setattr(
        "gmat_sweep.backends.joblib.run_spec_in_subprocess", _fake_run_spec_in_subprocess
    )

    with LocalJoblibPool(workers=1) as pool:
        f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
        list(pool.as_completed([f]))

    assert calls == [("run_one", 0)]


def test_reuse_gmat_context_false_dispatches_subprocess_hop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``reuse_gmat_context=False`` calls ``run_spec_in_subprocess`` per task."""
    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_run_spec_in_subprocess(spec: RunSpec) -> RunOutcome:
        calls.append(("run_spec_in_subprocess", spec.run_id))
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.joblib.run_one", _fake_run_one)
    monkeypatch.setattr(
        "gmat_sweep.backends.joblib.run_spec_in_subprocess", _fake_run_spec_in_subprocess
    )

    with LocalJoblibPool(workers=1, reuse_gmat_context=False) as pool:
        f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
        list(pool.as_completed([f]))

    assert calls == [("run_spec_in_subprocess", 0)]
