"""Tests for gmat_sweep.backends.process_pool.ProcessPoolExecutorPool.

ProcessPoolExecutor — unlike joblib's ``n_jobs=1`` fast path — always
fans tasks out to a real subprocess, so the in-process monkey-patching
pattern :mod:`tests.test_backends_joblib` uses (driver-side fakes
running inside an in-process worker) would not reach the executor's
worker. The pool's wrapper logic (submit / as_completed orchestration,
error handling, dispatch selection) is unit-tested here by stubbing
:meth:`concurrent.futures.ProcessPoolExecutor.submit` so the test runs
entirely in the driver process. The full subprocess fan-out path is
exercised by :mod:`tests.test_backend_equivalence` and
:mod:`tests.test_backend_throughput` against real GMAT.
"""

from __future__ import annotations

import sys

import pytest

# The module under test raises RuntimeError at import on Python < 3.11
# (see gmat_sweep.backends.process_pool's import-time gate). Skip module
# collection on 3.10 with allow_module_level=True so the import below
# never runs there.
if sys.version_info < (3, 11):
    pytest.skip("ProcessPoolExecutorPool requires Python 3.11+", allow_module_level=True)

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.process_pool import ProcessPoolExecutorPool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

_StubCapture = list[tuple[Callable[[RunSpec], RunOutcome], RunSpec]]


def _make_spec(*, output_dir: Path, run_id: int = 0) -> RunSpec:
    return RunSpec(
        script_path=Path("/missions/m.script"),
        overrides={},
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


def _ok_outcome(run_id: int) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.ok(
        run_id=run_id, output_paths={}, started_at=now, ended_at=now, duration_s=0.0
    )


@pytest.fixture
def stubbed_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ProcessPoolExecutorPool, _StubCapture]]:
    """Pool whose underlying ``executor.submit`` returns a pre-set Future synchronously.

    Lets unit tests cover submit/as_completed orchestration without spawning
    a real subprocess. The captured list records every (task_fn, spec) tuple
    the pool dispatched, so dispatch-selection tests can assert which task_fn
    was picked.
    """
    captured: _StubCapture = []

    def _stub_submit(
        self: ProcessPoolExecutor,
        fn: Callable[[RunSpec], RunOutcome],
        spec: RunSpec,
    ) -> Future[RunOutcome]:
        captured.append((fn, spec))
        f: Future[RunOutcome] = Future()
        f.set_result(_ok_outcome(spec.run_id))
        return f

    monkeypatch.setattr(ProcessPoolExecutor, "submit", _stub_submit)

    pool = ProcessPoolExecutorPool(max_workers=1)
    try:
        yield pool, captured
    finally:
        pool.close()


def test_processpoolexecutorpool_is_pool_subclass() -> None:
    assert issubclass(ProcessPoolExecutorPool, Pool)
    assert ProcessPoolExecutorPool.subprocess_isolated is True


def test_submit_returns_pending_future(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    pool, _ = stubbed_pool
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()


def test_as_completed_yields_outcome_and_sets_future_result(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    pool, _ = stubbed_pool
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 0
    assert outcomes[0].status == "ok"
    assert f.done()
    assert f.result() is outcomes[0]


def test_as_completed_drains_multiple_specs(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    pool, _ = stubbed_pool
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(4)
    ]

    outcomes = list(pool.as_completed(futures))

    assert {o.run_id for o in outcomes} == {0, 1, 2, 3}
    for f in futures:
        assert f.done()


def test_as_completed_supports_multiple_drain_batches(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    pool, _ = stubbed_pool
    f0 = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f0]))

    f1 = pool.submit(_make_spec(output_dir=tmp_path / "run_1", run_id=1))
    list(pool.as_completed([f1]))

    assert f0.done() and f1.done()
    assert f0.result().run_id == 0
    assert f1.result().run_id == 1


def test_as_completed_rejects_unknown_future(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    pool, _ = stubbed_pool
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))  # Drained — no longer pending.

    with pytest.raises(BackendError):
        list(pool.as_completed([f]))


def test_close_is_idempotent() -> None:
    pool = ProcessPoolExecutorPool(max_workers=1)
    pool.close()
    pool.close()


def test_submit_after_close_raises(tmp_path: Path) -> None:
    pool = ProcessPoolExecutorPool(max_workers=1)
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises() -> None:
    pool = ProcessPoolExecutorPool(max_workers=1)
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_pool_as_context_manager_closes_on_exit(tmp_path: Path) -> None:
    with ProcessPoolExecutorPool(max_workers=1) as pool:
        pass
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_close_cancels_pending_futures(tmp_path: Path) -> None:
    pool = ProcessPoolExecutorPool(max_workers=1)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


def test_default_dispatches_run_one_directly(
    stubbed_pool: tuple[ProcessPoolExecutorPool, _StubCapture], tmp_path: Path
) -> None:
    """Default ``reuse_gmat_context=True`` dispatches ``run_one`` per task — fast path."""
    pool, captured = stubbed_pool
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))

    assert len(captured) == 1
    assert captured[0][0] is run_one


def test_reuse_gmat_context_false_still_dispatches_run_one_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``reuse_gmat_context=False`` still dispatches ``run_one`` directly.

    The pool hardcodes ``max_tasks_per_child=1``, so every task already
    runs in a fresh interpreter. Routing through
    ``run_spec_in_subprocess`` for the ``reuse_gmat_context=False`` mode
    would just double-pay the subprocess hop without changing the
    contract; the pool collapses both modes onto the direct ``run_one``
    path.
    """
    captured: _StubCapture = []

    def _stub_submit(
        self: ProcessPoolExecutor,
        fn: Callable[[RunSpec], RunOutcome],
        spec: RunSpec,
    ) -> Future[RunOutcome]:
        captured.append((fn, spec))
        f: Future[RunOutcome] = Future()
        f.set_result(_ok_outcome(spec.run_id))
        return f

    monkeypatch.setattr(ProcessPoolExecutor, "submit", _stub_submit)

    with ProcessPoolExecutorPool(max_workers=1, reuse_gmat_context=False) as pool:
        f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
        list(pool.as_completed([f]))

    assert len(captured) == 1
    assert captured[0][0] is run_one


def test_worker_side_exception_folds_into_failed_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A task whose future ``.result()`` raises is folded into ``RunOutcome.failed``.

    Simulates the transport-failure path (e.g. ``BrokenProcessPool``)
    where the executor's future surfaces an exception instead of the
    expected outcome. The drain loop must keep the sweep alive and
    capture the traceback on the synthetic failed outcome's ``stderr``.
    """

    def _stub_submit(
        self: ProcessPoolExecutor,
        fn: Callable[[RunSpec], RunOutcome],
        spec: RunSpec,
    ) -> Future[RunOutcome]:
        f: Future[RunOutcome] = Future()
        f.set_exception(RuntimeError("worker exploded"))
        return f

    monkeypatch.setattr(ProcessPoolExecutor, "submit", _stub_submit)

    with ProcessPoolExecutorPool(max_workers=1) as pool:
        f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
        outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 0
    assert outcomes[0].status == "failed"
    assert outcomes[0].stderr is not None
    assert "worker exploded" in outcomes[0].stderr
    assert f.done()
    assert f.result() is outcomes[0]


def test_one_failed_run_does_not_abort_other_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mixed batch: run 1 raises, runs 0 and 2 return outcomes — sweep yields all three."""
    submitted: list[RunSpec] = []

    def _stub_submit(
        self: ProcessPoolExecutor,
        fn: Callable[[RunSpec], RunOutcome],
        spec: RunSpec,
    ) -> Future[RunOutcome]:
        submitted.append(spec)
        f: Future[RunOutcome] = Future()
        if spec.run_id == 1:
            f.set_exception(RuntimeError("rank 1 down"))
        else:
            f.set_result(_ok_outcome(spec.run_id))
        return f

    monkeypatch.setattr(ProcessPoolExecutor, "submit", _stub_submit)

    with ProcessPoolExecutorPool(max_workers=1) as pool:
        futures = [
            pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
        ]
        outcomes = sorted(pool.as_completed(futures), key=lambda o: o.run_id)

    assert [o.run_id for o in outcomes] == [0, 1, 2]
    assert [o.status for o in outcomes] == ["ok", "failed", "ok"]
    assert outcomes[1].stderr is not None
    assert "rank 1 down" in outcomes[1].stderr
