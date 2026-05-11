"""Tests for gmat_sweep.backends.mpi.MPIPool — submit/as_completed semantics + ownership."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
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


def _ok_outcome(run_id: int) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.ok(
        run_id=run_id, output_paths={}, started_at=now, ended_at=now, duration_s=0.0
    )


class _StubMPIPoolExecutor:
    """Stand-in for ``mpi4py.futures.MPIPoolExecutor`` covering the surface MPIPool uses.

    Tasks resolve synchronously through ``fn(spec)`` and the resulting
    outcome lands on a ``concurrent.futures.Future`` so the test process
    drains in completion order without an MPI runtime. Tests that need to
    suppress the real ``fn`` (e.g. to keep gmatpy out of the unit test
    cell) monkeypatch ``run_one`` / ``_mpi_run_one_impl`` on the
    ``gmat_sweep.backends.mpi`` module before calling
    :meth:`MPIPool.as_completed`.
    """

    def __init__(self, *, max_workers: int | None = None, **kwargs: Any) -> None:
        self.max_workers = max_workers
        self.extra_kwargs = kwargs
        self.init_kwargs: dict[str, Any] = {}
        self.submitted: list[tuple[Any, RunSpec]] = []
        self.shutdown_calls: list[dict[str, Any]] = []

    def submit(self, fn: Any, spec: RunSpec) -> Future[RunOutcome]:
        self.submitted.append((fn, spec))
        future: Future[RunOutcome] = Future()
        try:
            future.set_result(fn(spec))
        except Exception as exc:
            # Mirror MPIPoolExecutor's behaviour: a worker-side exception
            # is parked on the future and surfaces from ``.result()``.
            future.set_exception(exc)
        return future

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdown_calls.append({"wait": wait})


def _install_stub_executor(monkeypatch: pytest.MonkeyPatch, executor: _StubMPIPoolExecutor) -> None:
    """Make `MPIPool.__init__`'s ``mpi4py.futures.MPIPoolExecutor(...)`` return ``executor``.

    The stub avoids needing an MPI runtime in the unit-test cell.
    """
    import types

    def _build_executor(**kwargs: Any) -> _StubMPIPoolExecutor:
        executor.init_kwargs = dict(kwargs)
        return executor

    fake_module = types.SimpleNamespace(MPIPoolExecutor=_build_executor)
    fake_pkg = types.SimpleNamespace(futures=fake_module)
    monkeypatch.setitem(sys.modules, "mpi4py", fake_pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", fake_module)


def test_mpipool_is_pool_subclass() -> None:
    """``MPIPool`` honours the ``Pool`` ABC contract at the class level.

    ``subprocess_isolated=True`` proves the pool implements both reuse and
    isolation modes — see :class:`gmat_sweep.backends.base.Pool` for the
    contract enforced via ``__init_subclass__``.
    """
    from gmat_sweep.backends.mpi import MPIPool

    assert issubclass(MPIPool, Pool)
    assert MPIPool.subprocess_isolated is True


def test_subclass_setting_subprocess_isolated_false_rejected() -> None:
    """The Pool ABC's contract still applies to ``MPIPool`` subclasses."""
    from gmat_sweep.backends.mpi import MPIPool

    with pytest.raises(BackendError):

        class _Bad(MPIPool):  # pragma: no cover - body never runs
            subprocess_isolated: ClassVar[bool] = False

            def submit(self, spec: RunSpec) -> Future[RunOutcome]:
                return Future()

            def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
                return iter([])

            def close(self) -> None:
                pass


def test_lazy_import_raises_backenderror_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``mpi4py.futures`` cannot be imported, ``MPIPool()`` raises ``BackendError``."""
    from gmat_sweep.backends.mpi import MPIPool

    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", None)
    with pytest.raises(BackendError) as ei:
        MPIPool()
    assert "[mpi]" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_default_init_constructs_executor_and_owns_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default construction calls ``MPIPoolExecutor(max_workers=None)``."""
    from gmat_sweep.backends.mpi import MPIPool

    executor = _StubMPIPoolExecutor()
    _install_stub_executor(monkeypatch, executor)
    pool = MPIPool()
    assert pool._executor is executor
    pool.close()
    assert executor.shutdown_calls == [{"wait": True}]


def test_init_forwards_max_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_workers`` is forwarded verbatim to the executor."""
    from gmat_sweep.backends.mpi import MPIPool

    executor = _StubMPIPoolExecutor()
    _install_stub_executor(monkeypatch, executor)
    pool = MPIPool(max_workers=4)
    assert executor.init_kwargs["max_workers"] == 4
    pool.close()


def test_init_forwards_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra constructor kwargs are forwarded to ``MPIPoolExecutor``."""
    from gmat_sweep.backends.mpi import MPIPool

    executor = _StubMPIPoolExecutor()
    _install_stub_executor(monkeypatch, executor)
    pool = MPIPool(max_workers=2, path=("/opt/extra/lib",))
    assert executor.init_kwargs == {
        "max_workers": 2,
        "path": ("/opt/extra/lib",),
    }
    pool.close()


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    executor = _StubMPIPoolExecutor()
    _install_stub_executor(monkeypatch, executor)
    pool = MPIPool()
    pool.close()
    pool.close()
    # Second close is a no-op — shutdown is called exactly once.
    assert executor.shutdown_calls == [{"wait": True}]


def test_submit_returns_pending_future(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    pool = MPIPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()
    pool.close()


def test_submit_after_close_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    pool = MPIPool()
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    pool = MPIPool()
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_close_cancels_pending_futures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    pool = MPIPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


def test_as_completed_dispatches_via_executor_and_yields_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """as_completed submits one MPI task per parked spec; outcomes flow back."""
    from gmat_sweep.backends.mpi import MPIPool

    executor = _StubMPIPoolExecutor()
    _install_stub_executor(monkeypatch, executor)
    # Stub the task fn so the unit-test cell never imports gmatpy via the real run_one.
    monkeypatch.setattr(
        "gmat_sweep.backends.mpi.run_one",
        lambda spec: _ok_outcome(spec.run_id),
    )

    pool = MPIPool()
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
    ]
    outcomes = list(pool.as_completed(futures))

    submitted_run_ids = {spec.run_id for _, spec in executor.submitted}
    assert submitted_run_ids == {0, 1, 2}
    assert {o.run_id for o in outcomes} == {0, 1, 2}
    for f in futures:
        assert f.done()
        assert f.result().status == "ok"
    pool.close()


def test_as_completed_rejects_unknown_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    pool = MPIPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))
    with pytest.raises(BackendError):
        list(pool.as_completed([f]))
    pool.close()


def test_worker_side_exception_folds_into_failed_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rank-side exception is folded into a synthetic ``RunOutcome.failed``.

    Simulates the MPI rank-crash path where the executor's future surfaces
    an exception instead of an outcome. The drain loop must keep the sweep
    alive and capture the traceback on the synthetic failed outcome's
    ``stderr``.
    """
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())

    def _boom(_spec: RunSpec) -> RunOutcome:
        raise RuntimeError("rank exploded")

    monkeypatch.setattr("gmat_sweep.backends.mpi.run_one", _boom)

    pool = MPIPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    outcomes = list(pool.as_completed([f]))
    pool.close()

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 0
    assert outcomes[0].status == "failed"
    assert outcomes[0].stderr is not None
    assert "rank exploded" in outcomes[0].stderr
    assert f.done()
    assert f.result() is outcomes[0]


def test_one_failed_rank_does_not_abort_other_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mix one rank-crashing task with two successful ones — sweep yields all three."""
    from gmat_sweep.backends.mpi import MPIPool

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())

    def _maybe_boom(spec: RunSpec) -> RunOutcome:
        if spec.run_id == 1:
            raise RuntimeError("rank 1 down")
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.mpi.run_one", _maybe_boom)

    pool = MPIPool()
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
    ]
    outcomes = sorted(pool.as_completed(futures), key=lambda o: o.run_id)
    pool.close()

    assert [o.run_id for o in outcomes] == [0, 1, 2]
    assert [o.status for o in outcomes] == ["ok", "failed", "ok"]
    assert outcomes[1].stderr is not None
    assert "rank 1 down" in outcomes[1].stderr


def test_mpi_run_one_delegates_to_subprocess_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_mpi_run_one_impl`` is a thin wrapper around ``run_spec_in_subprocess``."""
    from gmat_sweep.backends.mpi import _mpi_run_one_impl

    captured: dict[str, RunSpec] = {}

    def _fake(spec: RunSpec) -> RunOutcome:
        captured["spec"] = spec
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.mpi.run_spec_in_subprocess", _fake)
    spec = _make_spec(output_dir=tmp_path / "run_0", run_id=42)
    outcome = _mpi_run_one_impl(spec)
    assert captured["spec"] is spec
    assert outcome.run_id == 42


def test_default_dispatches_run_one_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default ``reuse_gmat_context=True`` submits ``run_one`` to the executor."""
    from gmat_sweep.backends.mpi import MPIPool

    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_mpi_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("_mpi_run_one_impl", spec.run_id))
        return _ok_outcome(spec.run_id)

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    monkeypatch.setattr("gmat_sweep.backends.mpi.run_one", _fake_run_one)
    monkeypatch.setattr("gmat_sweep.backends.mpi._mpi_run_one_impl", _fake_mpi_run_one)

    pool = MPIPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))
    pool.close()

    assert calls == [("run_one", 0)]


def test_reuse_gmat_context_false_dispatches_subprocess_hop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``reuse_gmat_context=False`` submits ``_mpi_run_one_impl`` to the executor."""
    from gmat_sweep.backends.mpi import MPIPool

    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_mpi_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("_mpi_run_one_impl", spec.run_id))
        return _ok_outcome(spec.run_id)

    _install_stub_executor(monkeypatch, _StubMPIPoolExecutor())
    monkeypatch.setattr("gmat_sweep.backends.mpi.run_one", _fake_run_one)
    monkeypatch.setattr("gmat_sweep.backends.mpi._mpi_run_one_impl", _fake_mpi_run_one)

    pool = MPIPool(reuse_gmat_context=False)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))
    pool.close()

    assert calls == [("_mpi_run_one_impl", 0)]


def test_importing_mpi_module_does_not_import_gmatpy() -> None:
    """Loading ``gmat_sweep.backends.mpi`` in a fresh interpreter must not import gmatpy.

    The MPI worker rank inherits whatever the module top-level imports
    pulled in. If anything in the import chain triggered gmatpy, the
    subprocess-isolation contract would be silently violated.
    """
    code = (
        "import sys\n"
        "import gmat_sweep.backends.mpi  # noqa: F401\n"
        "assert 'gmatpy' not in sys.modules, sorted(m for m in sys.modules if 'gmat' in m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
