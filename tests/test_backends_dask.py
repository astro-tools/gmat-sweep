"""Tests for gmat_sweep.backends.dask.DaskPool — submit/as_completed semantics + ownership."""

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

distributed = pytest.importorskip("distributed")


from gmat_sweep.backends.dask import DaskPool, _dask_run_one  # noqa: E402


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


class _StubFuture:
    """Minimal stand-in for ``distributed.Future`` carrying a precomputed result.

    ``exc`` (optional) lets the stub surface an exception via the
    ``as_completed(... raise_errors=False)`` path — see
    :func:`_stub_as_completed`.
    """

    _next_key = 0

    def __init__(self, value: RunOutcome | None, *, exc: BaseException | None = None) -> None:
        self._value = value
        self._exc = exc
        # Mirror distributed.Future.key — DaskPool keys its run_id lookup on it.
        _StubFuture._next_key += 1
        self.key = f"stub-{_StubFuture._next_key}"

    def result(self) -> RunOutcome:
        if self._exc is not None:
            raise self._exc
        assert self._value is not None
        return self._value


class _StubClient:
    """Stand-in for ``distributed.Client`` covering only the surface DaskPool uses."""

    def __init__(self) -> None:
        self.submitted: list[RunSpec] = []
        self.closed = False

    def submit(self, fn: Any, spec: RunSpec, *, pure: bool = True) -> _StubFuture:
        self.submitted.append(spec)
        return _StubFuture(fn(spec))

    def close(self) -> None:
        self.closed = True


class _StubCluster:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _stub_as_completed(
    futures: Iterable[_StubFuture],
    *,
    with_results: bool = False,
    raise_errors: bool = True,
) -> Iterator[Any]:
    """Stand-in for ``distributed.as_completed`` covering the kwargs DaskPool uses.

    Mirrors the real distributed.as_completed surface: ``with_results=True``
    yields ``(future, payload)`` pairs; ``raise_errors=False`` surfaces a
    task's exception as the payload instead of re-raising. The drain code
    in :class:`DaskPool.as_completed` depends on both knobs together to
    fold worker-side exceptions.
    """
    for f in futures:
        if with_results:
            try:
                value = f.result()
            except Exception as exc:
                if raise_errors:
                    raise
                yield f, exc
                continue
            yield f, value
        else:
            yield f


def _patch_distributed_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cluster_factory: Any = None,
    client_factory: Any = None,
) -> None:
    """Replace distributed.LocalCluster, distributed.Client, distributed.as_completed."""
    if cluster_factory is None:
        cluster_factory = lambda **kwargs: _StubCluster()  # noqa: E731
    if client_factory is None:
        client_factory = lambda *args, **kwargs: _StubClient()  # noqa: E731

    monkeypatch.setattr(distributed, "LocalCluster", cluster_factory, raising=True)
    monkeypatch.setattr(distributed, "Client", client_factory, raising=True)
    monkeypatch.setattr(distributed, "as_completed", _stub_as_completed, raising=True)


def test_daskpool_is_pool_subclass() -> None:
    assert issubclass(DaskPool, Pool)
    assert DaskPool.subprocess_isolated is True


def test_subclass_setting_subprocess_isolated_false_rejected() -> None:
    """The Pool ABC's contract still applies to DaskPool subclasses."""
    with pytest.raises(BackendError):

        class _Bad(DaskPool):  # pragma: no cover - body never runs
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
    """If `distributed` cannot be imported, DaskPool() raises BackendError from ImportError."""
    monkeypatch.setitem(sys.modules, "distributed", None)
    with pytest.raises(BackendError) as ei:
        DaskPool()
    assert "[dask]" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_default_init_owns_cluster_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _StubCluster()
    client = _StubClient()
    _patch_distributed_with_stubs(
        monkeypatch,
        cluster_factory=lambda **kwargs: cluster,
        client_factory=lambda *_args, **_kwargs: client,
    )
    pool = DaskPool()
    assert pool._owns_cluster is True
    assert pool._owns_client is True
    pool.close()
    assert cluster.closed
    assert client.closed


def test_close_does_not_touch_user_supplied_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    user_client = _StubClient()
    pool = DaskPool(client=user_client)
    assert pool._owns_cluster is False
    assert pool._owns_client is False
    pool.close()
    assert user_client.closed is False


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool()
    pool.close()
    pool.close()


def test_submit_returns_pending_future(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_StubClient())
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()
    pool.close()


def test_submit_after_close_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_StubClient())
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_StubClient())
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_close_cancels_pending_futures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_StubClient())
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


def test_as_completed_dispatches_via_client_and_yields_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """as_completed submits one Dask task per parked spec; futures resolve in completion order."""
    submitted: list[RunSpec] = []

    class _RecordingClient:
        def submit(self, fn: Any, spec: RunSpec, *, pure: bool = True) -> _StubFuture:
            submitted.append(spec)
            return _StubFuture(_ok_outcome(spec.run_id))

        def close(self) -> None:
            pass

    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_RecordingClient())
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
    ]
    outcomes = list(pool.as_completed(futures))

    assert {s.run_id for s in submitted} == {0, 1, 2}
    assert {o.run_id for o in outcomes} == {0, 1, 2}
    for f in futures:
        assert f.done()
        assert f.result().status == "ok"
    pool.close()


def test_as_completed_rejects_unknown_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_StubClient())
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))
    with pytest.raises(BackendError):
        list(pool.as_completed([f]))
    pool.close()


def test_dask_run_one_delegates_to_subprocess_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_dask_run_one is a thin wrapper around run_spec_in_subprocess."""
    captured: dict[str, RunSpec] = {}

    def _fake(spec: RunSpec) -> RunOutcome:
        captured["spec"] = spec
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.dask.run_spec_in_subprocess", _fake)
    spec = _make_spec(output_dir=tmp_path / "run_0", run_id=42)
    outcome = _dask_run_one(spec)
    assert captured["spec"] is spec
    assert outcome.run_id == 42


def test_default_dispatches_run_one_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default ``reuse_gmat_context=True`` submits ``run_one`` to the Dask client."""
    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_dask_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("_dask_run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    _patch_distributed_with_stubs(monkeypatch)
    monkeypatch.setattr("gmat_sweep.backends.dask.run_one", _fake_run_one)
    monkeypatch.setattr("gmat_sweep.backends.dask._dask_run_one", _fake_dask_run_one)

    pool = DaskPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))
    pool.close()

    assert calls == [("run_one", 0)]


def test_worker_side_exception_folds_into_failed_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Dask task that raises is folded into ``RunOutcome.failed``.

    The drain loop runs ``distributed.as_completed`` with
    ``raise_errors=False`` and the worker-side exception arrives as the
    payload instead of propagating out of ``.result()``. The fold must
    keep the sweep alive and surface the traceback on ``stderr``.
    """

    class _ExplodingClient:
        def submit(self, fn: Any, spec: RunSpec, *, pure: bool = True) -> _StubFuture:
            # Park the exception on the future via the new exc= kwarg —
            # the stub_as_completed helper surfaces it as the payload.
            return _StubFuture(None, exc=RuntimeError("worker exploded"))

        def close(self) -> None:
            pass

    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_ExplodingClient())
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    outcomes = list(pool.as_completed([f]))
    pool.close()

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
    """Mix one task that raises with two that succeed — sweep yields all three."""

    class _MixedClient:
        def submit(self, fn: Any, spec: RunSpec, *, pure: bool = True) -> _StubFuture:
            if spec.run_id == 1:
                return _StubFuture(None, exc=RuntimeError("worker 1 down"))
            return _StubFuture(_ok_outcome(spec.run_id))

        def close(self) -> None:
            pass

    _patch_distributed_with_stubs(monkeypatch)
    pool = DaskPool(client=_MixedClient())
    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
    ]
    outcomes = sorted(pool.as_completed(futures), key=lambda o: o.run_id)
    pool.close()

    assert [o.run_id for o in outcomes] == [0, 1, 2]
    assert [o.status for o in outcomes] == ["ok", "failed", "ok"]
    assert outcomes[1].stderr is not None
    assert "worker 1 down" in outcomes[1].stderr


def test_reuse_gmat_context_false_dispatches_subprocess_hop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``reuse_gmat_context=False`` submits ``_dask_run_one`` to the Dask client."""
    calls: list[tuple[str, int]] = []

    def _fake_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    def _fake_dask_run_one(spec: RunSpec) -> RunOutcome:
        calls.append(("_dask_run_one", spec.run_id))
        return _ok_outcome(spec.run_id)

    _patch_distributed_with_stubs(monkeypatch)
    monkeypatch.setattr("gmat_sweep.backends.dask.run_one", _fake_run_one)
    monkeypatch.setattr("gmat_sweep.backends.dask._dask_run_one", _fake_dask_run_one)

    pool = DaskPool(reuse_gmat_context=False)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))
    pool.close()

    assert calls == [("_dask_run_one", 0)]


def test_importing_dask_module_does_not_import_gmatpy() -> None:
    """Loading gmat_sweep.backends.dask in a fresh interpreter must not import gmatpy.

    The Dask worker process inherits whatever the module top-level imports
    pulled in. If anything in the import chain triggered gmatpy, the
    subprocess-isolation contract would be silently violated.
    """
    code = (
        "import sys\n"
        "import gmat_sweep.backends.dask  # noqa: F401\n"
        "assert 'gmatpy' not in sys.modules, sorted(m for m in sys.modules if 'gmat' in m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
