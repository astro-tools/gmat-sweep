"""Tests for gmat_sweep.backends.ray.RayPool — submit/as_completed semantics + ownership."""

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

ray = pytest.importorskip("ray")


from gmat_sweep.backends.ray import RayPool, _ray_run_one_impl  # noqa: E402
from gmat_sweep.worker import run_one  # noqa: E402


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
    return RunOutcome.ok(run_id=run_id, output_paths={}, started_at=now, ended_at=now)


class _ObjectRef:
    """Stand-in for a Ray ObjectRef carrying a precomputed outcome.

    ``exc`` (optional) lets the stub surface a ``ray.get``-time exception
    — the analogue of a real ``RayTaskError`` from a remote-side raise.
    """

    def __init__(self, value: RunOutcome | None, *, exc: BaseException | None = None) -> None:
        self.value = value
        self.exc = exc


class _Remote:
    """Stand-in for a `ray.remote`-decorated function.

    A remote callable that raises is captured as ``ObjectRef(exc=...)`` so
    the surfacing happens on ``ray.get``, matching the real Ray contract
    where remote-side exceptions surface only on retrieval.
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self, spec: RunSpec) -> _ObjectRef:
        try:
            return _ObjectRef(self._fn(spec))
        except Exception as exc:
            return _ObjectRef(None, exc=exc)


def _patch_ray_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_initialized: bool,
    init_log: list[dict[str, Any]] | None = None,
    shutdown_log: list[None] | None = None,
) -> None:
    """Replace the ray surface used by RayPool with controllable stubs."""
    state = {"initialized": is_initialized}
    init_log = init_log if init_log is not None else []
    shutdown_log = shutdown_log if shutdown_log is not None else []

    def _is_initialized() -> bool:
        return state["initialized"]

    def _init(**kwargs: Any) -> None:
        init_log.append(kwargs)
        state["initialized"] = True

    def _shutdown() -> None:
        shutdown_log.append(None)
        state["initialized"] = False

    def _remote(fn: Any) -> _Remote:
        return _Remote(fn)

    def _wait(
        refs: list[_ObjectRef],
        *,
        num_returns: int = 1,
        fetch_local: bool = True,
    ) -> tuple[list[_ObjectRef], list[_ObjectRef]]:
        ready = refs[:num_returns]
        unready = refs[num_returns:]
        return ready, unready

    def _get(ref: _ObjectRef) -> RunOutcome:
        if ref.exc is not None:
            raise ref.exc
        assert ref.value is not None
        return ref.value

    monkeypatch.setattr(ray, "is_initialized", _is_initialized, raising=True)
    monkeypatch.setattr(ray, "init", _init, raising=True)
    monkeypatch.setattr(ray, "shutdown", _shutdown, raising=True)
    monkeypatch.setattr(ray, "remote", _remote, raising=True)
    monkeypatch.setattr(ray, "wait", _wait, raising=True)
    monkeypatch.setattr(ray, "get", _get, raising=True)


def test_raypool_is_pool_subclass() -> None:
    assert issubclass(RayPool, Pool)
    assert RayPool.subprocess_isolated is True


def test_subclass_setting_subprocess_isolated_false_rejected() -> None:
    """The Pool ABC's contract still applies to RayPool subclasses."""
    with pytest.raises(BackendError):

        class _Bad(RayPool):  # pragma: no cover - body never runs
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
    """If `ray` cannot be imported, RayPool() raises BackendError chained from ImportError."""
    monkeypatch.setitem(sys.modules, "ray", None)
    with pytest.raises(BackendError) as ei:
        RayPool()
    assert "[ray]" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_owns_runtime_when_ray_was_uninitialised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_log: list[dict[str, Any]] = []
    shutdown_log: list[None] = []
    _patch_ray_with_stubs(
        monkeypatch, is_initialized=False, init_log=init_log, shutdown_log=shutdown_log
    )
    pool = RayPool(num_cpus=2)
    assert pool._owns_runtime is True
    assert init_log == [{"num_cpus": 2}]
    pool.close()
    assert shutdown_log == [None]


def test_owns_runtime_when_address_supplied_and_ray_uninitialised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per #58: pool owns the local handle iff Ray was uninitialised at construction."""
    init_log: list[dict[str, Any]] = []
    shutdown_log: list[None] = []
    _patch_ray_with_stubs(
        monkeypatch, is_initialized=False, init_log=init_log, shutdown_log=shutdown_log
    )
    pool = RayPool(address="ray://example:10001", num_cpus=4)
    assert pool._owns_runtime is True
    assert init_log == [{"address": "ray://example:10001", "num_cpus": 4}]
    pool.close()
    assert shutdown_log == [None]


def test_does_not_own_runtime_when_already_initialised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_log: list[dict[str, Any]] = []
    shutdown_log: list[None] = []
    _patch_ray_with_stubs(
        monkeypatch, is_initialized=True, init_log=init_log, shutdown_log=shutdown_log
    )
    pool = RayPool()
    assert pool._owns_runtime is False
    assert init_log == []
    pool.close()
    assert shutdown_log == []


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_log: list[None] = []
    _patch_ray_with_stubs(monkeypatch, is_initialized=False, shutdown_log=shutdown_log)
    pool = RayPool()
    pool.close()
    pool.close()
    assert shutdown_log == [None]


def test_submit_returns_pending_future(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()
    pool.close()


def test_submit_after_close_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_close_cancels_pending_futures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


def test_as_completed_dispatches_via_remote_and_drains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """as_completed builds one ObjectRef per spec and drains via ray.wait/get."""
    submitted: list[RunSpec] = []

    def _capturing_impl(spec: RunSpec) -> RunOutcome:
        submitted.append(spec)
        return _ok_outcome(spec.run_id)

    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    monkeypatch.setattr(pool, "_remote_run_one", _Remote(_capturing_impl))

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


def test_as_completed_uses_fetch_local_in_ray_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``ray.wait`` is called with ``fetch_local=True`` to prime the local store.

    The pre-fix shape paid two network round-trips per outcome:
    ``ray.wait(num_returns=1)`` then a separate ``ray.get(ready[0])``. The
    ``fetch_local=True`` knob collapses that into one round-trip — pinning
    the kwarg here guards against a silent regression.
    """
    wait_kwargs: list[dict[str, Any]] = []

    _patch_ray_with_stubs(monkeypatch, is_initialized=True)

    def _spying_wait(
        refs: list[_ObjectRef],
        *,
        num_returns: int = 1,
        fetch_local: bool = False,
    ) -> tuple[list[_ObjectRef], list[_ObjectRef]]:
        wait_kwargs.append({"num_returns": num_returns, "fetch_local": fetch_local})
        return refs[:num_returns], refs[num_returns:]

    monkeypatch.setattr(ray, "wait", _spying_wait, raising=True)

    pool = RayPool()
    monkeypatch.setattr(pool, "_remote_run_one", _Remote(lambda spec: _ok_outcome(spec.run_id)))
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    list(pool.as_completed([f]))
    pool.close()

    assert wait_kwargs, "ray.wait was not called"
    assert all(call["fetch_local"] is True for call in wait_kwargs)


def test_worker_side_exception_folds_into_failed_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A remote task that raises is folded into ``RunOutcome.failed``.

    The drain loop wraps ``ray.get`` in ``try/except`` so a ``RayTaskError``
    from a remote-side raise (or a Ray transport error) folds into a
    synthetic failed outcome instead of aborting the sweep.
    """

    def _boom(_spec: RunSpec) -> RunOutcome:
        raise RuntimeError("remote exploded")

    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    monkeypatch.setattr(pool, "_remote_run_one", _Remote(_boom))

    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    outcomes = list(pool.as_completed([f]))
    pool.close()

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 0
    assert outcomes[0].status == "failed"
    assert outcomes[0].stderr is not None
    assert "remote exploded" in outcomes[0].stderr
    assert f.done()
    assert f.result() is outcomes[0]


def test_one_failed_run_does_not_abort_other_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mix one remote-failing task with two successful ones — sweep yields all three."""

    def _maybe_boom(spec: RunSpec) -> RunOutcome:
        if spec.run_id == 1:
            raise RuntimeError("remote 1 down")
        return _ok_outcome(spec.run_id)

    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    monkeypatch.setattr(pool, "_remote_run_one", _Remote(_maybe_boom))

    futures = [
        pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)
    ]
    outcomes = sorted(pool.as_completed(futures), key=lambda o: o.run_id)
    pool.close()

    assert [o.run_id for o in outcomes] == [0, 1, 2]
    assert [o.status for o in outcomes] == ["ok", "failed", "ok"]
    assert outcomes[1].stderr is not None
    assert "remote 1 down" in outcomes[1].stderr


def test_as_completed_rejects_unknown_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_ray_with_stubs(monkeypatch, is_initialized=True)
    pool = RayPool()
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    list(pool.as_completed([f]))
    with pytest.raises(BackendError):
        list(pool.as_completed([f]))
    pool.close()


def test_default_binds_run_one_as_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ``reuse_gmat_context=True`` binds ``run_one`` as the Ray remote impl."""
    _patch_ray_with_stubs(monkeypatch, is_initialized=False)
    pool = RayPool()
    # `_Remote` (the ray.remote stub) captures the impl in `_fn`; the stub's
    # behaviour mirrors `ray.remote(fn).remote(...) -> ObjectRef(fn(...))`,
    # so checking `_fn` is the right way to verify the binding.
    assert pool._remote_run_one._fn is run_one
    pool.close()


def test_reuse_gmat_context_false_binds_subprocess_impl_as_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reuse_gmat_context=False`` binds ``_ray_run_one_impl`` (subprocess hop) as the remote."""
    _patch_ray_with_stubs(monkeypatch, is_initialized=False)
    pool = RayPool(reuse_gmat_context=False)
    assert pool._remote_run_one._fn is _ray_run_one_impl
    pool.close()


def test_ray_run_one_impl_delegates_to_subprocess_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_ray_run_one_impl is a thin wrapper around run_spec_in_subprocess."""
    captured: dict[str, RunSpec] = {}

    def _fake(spec: RunSpec) -> RunOutcome:
        captured["spec"] = spec
        return _ok_outcome(spec.run_id)

    monkeypatch.setattr("gmat_sweep.backends.ray.run_spec_in_subprocess", _fake)
    spec = _make_spec(output_dir=tmp_path / "run_0", run_id=99)
    outcome = _ray_run_one_impl(spec)
    assert captured["spec"] is spec
    assert outcome.run_id == 99


def test_importing_ray_module_does_not_import_gmatpy() -> None:
    """Loading gmat_sweep.backends.ray in a fresh interpreter must not import gmatpy.

    The Ray worker process inherits whatever the module top-level imports
    pulled in. If anything in the import chain triggered gmatpy, the
    subprocess-isolation contract would be silently violated.
    """
    code = (
        "import sys\n"
        "import gmat_sweep.backends.ray  # noqa: F401\n"
        "assert 'gmatpy' not in sys.modules, sorted(m for m in sys.modules if 'gmat' in m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_raypool_workers_can_import_ray_under_uv_run() -> None:
    """Real-Ray regression for #76: workers must import `ray` under `uv run`.

    With Ray's auto-uv runtime_env hook on (its default), ``ray.init()`` from a
    driver launched by ``uv run`` rewrites worker startup to relaunch each
    worker with ``uv run python ...`` from a packaged copy of the working dir.
    uv then rebuilds the worker venv from the project's base dependencies
    only — without the ``[ray]`` extra — and the worker's ``import ray``
    raises ``ModuleNotFoundError``. Constructing :class:`RayPool` here is what
    triggers gmat-sweep's env-var override of the hook (see
    :mod:`gmat_sweep.backends`); the assertion verifies that a freshly spawned
    Ray worker can complete an ``import ray`` + ``ray.__version__`` call.

    The marker is ``integration`` because the test launches a real Ray runtime,
    not because GMAT is involved (it is not).

    Skipped on macOS: Ray's GCS-registration step has a hardcoded 30 s timeout
    in ``ray._private.node`` (no Python-level override), and the macos-latest
    GitHub-Actions runner consistently exceeds it during the initial bootstrap.
    The bug we're guarding (#76) is a uv-run interaction that is identical on
    Linux and Windows — covering it on those two platforms is sufficient.
    """
    if sys.platform == "darwin":
        pytest.skip(
            "Ray GCS bootstrap exceeds 30 s on macos-latest runners; the "
            "regression we're guarding (#76) is platform-independent and is "
            "covered on the Linux and Windows cells."
        )
    pool = RayPool(num_cpus=1, include_dashboard=False)
    try:

        def _worker_ray_version() -> str:
            import ray as _r

            return str(_r.__version__)

        remote_check = pool._ray.remote(_worker_ray_version)
        version = pool._ray.get(remote_check.remote())
        assert isinstance(version, str) and version
    finally:
        pool.close()
