"""RayPool: distributed execution backend using Ray.

The pool fans :class:`gmat_sweep.spec.RunSpec` work across a Ray cluster.
Ray reuses worker processes for successive tasks; the ``reuse_gmat_context``
flag (see :class:`gmat_sweep.backends.base.Pool`) chooses how that reuse
interacts with GMAT bootstrap:

- ``reuse_gmat_context=True`` (default) — each task runs
  :func:`gmat_sweep.worker.run_one` directly inside the Ray worker. The
  first task per worker bootstraps ``gmatpy``; subsequent tasks reuse it.
  The Ray worker process holds gmatpy state across tasks.
- ``reuse_gmat_context=False`` — each task runs the top-level
  :func:`_ray_run_one_impl` callable (registered as a Ray remote function
  at construction time), which delegates to
  :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess` and spawns
  ``python -m gmat_sweep._run_subprocess`` for the actual GMAT load. The
  Ray worker process itself never imports ``gmatpy``.

Lifecycle
---------

``RayPool`` connects to a Ray runtime by calling :func:`ray.init` (with
``address`` and any extra keyword arguments forwarded). :meth:`close` calls
:func:`ray.shutdown` only if the pool's ``__init__`` was what initialised
the runtime — that is, Ray reported uninitialised when the pool was
constructed. If the user had already called :func:`ray.init` before
constructing the pool, the pool does not own the runtime and leaves it
alone on close.

This rule applies regardless of ``address``: ``ray.shutdown`` only severs
the local handle, so even a remote-cluster connection is safe to drop on
close — as long as the pool was the caller that opened it.

Object-store note
-----------------

Ray serialises task arguments through cloudpickle into its plasma object
store. :class:`RunSpec` is :func:`dataclasses.dataclass` with
JSON-encodable fields, well within Ray's serialisation surface; values
inside ``overrides`` and ``run_options`` must already be JSON-encodable
per the v0.1 spec contract.

Transport-failure semantics
---------------------------

A worker-side exception that escapes the remote task — ``RayTaskError``
from a worker crash, a serialisation fault, … — is caught at the drain
site (``ray.get`` in :meth:`as_completed`) and folded into a synthetic
:meth:`gmat_sweep.spec.RunOutcome.failed` so the parent sweep does not
abort on a single transport failure.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

if TYPE_CHECKING:
    import ray as _ray_typing  # noqa: F401

__all__ = ["RayPool"]


def _ray_run_one_impl(spec: RunSpec) -> RunOutcome:
    """Body of the Ray remote task — delegates to the subprocess hop.

    Defined at module scope so Ray's serialiser can pickle it. Crucially,
    this function does **not** import ``gmatpy``: the subprocess hop is what
    loads GMAT, in a fresh interpreter.
    """
    return run_spec_in_subprocess(spec)


class RayPool(Pool):
    """Distributed pool backed by Ray.

    Parameters
    ----------
    address:
        Forwarded to :func:`ray.init` to connect to an existing cluster
        (``"auto"`` for a local cluster, ``"ray://host:port"`` for a remote
        Ray Client server, or a raw GCS address). ``None`` (default) starts
        a local Ray runtime.
    num_cpus:
        Forwarded to :func:`ray.init` for the local-runtime case. Ignored
        when connecting to an existing cluster via ``address``.
    reuse_gmat_context:
        ``True`` (default) lets Ray workers reuse a single ``gmatpy``
        import across tasks — fast, but only safe when every spec
        dispatched through this pool loads the same script. ``False``
        binds the Ray remote to :func:`_ray_run_one_impl`, which spawns a
        fresh Python interpreter per task and bootstraps ``gmatpy`` from
        scratch — slower, but supports cross-script sweeps. See
        :class:`gmat_sweep.backends.base.Pool` for the contract.
    **ray_init_kwargs:
        Extra keyword arguments forwarded verbatim to :func:`ray.init`.
    """

    def __init__(
        self,
        *,
        address: str | None = None,
        num_cpus: int | None = None,
        reuse_gmat_context: bool = True,
        **ray_init_kwargs: Any,
    ) -> None:
        try:
            import ray as _ray
        except ImportError as exc:
            raise BackendError(
                "RayPool requires the [ray] extra: pip install gmat-sweep[ray]"
            ) from exc

        self._reuse_gmat_context = reuse_gmat_context
        self._owns_runtime = not _ray.is_initialized()
        if self._owns_runtime:
            init_kwargs: dict[str, Any] = dict(ray_init_kwargs)
            if address is not None:
                init_kwargs["address"] = address
            if num_cpus is not None:
                init_kwargs["num_cpus"] = num_cpus
            _ray.init(**init_kwargs)

        self._ray = _ray
        impl = run_one if reuse_gmat_context else _ray_run_one_impl
        self._remote_run_one = _ray.remote(impl)
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("RayPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("RayPool is closed; cannot drain futures")

        wanted = list(futures)
        future_by_run_id: dict[int, Future[RunOutcome]] = {}
        run_id_by_ref: dict[Any, int] = {}
        object_refs: list[Any] = []
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            future_by_run_id[spec.run_id] = f
            ref = self._remote_run_one.remote(spec)
            object_refs.append(ref)
            run_id_by_ref[ref] = spec.run_id

        unready: list[Any] = list(object_refs)
        while unready:
            # ``fetch_local=True`` primes the local object store with the
            # ready refs as part of the wait, so the per-ref ``ray.get``
            # below is an in-store read rather than a second network
            # round-trip. The old shape — ``wait(num_returns=1)`` then a
            # separate ``ray.get(ready[0])`` — paid two trips per outcome.
            ready, unready = self._ray.wait(unready, num_returns=1, fetch_local=True)
            for ref in ready:
                run_id = run_id_by_ref[ref]
                try:
                    outcome: RunOutcome = self._ray.get(ref)
                except Exception as exc:
                    # ``RayTaskError`` (a remote-side raise that escaped
                    # ``run_one``) or a Ray transport error — fold into a
                    # synthetic failed outcome so the sweep does not abort.
                    outcome = _failed_outcome(run_id, "".join(traceback.format_exception(exc)))
                f = future_by_run_id.pop(outcome.run_id)
                f.set_result(outcome)
                yield outcome

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for f in self._pending:
                f.cancel()
            self._pending.clear()
        finally:
            if self._owns_runtime:
                self._ray.shutdown()


def _failed_outcome(run_id: int, stderr: str) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.failed(run_id=run_id, stderr=stderr, started_at=now, ended_at=now)
