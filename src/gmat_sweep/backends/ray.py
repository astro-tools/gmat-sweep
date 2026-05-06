"""RayPool: distributed execution backend using Ray.

The pool fans :class:`gmat_sweep.spec.RunSpec` work across a Ray cluster.
Ray reuses worker processes for successive tasks, so the
:class:`gmat_sweep.backends.base.Pool` per-run fresh-interpreter contract is
honoured by an explicit subprocess hop inside each task: the top-level
:func:`_ray_run_one_impl` callable (registered as a Ray remote function once
``ray`` is importable) delegates to
:func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`, which spawns
``python -m gmat_sweep._run_subprocess`` for the actual GMAT load. The Ray
worker process itself never imports ``gmatpy``.

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
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

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
    **ray_init_kwargs:
        Extra keyword arguments forwarded verbatim to :func:`ray.init`.
    """

    def __init__(
        self,
        *,
        address: str | None = None,
        num_cpus: int | None = None,
        **ray_init_kwargs: Any,
    ) -> None:
        try:
            import ray as _ray
        except ImportError as exc:
            raise BackendError(
                "RayPool requires the [ray] extra: pip install gmat-sweep[ray]"
            ) from exc

        self._owns_runtime = not _ray.is_initialized()
        if self._owns_runtime:
            init_kwargs: dict[str, Any] = dict(ray_init_kwargs)
            if address is not None:
                init_kwargs["address"] = address
            if num_cpus is not None:
                init_kwargs["num_cpus"] = num_cpus
            _ray.init(**init_kwargs)

        self._ray = _ray
        self._remote_run_one = _ray.remote(_ray_run_one_impl)
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
        object_refs: list[Any] = []
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            future_by_run_id[spec.run_id] = f
            object_refs.append(self._remote_run_one.remote(spec))

        unready: list[Any] = list(object_refs)
        while unready:
            ready, unready = self._ray.wait(unready, num_returns=1)
            outcome: RunOutcome = self._ray.get(ready[0])
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
