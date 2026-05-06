"""DaskPool: distributed execution backend using ``dask.distributed``.

The pool fans :class:`gmat_sweep.spec.RunSpec` work across a Dask cluster.
Dask reuses worker processes for successive tasks, so the
:class:`gmat_sweep.backends.base.Pool` per-run fresh-interpreter contract is
honoured by an explicit subprocess hop inside each task: the top-level
:func:`_dask_run_one` callable delegates to
:func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`, which spawns
``python -m gmat_sweep._run_subprocess`` for the actual GMAT load. The Dask
worker process itself never imports ``gmatpy``.

Lifecycle
---------

``DaskPool`` accepts an existing ``distributed.Client`` or, when given none,
spins up a local ``distributed.LocalCluster`` and a ``Client`` connected to
it. :meth:`close` shuts down only what the pool owns:

- ``client=None``, ``n_workers=None`` — own the LocalCluster and the Client.
- ``client=None``, ``n_workers=K`` — own the LocalCluster and the Client.
- ``client=<Client>`` — own neither; ``close`` does not touch the client.

Submission semantics match :class:`gmat_sweep.backends.joblib.LocalJoblibPool`:
:meth:`submit` parks the spec under a placeholder
:class:`concurrent.futures.Future` and returns immediately; dispatch happens
inside :meth:`as_completed`, which submits one Dask task per parked spec and
drains via :func:`distributed.as_completed`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

if TYPE_CHECKING:
    import distributed

__all__ = ["DaskPool"]


def _dask_run_one(spec: RunSpec) -> RunOutcome:
    """Top-level Dask task body — delegates to the subprocess hop.

    Defined at module scope so Dask's serialiser can pickle it. Crucially,
    this function does **not** import ``gmatpy``: the subprocess hop is what
    loads GMAT, in a fresh interpreter.
    """
    return run_spec_in_subprocess(spec)


class DaskPool(Pool):
    """Distributed pool backed by ``dask.distributed``.

    Parameters
    ----------
    client:
        An existing :class:`distributed.Client` to dispatch through. When
        supplied, the pool does not create or own a cluster, and
        :meth:`close` does not shut the client down.
    n_workers:
        Number of workers in the auto-spawned :class:`distributed.LocalCluster`.
        Ignored when ``client`` is supplied. ``None`` (default) uses
        :func:`os.cpu_count`.
    threads_per_worker:
        Threads per worker for the auto-spawned :class:`distributed.LocalCluster`.
        Ignored when ``client`` is supplied. Defaults to ``1`` so each worker
        is a single-threaded subprocess shell.
    """

    def __init__(
        self,
        *,
        client: distributed.Client | None = None,
        n_workers: int | None = None,
        threads_per_worker: int = 1,
    ) -> None:
        try:
            import distributed as _distributed
        except ImportError as exc:
            raise BackendError(
                "DaskPool requires the [dask] extra: pip install gmat-sweep[dask]"
            ) from exc

        self._owns_client = False
        self._owns_cluster = False
        self._cluster: Any = None

        if client is None:
            workers = n_workers if n_workers is not None else os.cpu_count() or 1
            self._cluster = _distributed.LocalCluster(
                n_workers=workers,
                threads_per_worker=threads_per_worker,
            )
            self._client = _distributed.Client(self._cluster)
            self._owns_client = True
            self._owns_cluster = True
        else:
            self._client = client

        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False
        self._distributed = _distributed

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("DaskPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("DaskPool is closed; cannot drain futures")

        wanted = list(futures)
        future_by_run_id: dict[int, Future[RunOutcome]] = {}
        dask_futures: list[Any] = []
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            future_by_run_id[spec.run_id] = f
            dask_futures.append(self._client.submit(_dask_run_one, spec, pure=False))

        for dask_future in self._distributed.as_completed(dask_futures):
            outcome: RunOutcome = dask_future.result()
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
            if self._owns_client:
                self._client.close()
            if self._owns_cluster and self._cluster is not None:
                self._cluster.close()
