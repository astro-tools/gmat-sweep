"""MPIPool: distributed execution backend using ``mpi4py.futures``.

The pool fans :class:`gmat_sweep.spec.RunSpec` work across MPI worker ranks
through :class:`mpi4py.futures.MPIPoolExecutor`. The executor reuses worker
ranks for successive tasks; the ``reuse_gmat_context`` flag (see
:class:`gmat_sweep.backends.base.Pool`) chooses how that reuse interacts
with GMAT bootstrap:

- ``reuse_gmat_context=True`` (default) — each task runs
  :func:`gmat_sweep.worker.run_one` directly inside the worker rank. The
  first task per rank bootstraps ``gmatpy``; subsequent tasks reuse it.
  The rank's Python process holds gmatpy state across tasks.
- ``reuse_gmat_context=False`` — each task runs the top-level
  :func:`_mpi_run_one_impl` callable, which delegates to
  :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess` and spawns
  ``python -m gmat_sweep._run_subprocess`` for the actual GMAT load. The
  rank's Python process itself never imports ``gmatpy``.

Launch modes
------------

``MPIPoolExecutor`` supports two launch modes natively; ``MPIPool`` does
not second-guess upstream's mode detection — both work without any
configuration on this side:

- **Dynamic spawn.** Plain ``python script.py`` (or
  ``gmat-sweep run --backend mpi …``). The executor calls
  ``MPI_Comm_spawn`` to launch ``max_workers`` worker ranks on demand.
  Best for laptop / CI / dev runs where the user has Open MPI installed
  but no allocation in hand.
- **Pre-allocated ranks.**
  ``mpirun -n K python -m mpi4py.futures script.py``
  (or ``mpirun -n K python -m mpi4py.futures -m gmat_sweep …``). Ranks
  1..K-1 enter ``mpi4py.futures``'s worker loop inside the launcher shim,
  rank 0 runs user code; ``MPIPoolExecutor()`` uses the pre-allocated
  ranks. Best for SLURM allocations and HPC clusters.

In both modes the user-side code is identical (``with MPIPool(...)``);
``mpi4py.futures`` decides which path applies based on how the parent
process was launched.

Lifecycle
---------

``MPIPool`` always owns the underlying ``MPIPoolExecutor``. :meth:`close`
calls :meth:`MPIPoolExecutor.shutdown` with ``wait=True`` so worker-rank
shutdown is bounded by the pool's context-manager exit.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from concurrent.futures import as_completed as _futures_as_completed
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

if TYPE_CHECKING:
    import mpi4py.futures as _mpi_futures_typing  # noqa: F401

__all__ = ["MPIPool"]


def _mpi_run_one_impl(spec: RunSpec) -> RunOutcome:
    """Body of the MPI task — delegates to the subprocess hop.

    Defined at module scope so ``mpi4py.futures``'s pickle-based dispatch can
    serialise it. Crucially, this function does **not** import ``gmatpy``:
    the subprocess hop is what loads GMAT, in a fresh interpreter.
    """
    return run_spec_in_subprocess(spec)


class MPIPool(Pool):
    """Distributed pool backed by ``mpi4py.futures``.

    Parameters
    ----------
    max_workers:
        Number of MPI worker ranks. Forwarded verbatim to
        :class:`mpi4py.futures.MPIPoolExecutor`. ``None`` (default) lets
        the executor pick — under
        ``mpirun -n K python -m mpi4py.futures …`` that means K-1
        pre-allocated workers; under plain ``python …`` the executor
        falls back to ``MPI_Comm_spawn`` with an implementation-defined
        default count, so an explicit ``max_workers`` is recommended for
        the dynamic-spawn path.
    reuse_gmat_context:
        ``True`` (default) lets MPI worker ranks reuse a single ``gmatpy``
        import across tasks — fast, but only safe when every spec
        dispatched through this pool loads the same script. ``False``
        binds the executor task to :func:`_mpi_run_one_impl`, which spawns
        a fresh Python interpreter per task and bootstraps ``gmatpy`` from
        scratch — slower, but supports cross-script sweeps. See
        :class:`gmat_sweep.backends.base.Pool` for the contract.
    **mpi_executor_kwargs:
        Extra keyword arguments forwarded verbatim to
        :class:`mpi4py.futures.MPIPoolExecutor`.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        reuse_gmat_context: bool = True,
        **mpi_executor_kwargs: Any,
    ) -> None:
        try:
            import mpi4py.futures as _mpi_futures
        except ImportError as exc:
            raise BackendError(
                "MPIPool requires the [mpi] extra: pip install gmat-sweep[mpi]"
            ) from exc

        self._reuse_gmat_context = reuse_gmat_context
        self._executor = _mpi_futures.MPIPoolExecutor(
            max_workers=max_workers, **mpi_executor_kwargs
        )
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("MPIPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("MPIPool is closed; cannot drain futures")

        wanted = list(futures)
        future_by_mpi_future: dict[Future[RunOutcome], Future[RunOutcome]] = {}
        task_fn = run_one if self._reuse_gmat_context else _mpi_run_one_impl
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            mpi_future = self._executor.submit(task_fn, spec)
            future_by_mpi_future[mpi_future] = f

        for mpi_future in _futures_as_completed(future_by_mpi_future):
            outcome: RunOutcome = mpi_future.result()
            user_future = future_by_mpi_future[mpi_future]
            user_future.set_result(outcome)
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
            self._executor.shutdown(wait=True)
