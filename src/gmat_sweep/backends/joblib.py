"""LocalJoblibPool: default execution backend using joblib loky workers.

The pool drives a long-lived :class:`joblib.Parallel` instance configured with
``backend="loky"`` and ``return_as="generator_unordered"``. Loky spawns fresh
Python interpreters as worker processes so the driver's ``gmatpy`` state never
leaks into a worker, satisfying the
:class:`gmat_sweep.backends.base.Pool` subprocess-isolation contract.

Submission semantics match the issue:

- :meth:`submit` parks the spec under a placeholder
  :class:`concurrent.futures.Future` and returns immediately. Nothing is
  dispatched to a worker yet.
- :meth:`as_completed` is the dispatch point. It hands the parked specs to the
  underlying ``Parallel`` context as :func:`joblib.delayed` calls of
  :func:`gmat_sweep.worker.run_one` and yields :class:`RunOutcome` values in
  completion order, marking each future done as it goes.
- :meth:`close` exits the underlying ``Parallel`` context, terminating loky
  workers, and cancels any still-pending futures.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future

import joblib

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

__all__ = ["LocalJoblibPool"]


class LocalJoblibPool(Pool):
    """Local subprocess pool backed by ``joblib.Parallel(backend="loky")``.

    Parameters
    ----------
    workers:
        Number of loky worker processes. ``-1`` (the default) uses every
        available core. Any other negative value or ``0`` is rejected with
        :class:`gmat_sweep.errors.BackendError`.
    """

    def __init__(self, workers: int = -1) -> None:
        if workers == 0 or workers < -1:
            raise BackendError(
                f"workers must be -1 (all cores) or a positive integer, got {workers!r}"
            )
        self._workers = workers
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False
        self._parallel = joblib.Parallel(
            backend="loky",
            n_jobs=workers,
            return_as="generator_unordered",
        )
        self._parallel.__enter__()

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("LocalJoblibPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("LocalJoblibPool is closed; cannot drain futures")

        wanted = list(futures)
        specs: list[RunSpec] = []
        future_by_run_id: dict[int, Future[RunOutcome]] = {}
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            specs.append(spec)
            future_by_run_id[spec.run_id] = f

        for outcome in self._parallel(joblib.delayed(run_one)(s) for s in specs):
            f = future_by_run_id.pop(outcome.run_id)
            f.set_result(outcome)
            yield outcome

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._parallel.__exit__(None, None, None)
        finally:
            for f in self._pending:
                f.cancel()
            self._pending.clear()
