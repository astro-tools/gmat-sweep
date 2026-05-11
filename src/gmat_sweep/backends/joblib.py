"""LocalJoblibPool: default execution backend using joblib loky workers.

The pool drives a long-lived :class:`joblib.Parallel` instance configured with
``backend="loky"`` and ``return_as="generator_unordered"``. Loky spawns fresh
Python interpreters as worker processes so the driver's ``gmatpy`` state never
leaks into a worker.

Submission semantics:

- :meth:`submit` parks the spec under a placeholder
  :class:`concurrent.futures.Future` and returns immediately. Nothing is
  dispatched to a worker yet.
- :meth:`as_completed` is the dispatch point. It hands the parked specs to the
  underlying ``Parallel`` context as :func:`joblib.delayed` calls of either
  :func:`gmat_sweep.worker.run_one` (when ``reuse_gmat_context=True``) or
  :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess` (when
  ``reuse_gmat_context=False``), and yields :class:`RunOutcome` values in
  completion order, marking each future done as it goes. A worker-side
  exception that escapes the task callable (loky worker death, pickling
  failure, …) is folded into a synthetic :meth:`RunOutcome.failed` carrying
  the formatted traceback as ``stderr`` so the parent sweep does not abort
  on a single transport failure.
- :meth:`close` cancels any still-pending futures and then exits the
  underlying ``Parallel`` context. Cancellation runs before the loky exit
  so a parked future whose outcome loky is about to compute on the way out
  is not silently dropped.

See :class:`gmat_sweep.backends.base.Pool` for the semantics of
``reuse_gmat_context``.
"""

from __future__ import annotations

import os
import traceback
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone

import joblib

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

__all__ = ["LocalJoblibPool"]


class LocalJoblibPool(Pool):
    """Local subprocess pool backed by ``joblib.Parallel(backend="loky")``.

    Parameters
    ----------
    max_workers:
        Number of loky worker processes. ``-1`` (the default) uses every
        available core. Any other negative value or ``0`` is rejected with
        :class:`gmat_sweep.errors.BackendError`. ``workers=`` is accepted as
        a deprecated alias.
    workers:
        Deprecated alias for ``max_workers=``. Emits a
        :class:`DeprecationWarning`; will be removed in a future release.
    reuse_gmat_context:
        ``True`` (default) dispatches each task as
        :func:`gmat_sweep.worker.run_one`, which imports ``gmat_run`` once
        per loky worker and reuses the import across tasks — fast, but only
        safe when every spec dispatched through this pool loads the same
        script. ``False`` dispatches each task through
        :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`,
        spawning a fresh Python interpreter inside the loky worker so each
        task bootstraps ``gmatpy`` from scratch — slower, but supports
        cross-script sweeps. See
        :class:`gmat_sweep.backends.base.Pool` for the contract.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        *,
        workers: int | None = None,
        reuse_gmat_context: bool = True,
    ) -> None:
        if workers is not None and max_workers is not None:
            raise BackendError(
                "pass either workers= or max_workers=, not both (workers= is the deprecated alias)"
            )
        if workers is not None:
            warnings.warn(
                "LocalJoblibPool(workers=...) is deprecated and will be removed in "
                "a future release; pass max_workers= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved = workers
        else:
            resolved = -1 if max_workers is None else max_workers
        if resolved == 0 or resolved < -1:
            raise BackendError(
                f"max_workers must be -1 (all cores) or a positive integer, got {resolved!r}"
            )
        self._workers = resolved
        self._reuse_gmat_context = reuse_gmat_context
        # Resolve -1 ("all cores") to a concrete count for max_workers — the
        # Pool.imap default uses this for the in-flight cap.
        self._resolved_max_workers: int = (os.cpu_count() or 1) if resolved == -1 else resolved
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False
        self._parallel = joblib.Parallel(
            backend="loky",
            n_jobs=resolved,
            return_as="generator_unordered",
        )
        self._parallel.__enter__()

    @property
    def max_workers(self) -> int:
        return self._resolved_max_workers

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

        task_fn: Callable[[RunSpec], RunOutcome] = (
            run_one if self._reuse_gmat_context else run_spec_in_subprocess
        )
        gen = self._parallel(joblib.delayed(task_fn)(s) for s in specs)
        while future_by_run_id:
            try:
                outcome = next(gen)
            except StopIteration:
                break
            except Exception as exc:
                # Worker-side transport failure (loky worker death, pickling
                # error, …): the generator has raised before we could match
                # an outcome to a run_id. Fold synthetic failures for every
                # remaining run_id, carrying the captured traceback as
                # ``stderr``. The Parallel context is unrecoverable after
                # this point in ``generator_unordered`` mode, so we drain
                # the leftover futures here rather than try to resume.
                # KeyboardInterrupt deliberately is not caught — Ctrl-C
                # should reach the driver.
                stderr = "".join(traceback.format_exception(exc))
                for run_id in list(future_by_run_id):
                    f = future_by_run_id.pop(run_id)
                    folded = _failed_outcome(run_id, stderr)
                    f.set_result(folded)
                    yield folded
                return
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
            self._parallel.__exit__(None, None, None)


def _failed_outcome(run_id: int, stderr: str) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.failed(
        run_id=run_id,
        stderr=stderr,
        started_at=now,
        ended_at=now,
        duration_s=0.0,
    )
