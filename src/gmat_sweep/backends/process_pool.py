"""ProcessPoolExecutorPool: stdlib local execution backend (Python 3.11+).

Wraps :class:`concurrent.futures.ProcessPoolExecutor` with
``max_tasks_per_child=1`` so every task runs in a fresh Python interpreter
— the per-task subprocess-isolation contract from
:class:`gmat_sweep.backends.base.Pool` holds by construction. Stdlib only:
no ``joblib`` / ``loky`` runtime dependency.

Ships as a second local backend alongside
:class:`gmat_sweep.backends.joblib.LocalJoblibPool`. ``LocalJoblibPool``
remains the default; ``ProcessPoolExecutorPool`` is opt-in via explicit
construction. Pick it when avoiding the ``joblib`` runtime dep matters and
the per-task gmatpy bootstrap cost is acceptable.

Python floor
------------

``ProcessPoolExecutor.max_tasks_per_child`` was added in Python 3.11.
Importing this module on Python < 3.11 raises :class:`RuntimeError` with a
message pointing at :class:`LocalJoblibPool` for the 3.10 path.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# Hide the gate from mypy's `sys.version_info` static narrowing — under
# `python_version = "3.10"` (the project's mypy floor) mypy would otherwise
# treat the rest of the module as unreachable and the public surface would
# vanish from type-check view. At runtime, ``not TYPE_CHECKING`` is True so
# the gate fires whenever the interpreter actually is < 3.11.
if not TYPE_CHECKING and sys.version_info < (3, 11):
    raise RuntimeError(
        "ProcessPoolExecutorPool requires Python 3.11+ "
        "(needs concurrent.futures.ProcessPoolExecutor's max_tasks_per_child); "
        "use gmat_sweep.LocalJoblibPool on Python 3.10."
    )

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import as_completed as _futures_as_completed

from gmat_sweep.backends._subprocess import run_spec_in_subprocess
from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

__all__ = ["ProcessPoolExecutorPool"]


class ProcessPoolExecutorPool(Pool):
    """Local subprocess pool backed by :class:`concurrent.futures.ProcessPoolExecutor`.

    Always constructs the executor with ``max_tasks_per_child=1`` so every
    task runs in a fresh Python interpreter. The subprocess-isolation
    contract from :class:`gmat_sweep.backends.base.Pool` therefore holds
    by construction, and the ``reuse_gmat_context`` flag chooses only the
    in-worker dispatch path.

    Parameters
    ----------
    max_workers:
        Number of worker processes. Forwarded verbatim to
        :class:`concurrent.futures.ProcessPoolExecutor`. ``None`` (the
        default) lets the executor pick — :func:`os.process_cpu_count`
        on Python 3.13+, :func:`os.cpu_count` on 3.11-3.12.
    reuse_gmat_context:
        ``True`` (default) dispatches each task as
        :func:`gmat_sweep.worker.run_one` directly inside the fresh worker
        process — one gmatpy bootstrap per task (the worker is already
        fresh thanks to ``max_tasks_per_child=1``). ``False`` dispatches
        through :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`,
        spawning a *second* nested Python interpreter for the gmatpy load
        — wasteful but contract-compliant. Same flag, same contract as
        :class:`gmat_sweep.backends.base.Pool` documents.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        reuse_gmat_context: bool = True,
    ) -> None:
        self._reuse_gmat_context = reuse_gmat_context
        # `max_tasks_per_child` requires Python 3.11+; the module-level gate
        # above fails fast on 3.10. The type: ignore is here because the
        # project pins mypy to python_version=3.10 (the supported floor);
        # under that target, typeshed's ProcessPoolExecutor signature is the
        # 3.10 one, which lacks `max_tasks_per_child`.
        self._executor = ProcessPoolExecutor(  # type: ignore[call-overload]
            max_workers=max_workers,
            max_tasks_per_child=1,
        )
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("ProcessPoolExecutorPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("ProcessPoolExecutorPool is closed; cannot drain futures")

        wanted = list(futures)
        task_fn: Callable[[RunSpec], RunOutcome] = (
            run_one if self._reuse_gmat_context else run_spec_in_subprocess
        )
        future_by_executor_future: dict[Future[RunOutcome], Future[RunOutcome]] = {}
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            executor_future = self._executor.submit(task_fn, spec)
            future_by_executor_future[executor_future] = f

        for executor_future in _futures_as_completed(future_by_executor_future):
            outcome: RunOutcome = executor_future.result()
            user_future = future_by_executor_future[executor_future]
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
