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

Because ``max_tasks_per_child=1`` already gives every task a fresh
interpreter, this backend dispatches each task as
:func:`gmat_sweep.worker.run_one` directly regardless of
``reuse_gmat_context``. The nested ``run_spec_in_subprocess`` hop that
other backends use for the ``reuse_gmat_context=False`` mode would just
double-pay the subprocess cost here without changing the contract.

Transport-failure semantics
---------------------------

A worker-side exception that escapes :func:`gmat_sweep.worker.run_one`
(``BrokenProcessPool`` on worker death, an unpicklable return value, …)
is caught at the drain site and folded into a synthetic
:meth:`gmat_sweep.spec.RunOutcome.failed` carrying the formatted traceback
as ``stderr`` so the parent sweep does not abort.

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

import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import as_completed as _futures_as_completed
from datetime import datetime, timezone

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
        Accepted for :class:`Pool` API parity. Both values dispatch each
        task as :func:`gmat_sweep.worker.run_one` directly inside the
        worker process — the ``max_tasks_per_child=1`` baked into the
        executor already provides the per-task fresh-interpreter
        guarantee the ``reuse_gmat_context=False`` contract requires.
        Nested ``run_spec_in_subprocess`` hops on this backend would
        double-pay the subprocess cost without changing the contract.
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
        executor_future_to_run_id: dict[Future[RunOutcome], int] = {}
        future_by_executor_future: dict[Future[RunOutcome], Future[RunOutcome]] = {}
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            executor_future = self._executor.submit(run_one, spec)
            future_by_executor_future[executor_future] = f
            executor_future_to_run_id[executor_future] = spec.run_id

        for executor_future in _futures_as_completed(future_by_executor_future):
            user_future = future_by_executor_future[executor_future]
            run_id = executor_future_to_run_id[executor_future]
            try:
                outcome: RunOutcome = executor_future.result()
            except Exception as exc:
                # Worker-side transport failure (BrokenProcessPool from
                # worker death, unpicklable result, …). ``run_one`` itself
                # already catches its own exceptions into RunOutcome.failed;
                # anything that escapes here is a transport-level fault.
                outcome = _failed_outcome(run_id, "".join(traceback.format_exception(exc)))
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


def _failed_outcome(run_id: int, stderr: str) -> RunOutcome:
    now = datetime.now(timezone.utc)
    return RunOutcome.failed(run_id=run_id, stderr=stderr, started_at=now, ended_at=now)
