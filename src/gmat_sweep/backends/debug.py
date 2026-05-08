"""DebugPool: in-process, single-run backend for breakpoint-driven debugging.

The pool runs one :class:`gmat_sweep.spec.RunSpec` on the driver process —
no subprocess, no parallelism — so a ``breakpoint()`` placed in user code,
override-application logic, or :mod:`gmat_run` itself drops directly into
the driver's debugger and IDE step-through Just Works. This explicitly
violates the subprocess-isolation contract every other pool implements;
that violation is the feature.

Two consequences follow from the in-process design:

- **Single-run only.** GMAT relies on process-global singletons that
  cannot be reused across loads of different scripts, and re-isolating
  in-process between specs is not implemented. :meth:`Sweep.run` raises
  :class:`gmat_sweep.errors.BackendError` if more than one spec is
  submitted to a :class:`DebugPool`. Use :class:`LocalJoblibPool` /
  :class:`ProcessPoolExecutorPool` for any sweep with N > 1.
- **Two opt-ins.** Constructing the pool requires
  ``allow_unisolated_pool=True``; passing it through to a sweep also
  requires ``Sweep(..., allow_unisolated_pool=True)``. Both raise
  :class:`gmat_sweep.errors.BackendError` if the flag is missing, so the
  isolation violation never happens silently.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from typing import ClassVar, Literal

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.worker import run_one

__all__ = ["DebugPool"]


class DebugPool(Pool):
    """In-process, single-run pool for ``breakpoint()``-driven debugging.

    Dispatches each spec by calling :func:`gmat_sweep.worker.run_one`
    synchronously on the driver process. No worker pool is constructed, no
    subprocess is spawned, and nothing is parallelised — the point is for
    the driver's debugger to be the worker's debugger. The trade-off is
    that the GMAT singletons in the driver process get dirtied by the
    run; reusing the same Python interpreter for a second run is not
    supported, and :class:`gmat_sweep.sweep.Sweep` enforces the limit.

    Parameters
    ----------
    allow_unisolated_pool:
        Required opt-in. Defaults to :data:`False`, which raises
        :class:`gmat_sweep.errors.BackendError` from ``__init__`` so the
        violation cannot happen accidentally. Pass :data:`True` to
        construct the pool — and remember to pass the matching flag to
        :class:`gmat_sweep.sweep.Sweep` as well.

    Examples
    --------
    >>> from gmat_sweep import Sweep
    >>> from gmat_sweep.backends.debug import DebugPool
    >>> pool = DebugPool(allow_unisolated_pool=True)  # doctest: +SKIP
    >>> sweep = Sweep(  # doctest: +SKIP
    ...     runs=[only_spec],
    ...     backend=pool,
    ...     manifest_path=out / "manifest.jsonl",
    ...     output_dir=out,
    ...     script_path=mission,
    ...     parameter_spec={"_kind": "explicit", ...},
    ...     allow_unisolated_pool=True,
    ... )
    >>> with pool:  # doctest: +SKIP
    ...     sweep.run()
    """

    subprocess_isolated: ClassVar[Literal["debug"]] = "debug"

    def __init__(self, *, allow_unisolated_pool: bool = False) -> None:
        if not allow_unisolated_pool:
            raise BackendError(
                "DebugPool violates the subprocess-isolation contract — every "
                "spec runs on the driver process so breakpoint() drops in. "
                "Pass allow_unisolated_pool=True to acknowledge."
            )
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("DebugPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("DebugPool is closed; cannot drain futures")

        for f in list(futures):
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            outcome = run_one(spec)
            f.set_result(outcome)
            yield outcome

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for f in self._pending:
            f.cancel()
        self._pending.clear()
