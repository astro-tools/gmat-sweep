"""Pool ABC: subprocess-isolation as a configurable per-pool choice.

The :class:`Pool` ABC is the abstraction every gmat-sweep execution backend
implements. Two modes are exposed via the ``reuse_gmat_context`` keyword on
each subclass's constructor and stored on the pool instance:

- ``reuse_gmat_context=True`` (default) — the *fast path*. A worker process
  imports ``gmat_run`` once and reuses the resulting state across many tasks;
  bootstrap cost is paid once per worker, then amortised across every
  subsequent task on that worker. Safe **only when every task dispatched
  through the pool loads the same script** — GMAT relies on process-global
  singletons that cannot be reused across runs that load different scripts.
- ``reuse_gmat_context=False`` — the *isolation path*. Every task spawns a
  fresh Python interpreter that bootstraps ``gmatpy`` from scratch via
  :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`. Slower but
  allows arbitrary heterogeneous scripts in a single sweep, and is the
  contract a caller wants when they cannot vouch for same-script discipline.

The :attr:`Pool.subprocess_isolated` class attribute is the structural
marker that a subclass implements both modes correctly; subclasses that try
to opt out are rejected at class-definition time via
:meth:`__init_subclass__` so the misconfiguration surfaces during ``import``,
not after a sweep is half-way through.

The submission surface is intentionally narrow:

- :meth:`submit` enqueues a :class:`gmat_sweep.spec.RunSpec` and returns a
  :class:`concurrent.futures.Future` the caller can hold.
- :meth:`as_completed` drains the supplied futures, yielding
  :class:`gmat_sweep.spec.RunOutcome` instances in completion order.
- :meth:`close` releases backend resources; :class:`Pool` is also a context
  manager so ``with LocalJoblibPool() as pool:`` works.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from types import TracebackType
from typing import Any, ClassVar, Literal

from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

__all__ = ["Pool"]


class Pool(ABC):
    """Abstract execution backend.

    Subclasses MUST keep :attr:`subprocess_isolated` set to :data:`True`,
    signalling that they implement both the per-worker-reuse and per-task-
    fresh-bootstrap modes correctly. Setting the attribute to anything else
    (``False``, ``None``, a truthy non-``True`` value) raises
    :class:`gmat_sweep.errors.BackendError` from :meth:`__init_subclass__` so
    the error fires when the bad backend's module is imported, not at sweep
    time.

    The single recognised opt-out is the literal string ``"debug"``, used by
    :class:`gmat_sweep.backends.debug.DebugPool` to declare in-process,
    single-run dispatch for ``breakpoint()``-driven debugging. Sweeps refuse
    to dispatch through any pool whose ``subprocess_isolated`` is not
    :data:`True` unless the caller acknowledges the violation via
    ``Sweep(..., allow_unisolated_pool=True)``.

    Subclasses accept ``reuse_gmat_context: bool = True`` as a keyword-only
    parameter on ``__init__`` and store it; concrete dispatch in
    :meth:`as_completed` reads ``self._reuse_gmat_context`` to choose between
    calling :func:`gmat_sweep.worker.run_one` directly (fast path) and
    delegating to
    :func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess` (isolation
    path).
    """

    subprocess_isolated: ClassVar[bool | Literal["debug"]] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.subprocess_isolated is not True and cls.subprocess_isolated != "debug":
            raise BackendError(
                f"{cls.__name__}.subprocess_isolated is "
                f"{cls.subprocess_isolated!r}; Pool subclasses must implement "
                "both reuse and isolation modes (set to True or omit), or "
                'declare in-process debug dispatch via the "debug" sentinel.'
            )

    @abstractmethod
    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        """Enqueue ``spec`` and return a future that resolves once it runs."""

    @abstractmethod
    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        """Drain ``futures``, yielding outcomes in completion order."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources. Idempotent."""

    @property
    def max_workers(self) -> int:
        """Best-effort worker count for sizing the in-flight cap.

        Subclasses should override to expose their actual worker count
        (loky's ``n_jobs`` resolved against ``cpu_count``, Ray's
        cluster size, …). Returning ``1`` is the safe fallback when
        the backend cannot answer — the caller's bounded-submit loop
        will dispatch sequentially, which is correct (if slow) for
        every backend.
        """
        return 1

    def imap(
        self,
        specs: Iterable[RunSpec],
        *,
        in_flight: int | None = None,
    ) -> Iterator[tuple[RunSpec, RunOutcome]]:
        """Stream ``specs`` through the pool, yielding ``(spec, outcome)`` pairs.

        Outcomes are yielded in completion order, paired with the
        :class:`RunSpec` that produced them. Bounds the in-flight set
        to ``in_flight`` (default ``4 * self.max_workers``) so a 10⁵-spec
        iterator does not pin 10⁵ payloads + 10⁵ futures in driver memory.
        Specs are pulled from the iterator lazily — only ``in_flight``
        specs are materialised at any one time.

        Each yielded pair carries the :class:`RunSpec` that produced
        the :class:`RunOutcome`: the caller does not need to maintain a
        side-table from ``outcome.run_id`` back to the spec, and the
        spec object becomes garbage-collectable the moment the caller
        finishes processing it.

        Default implementation: chunked submit / drain. Specs are
        consumed ``in_flight`` at a time, each chunk drained through
        :meth:`as_completed` before the next chunk is submitted. This
        bounds RSS but loses pipelining within a chunk — backends with
        true future-by-future progress (e.g. those backed by
        :class:`concurrent.futures.Executor`) should override with a
        sliding-window submit/wait loop for better throughput.
        """
        if in_flight is None:
            in_flight = max(1, 4 * self.max_workers)
        if in_flight < 1:
            raise ValueError(f"in_flight must be >= 1, got {in_flight}")

        iterator = iter(specs)
        while True:
            chunk_specs: list[RunSpec] = []
            chunk_futures: list[Future[RunOutcome]] = []
            for _ in range(in_flight):
                try:
                    spec = next(iterator)
                except StopIteration:
                    break
                chunk_specs.append(spec)
                chunk_futures.append(self.submit(spec))
            if not chunk_specs:
                return
            spec_by_run_id: dict[int, RunSpec] = {s.run_id: s for s in chunk_specs}
            for outcome in self.as_completed(chunk_futures):
                yield spec_by_run_id[outcome.run_id], outcome

    def __enter__(self) -> Pool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
