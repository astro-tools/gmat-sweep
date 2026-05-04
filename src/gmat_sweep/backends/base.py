"""Pool ABC: subprocess-isolation contract enforced at class-definition time.

The :class:`Pool` ABC is the abstraction every gmat-sweep execution backend
implements. Each backend must guarantee that GMAT runs in a fresh Python
interpreter — ``gmatpy`` bootstrap is heavy and GMAT itself relies on
process-global singletons that cannot be safely reused across runs that load
different scripts. The :attr:`Pool.subprocess_isolated` class attribute is the
contract; subclasses that try to opt out are rejected at class-definition time
via :meth:`__init_subclass__` so the misconfiguration surfaces during
``import``, not after a sweep is half-way through.

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
from typing import Any, ClassVar

from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

__all__ = ["Pool"]


class Pool(ABC):
    """Abstract execution backend with a subprocess-isolation contract.

    Subclasses MUST keep :attr:`subprocess_isolated` set to :data:`True`.
    Setting it to anything else (``False``, ``None``, a truthy non-``True``
    value) raises :class:`gmat_sweep.errors.BackendError` from
    :meth:`__init_subclass__` so the error fires when the bad backend's module
    is imported, not at sweep time.
    """

    subprocess_isolated: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.subprocess_isolated is not True:
            raise BackendError(
                f"{cls.__name__}.subprocess_isolated is "
                f"{cls.subprocess_isolated!r}; Pool subclasses must guarantee "
                "one fresh Python interpreter per run (set to True or omit)."
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

    def __enter__(self) -> Pool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
