"""Tests for gmat_sweep.backends.base.Pool — ABC contract enforcement."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from typing import ClassVar

import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec


class _MinimalPool(Pool):
    """Smallest concrete Pool used to exercise the ABC machinery from tests."""

    def __init__(self) -> None:
        self.submitted: list[RunSpec] = []
        self.closed = False

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        self.submitted.append(spec)
        return Future()

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        return iter([])

    def close(self) -> None:
        self.closed = True


def test_default_pool_subclass_is_subprocess_isolated() -> None:
    assert _MinimalPool.subprocess_isolated is True
    assert _MinimalPool().subprocess_isolated is True


def test_pool_is_abstract_and_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Pool()  # type: ignore[abstract]


def test_subclass_setting_subprocess_isolated_false_raises() -> None:
    with pytest.raises(BackendError) as ei:

        class _Bad(Pool):
            subprocess_isolated: ClassVar[bool] = False

            def submit(self, spec: RunSpec) -> Future[RunOutcome]:
                return Future()

            def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
                return iter([])

            def close(self) -> None:
                pass

    msg = str(ei.value)
    assert "_Bad" in msg
    assert "False" in msg


def test_subclass_setting_subprocess_isolated_truthy_non_true_raises() -> None:
    with pytest.raises(BackendError):

        class _Bad(Pool):
            subprocess_isolated = "yes"  # type: ignore[assignment]

            def submit(self, spec: RunSpec) -> Future[RunOutcome]:
                return Future()

            def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
                return iter([])

            def close(self) -> None:
                pass


def test_subclass_setting_subprocess_isolated_none_raises() -> None:
    with pytest.raises(BackendError):

        class _Bad(Pool):
            subprocess_isolated = None  # type: ignore[assignment]

            def submit(self, spec: RunSpec) -> Future[RunOutcome]:
                return Future()

            def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
                return iter([])

            def close(self) -> None:
                pass


def test_pool_works_as_context_manager() -> None:
    pool = _MinimalPool()
    with pool as p:
        assert p is pool
        assert not pool.closed
    assert pool.closed
