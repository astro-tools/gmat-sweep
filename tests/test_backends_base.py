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


def test_subclass_setting_subprocess_isolated_to_debug_string_is_accepted() -> None:
    # The "debug" sentinel is the one recognised opt-out — DebugPool uses it
    # to declare in-process single-run dispatch.
    class _Debug(Pool):
        subprocess_isolated: ClassVar[str] = "debug"  # type: ignore[assignment]

        def submit(self, spec: RunSpec) -> Future[RunOutcome]:
            return Future()

        def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
            return iter([])

        def close(self) -> None:
            pass

    assert _Debug.subprocess_isolated == "debug"


def test_subclass_setting_subprocess_isolated_to_unknown_string_raises() -> None:
    # Only the literal "debug" string is accepted; any other string still
    # raises so the contract has exactly one named opt-out.
    with pytest.raises(BackendError):

        class _Bad(Pool):
            subprocess_isolated: ClassVar[str] = "fast"  # type: ignore[assignment]

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


# ---- Pool.imap default implementation (issue #134 item 2) -----------------


class _RecordingPool(Pool):
    """Concrete Pool that records peak in-flight specs so imap behaviour is testable."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self.submitted: list[RunSpec] = []
        self.peak_in_flight = 0
        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self.closed = False

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        self.submitted.append(spec)
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        # Track the peak before draining — the imap default submits a chunk
        # before calling as_completed, so peak == chunk size.
        self.peak_in_flight = max(self.peak_in_flight, len(self._pending))
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        from datetime import datetime, timezone

        for f in list(futures):
            spec = self._pending.pop(f)
            now = datetime.now(timezone.utc)
            outcome = RunOutcome.ok(
                run_id=spec.run_id,
                output_paths={},
                started_at=now,
                ended_at=now,
                duration_s=0.0,
            )
            f.set_result(outcome)
            yield outcome

    def close(self) -> None:
        self.closed = True


def _make_specs(n: int) -> list[RunSpec]:
    from pathlib import Path

    return [
        RunSpec(
            script_path=Path("/m.script"),
            overrides={"x": i},
            output_dir=Path(f"/o/run-{i}"),
            run_id=i,
            seed=None,
            run_options={},
        )
        for i in range(n)
    ]


def test_imap_default_bounds_in_flight_to_four_times_max_workers() -> None:
    """The headline #134 item 2 fix: 100 specs through a 2-worker pool must
    never have more than ``4 * 2 = 8`` futures pending at once."""
    pool = _RecordingPool(max_workers=2)
    specs = _make_specs(100)
    yielded = list(pool.imap(specs))
    assert len(yielded) == 100
    # 8 = chunk size; nothing materialised beyond it.
    assert pool.peak_in_flight == 8
    # All specs were dispatched in run_id order (each chunk submitted in order).
    assert [s.run_id for s, _ in yielded] == list(range(100))


def test_imap_explicit_in_flight_overrides_default() -> None:
    pool = _RecordingPool(max_workers=2)
    yielded = list(pool.imap(_make_specs(20), in_flight=3))
    assert len(yielded) == 20
    assert pool.peak_in_flight == 3


def test_imap_yields_spec_outcome_pairs() -> None:
    """Each yielded pair carries the originating :class:`RunSpec` so the caller
    doesn't need to maintain a run_id -> spec side-table."""
    pool = _RecordingPool(max_workers=1)
    specs = _make_specs(5)
    pairs = list(pool.imap(specs))
    for spec, outcome in pairs:
        assert isinstance(spec, RunSpec)
        assert isinstance(outcome, RunOutcome)
        assert spec.run_id == outcome.run_id


def test_imap_rejects_non_positive_in_flight() -> None:
    pool = _RecordingPool(max_workers=1)
    with pytest.raises(ValueError):
        list(pool.imap(_make_specs(1), in_flight=0))


def test_imap_consumes_iterator_lazily() -> None:
    """Iterator inputs are pulled only as outcomes drain — a true 10⁵-spec
    generator never materialises in full."""
    pool = _RecordingPool(max_workers=1)
    counter = {"pulled": 0}

    def _gen() -> Iterator[RunSpec]:
        for i in range(1000):
            counter["pulled"] += 1
            from pathlib import Path

            yield RunSpec(
                script_path=Path("/m.script"),
                overrides={"x": i},
                output_dir=Path(f"/o/run-{i}"),
                run_id=i,
                seed=None,
                run_options={},
            )

    # Pull just the first 5 outcomes and abandon the iterator.
    it = pool.imap(_gen(), in_flight=4)
    first_five = [next(it) for _ in range(5)]
    assert len(first_five) == 5
    # We've consumed 5 outcomes; the iterator should not have been fully drained
    # (it's a 1000-spec generator). The chunked default pulls in_flight=4 at a
    # time, so after 5 outcomes we've pulled either 4 (first chunk active) or 8
    # (second chunk started). Either way, far less than 1000.
    assert counter["pulled"] < 1000


def test_default_max_workers_is_one() -> None:
    """The base ABC fallback is 1 so unknown backends still get a sane (slow)
    sequential default rather than 0 or a crash."""

    pool = _RecordingPool()  # uses the override
    assert pool.max_workers == 2

    # Reach into _MinimalPool, which does NOT override max_workers.
    minimal = _MinimalPool()
    assert minimal.max_workers == 1
