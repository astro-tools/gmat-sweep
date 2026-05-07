"""Unit-level coverage for the helpers that back the throughput regression test.

The integration test in :mod:`tests.test_backend_throughput` requires a real
GMAT install on three backends, so the contract that *a deliberate slowdown
trips the regression test with an actionable message* is exercised here against
synthesised :class:`~tests.data.benchmark_sweep.BenchmarkRecord` records — the
contract holds even when GMAT is not available on the runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.data.benchmark_sweep import (
    BACKENDS,
    BenchmarkRecord,
    assert_meets_floor,
    build_grid,
    build_pool,
)


def _record(throughput: float, *, backend: str = "local", scale: int = 50) -> BenchmarkRecord:
    return BenchmarkRecord(
        backend=backend,
        workers=4,
        scale=scale,
        n_runs=scale,
        wall_seconds=scale / throughput,
        throughput_runs_per_sec=throughput,
    )


def test_build_grid_shape() -> None:
    grid = build_grid(50)
    assert list(grid) == ["Sat.SMA"]
    assert len(grid["Sat.SMA"]) == 50
    assert grid["Sat.SMA"][0] == pytest.approx(7000.0)
    assert grid["Sat.SMA"][-1] == pytest.approx(8000.0)
    np.testing.assert_allclose(grid["Sat.SMA"], np.linspace(7000.0, 8000.0, 50).tolist())


def test_build_grid_rejects_zero_scale() -> None:
    with pytest.raises(ValueError, match="scale must be >= 1"):
        build_grid(0)


def test_build_pool_rejects_unknown_backend() -> None:
    # Cast away the Literal at the call site so the runtime-validation path is
    # reachable from a typed test.
    with pytest.raises(ValueError, match="unknown backend"):
        build_pool("nope", workers=1)  # type: ignore[arg-type]


def test_assert_meets_floor_passes_at_or_above_floor() -> None:
    assert_meets_floor(_record(2.0), floor=2.0)
    assert_meets_floor(_record(2.5), floor=2.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_assert_meets_floor_fails_below_floor_with_actionable_message(backend: str) -> None:
    record = _record(0.5, backend=backend, scale=50)
    with pytest.raises(AssertionError) as exc_info:
        assert_meets_floor(record, floor=1.0)
    msg = str(exc_info.value)
    # Failure line names the backend, the measured rate, the floor, and the
    # scale so a CI log reader can act on it without re-running locally.
    assert backend in msg
    assert "0.500 runs/sec" in msg
    assert "1.000 runs/sec" in msg
    assert "50-run scaled sweep" in msg
