"""Per-backend throughput regression test.

Runs the 50-run scaled variant of the canonical reference sweep
(:mod:`tests.data.benchmark_sweep`) on each of :class:`LocalJoblibPool`,
:class:`DaskPool`, and :class:`RayPool`, and asserts each backend's measured
throughput meets the per-backend floor recorded in
``tests/data/throughput_floor.json``. A regression below the floor fails CI with
a message naming the backend, the measured rate, and the floor.

Updating the floor
------------------

The floor JSON is a deliberate diff in PRs. After the first green CI run on a
fresh runner, tighten each entry to roughly ``0.7 * first_green_run_throughput``
- 30% headroom against natural runner-to-runner variance on the
GitHub-Actions free tier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.data.benchmark_sweep import BACKENDS, Backend, assert_meets_floor, run_benchmark

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("gmat_run")


_FLOOR_PATH = Path(__file__).parent / "data" / "throughput_floor.json"
_CI_SCALE = 50
_CI_WORKERS = 4


def _load_floor() -> dict[str, float]:
    return cast(dict[str, float], json.loads(_FLOOR_PATH.read_text()))


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_throughput_meets_floor(backend: Backend) -> None:
    if backend == "dask":
        pytest.importorskip("distributed")
    if backend == "ray":
        pytest.importorskip("ray")

    floor = _load_floor()[backend]
    record = run_benchmark(backend=backend, scale=_CI_SCALE, workers=_CI_WORKERS)
    # Surface the measured record so a CI log reader can recalibrate the floor
    # ('roughly 0.7 * first_green_run_throughput') without re-running locally.
    # `pytest -s` in the dedicated CI cell keeps this visible on a passing run.
    print(json.dumps(dict(record), indent=2))
    assert_meets_floor(record, floor)
