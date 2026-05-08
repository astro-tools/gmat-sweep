"""Canonical reference benchmark fixture: one sweep, two callers.

A single ``Sat.SMA`` linspace sweep against the LEO basic mission fixture under
:mod:`tests.data.leo_basic` exercises the full gmat-sweep pipeline (script load,
propagator, ReportFile output, Parquet write) at minimal wall-clock cost. The
``--scale N`` parameter dials the run count: the docs benchmark page measures
``--scale 1000``, the CI throughput regression test runs ``--scale 50`` against
the same definition so the docs and CI numbers cannot drift.

Importable as a module — :func:`build_grid`, :func:`build_pool`,
:func:`run_benchmark`, and :func:`assert_meets_floor` are the public surface —
and runnable as ``python -m tests.data.benchmark_sweep`` to print a JSON timing
record on stdout.

The ``k8s`` backend reads two environment variables — ``GMAT_SWEEP_K8S_IMAGE``
and ``GMAT_SWEEP_K8S_PVC`` — set by the CI cell that provisions a kind cluster
and a from-source-built sweep image. Outside that cell those env vars are
unset and ``build_pool("k8s", ...)`` raises a clear error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np

from gmat_sweep import LocalJoblibPool, sweep
from gmat_sweep.backends.base import Pool

__all__ = [
    "BACKENDS",
    "SCRIPT_PATH",
    "Backend",
    "BenchmarkRecord",
    "assert_meets_floor",
    "build_grid",
    "build_pool",
    "main",
    "run_benchmark",
]

Backend = Literal["local", "dask", "ray", "k8s", "mpi"]
BACKENDS: tuple[Backend, ...] = ("local", "dask", "ray", "k8s", "mpi")

SCRIPT_PATH = Path(__file__).resolve().parent / "leo_basic.script"


class BenchmarkRecord(TypedDict):
    """JSON-serialisable timing record returned by :func:`run_benchmark`."""

    backend: str
    workers: int
    scale: int
    n_runs: int
    wall_seconds: float
    throughput_runs_per_sec: float


def build_grid(scale: int) -> dict[str, list[float]]:
    """Canonical sweep grid: ``Sat.SMA`` ∈ ``np.linspace(7000, 8000, scale)``.

    ``scale`` is the run count; the docs benchmark uses ``1000`` and the CI
    throughput regression uses ``50``.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale!r}")
    return {"Sat.SMA": np.linspace(7000.0, 8000.0, scale).tolist()}


def build_pool(backend: Backend, workers: int) -> Pool:
    """Construct the requested backend pool with ``workers`` worker processes.

    :class:`~gmat_sweep.backends.dask.DaskPool`,
    :class:`~gmat_sweep.backends.ray.RayPool`, and
    :class:`~gmat_sweep.backends.kubernetes.KubernetesJobPool` are imported
    lazily so a minimal install (no ``[dask]`` / ``[ray]`` / ``[k8s]``
    extras) can still import this module to read :func:`build_grid` or run
    the local-backend benchmark.
    """
    if backend == "local":
        return LocalJoblibPool(workers=workers)
    if backend == "dask":
        from gmat_sweep.backends.dask import DaskPool

        return DaskPool(n_workers=workers)
    if backend == "ray":
        from gmat_sweep.backends.ray import RayPool

        return RayPool(num_cpus=workers)
    if backend == "k8s":
        from gmat_sweep.backends.kubernetes import KubernetesJobPool

        image = os.environ.get("GMAT_SWEEP_K8S_IMAGE")
        pvc_name = os.environ.get("GMAT_SWEEP_K8S_PVC")
        mount_path = os.environ.get("GMAT_SWEEP_K8S_MOUNT_PATH", "/sweep")
        driver_mount = os.environ.get("GMAT_SWEEP_K8S_DRIVER_MOUNT_PATH", mount_path)
        if not image or not pvc_name:
            raise RuntimeError(
                "k8s backend requires GMAT_SWEEP_K8S_IMAGE and GMAT_SWEEP_K8S_PVC; "
                "the kind-CI cell sets both."
            )
        return KubernetesJobPool(
            image=image,
            pvc_name=pvc_name,
            pvc_mount_path=mount_path,
            driver_mount_path=driver_mount,
            parallelism=workers,
        )
    if backend == "mpi":
        from gmat_sweep.backends.mpi import MPIPool

        return MPIPool(max_workers=workers)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def run_benchmark(*, backend: Backend, scale: int, workers: int) -> BenchmarkRecord:
    """Run the canonical sweep and return wall-clock and throughput metrics.

    The pool is constructed inside this call and closed before timing stops, so
    worker shutdown is included in the measured wall-clock — that matches what
    a downstream caller pays end-to-end.
    """
    grid = build_grid(scale)
    pool = build_pool(backend, workers)
    start = time.perf_counter()
    try:
        df = sweep(SCRIPT_PATH, grid=grid, backend=pool, progress=False)
        n_runs = int(df.index.get_level_values("run_id").nunique())
    finally:
        pool.close()
    wall_seconds = time.perf_counter() - start
    return BenchmarkRecord(
        backend=backend,
        workers=workers,
        scale=scale,
        n_runs=n_runs,
        wall_seconds=wall_seconds,
        throughput_runs_per_sec=n_runs / wall_seconds,
    )


def assert_meets_floor(record: Mapping[str, Any], floor: float) -> None:
    """Raise ``AssertionError`` if ``record``'s throughput is below ``floor``.

    The error message names the backend, the measured rate, the floor, and the
    sweep scale so a CI failure is directly actionable from the log line.
    """
    measured = float(record["throughput_runs_per_sec"])
    if measured < floor:
        raise AssertionError(
            f"{record['backend']} throughput {measured:.3f} runs/sec below floor "
            f"{floor:.3f} runs/sec ({record['scale']}-run scaled sweep)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.data.benchmark_sweep",
        description="Run the canonical gmat-sweep reference benchmark.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1000,
        help="number of runs in the sweep (default: 1000)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="local",
        help="execution backend (default: local)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="worker processes for the backend (default: 8)",
    )
    args = parser.parse_args(argv)

    record = run_benchmark(backend=args.backend, scale=args.scale, workers=args.workers)
    json.dump(cast(dict[str, Any], record), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
