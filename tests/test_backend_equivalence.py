"""Backend equivalence validation suite — issue #62.

Asserts the ``Pool`` abstraction is honest: the three execution backends
shipped today (:class:`~gmat_sweep.LocalJoblibPool`,
:class:`~gmat_sweep.backends.DaskPool`, :class:`~gmat_sweep.backends.RayPool`)
must produce bit-equal aggregated DataFrames and bit-equal
reproducibility-bearing manifest fields for every reference sweep. The only
manifest field allowed to differ across backends is the ``backend`` header
itself.

Reference sweeps
----------------

- **Grid** — 16-run ``Sat.SMA`` linspace against the LEO basic mission
  fixture. Same shape as :mod:`tests.test_reference_sweep` so a divergence
  here is comparable to that test's golden.
- **Monte Carlo** — 32-run, four-axis perturbation around an injection-burn
  scenario at a fixed seed. Mirrors charter §3's "32-run Monte Carlo" and
  the perturbation cube exercised by
  :file:`docs/examples/04_monte_carlo_dispersion.ipynb`.
- **Latin hypercube** — 16-run LH sweep at a fixed seed against the same
  injection-dispersion fixture.

Test design
-----------

The reference backend is :class:`~gmat_sweep.LocalJoblibPool` (no extras
required, the most-tested code path). Each non-reference backend is
parametrized into one test per sweep shape and compared against a
module-scoped reference sweep computed once. Equivalence is transitive —
if every backend matches the reference, all backends match each other —
so adding a fourth backend is a one-line edit to ``BACKENDS`` in
:mod:`tests.data.benchmark_sweep` (already the source of truth shared
with :mod:`tests.test_backend_throughput`).

Determinism contract
--------------------

Two consecutive Monte Carlo sweeps at the same seed on the same backend
must produce bit-equal DataFrames — this is the v0.2 contract pinned for
``LocalJoblibPool`` by :mod:`tests.test_monte_carlo_determinism`,
restated here for every backend so a Dask or Ray regression that
introduced scheduling-dependent draws would fail this gate.

The Dask path additionally pins **cross-process** determinism: a fresh
driver-process Python interpreter that re-runs the same Monte Carlo sweep
must produce a bit-equal DataFrame. Ray's actor lifecycle makes the same
fixture expensive to set up, so cross-process is Dask-only per the
issue's scope note.

Wall-clock budget
-----------------

Gated as ``integration and slow``; opted into by a single CI cell
(ubuntu-latest, py3.12, R2026a). The 30-minute job timeout has comfortable
headroom over the ~20-minute engineering target.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import gmat_sweep
from gmat_sweep.backends.base import Pool
from gmat_sweep.manifest import Manifest, ManifestEntry

pytestmark = [pytest.mark.integration, pytest.mark.slow]

pytest.importorskip("gmat_run")
pytest.importorskip("distributed")
pytest.importorskip("ray")

# Import after the importorskip guards so a minimal install still collects
# the module cleanly with the expected skips.
from tests.data.benchmark_sweep import BACKENDS, Backend, build_pool  # noqa: E402

REFERENCE_BACKEND: Backend = "local"
CANDIDATE_BACKENDS: tuple[Backend, ...] = tuple(b for b in BACKENDS if b != REFERENCE_BACKEND)

_DATA_DIR = Path(__file__).parent / "data"
_LEO_SCRIPT = _DATA_DIR / "leo_basic.script"
_INJ_SCRIPT = _DATA_DIR / "injection_dispersion.script"
_MANIFEST_FILENAME = "manifest.jsonl"

# Per-sweep wall-clock-friendly sizes. The grid mirrors the existing v0.1
# reference sweep verbatim; the MC count tracks the charter's 32-run
# language; the LH count matches the grid for symmetry.
_GRID = {"Sat.SMA": list(np.linspace(7000.0, 7300.0, 16))}
_MC_N = 32
_LH_N = 16
_SEED = 42

# Four-axis perturbation cube: launch-time-slip proxy plus the three VNB
# components of the injection delta-V. Per-axis sigmas are tight (sub-1 %
# of the nominal magnitude) so the trajectories stay numerically benign
# and the per-run wall-clock cost is uniform across draws.
_PERTURB: Mapping[str, tuple[str, float, float]] = {
    "CoastTime.Value": ("normal", 600.0, 30.0),
    "Inj.Element1": ("normal", 1.0, 0.005),
    "Inj.Element2": ("normal", 0.0, 0.005),
    "Inj.Element3": ("normal", 0.0, 0.005),
}

_WORKERS = 2

# Manifest entry fields that vary by execution and don't carry
# reproducibility meaning — excluded from cross-backend comparison.
# ``output_paths`` and ``log_path`` both embed per-sweep tmp_path roots;
# the timing trio is execution-cost.
_ENTRY_VOLATILE_FIELDS = frozenset(
    {"started_at", "ended_at", "duration_s", "output_paths", "log_path"}
)


@dataclass(frozen=True)
class _SweepResult:
    """Aggregated DataFrame plus the on-disk manifest the sweep produced."""

    df: pd.DataFrame
    manifest: Manifest


def _load_manifest(out: Path) -> Manifest:
    return Manifest.load(out / _MANIFEST_FILENAME)


def _run_grid(backend: Pool, out: Path) -> _SweepResult:
    df = gmat_sweep.sweep(_LEO_SCRIPT, grid=_GRID, backend=backend, out=out, progress=False)
    return _SweepResult(df=df, manifest=_load_manifest(out))


def _run_monte_carlo(backend: Pool, out: Path, *, seed: int = _SEED) -> _SweepResult:
    df = gmat_sweep.monte_carlo(
        _INJ_SCRIPT,
        n=_MC_N,
        perturb=_PERTURB,
        seed=seed,
        backend=backend,
        out=out,
        progress=False,
    )
    return _SweepResult(df=df, manifest=_load_manifest(out))


def _run_latin_hypercube(backend: Pool, out: Path, *, seed: int = _SEED) -> _SweepResult:
    df = gmat_sweep.latin_hypercube(
        _INJ_SCRIPT,
        n=_LH_N,
        perturb=_PERTURB,
        seed=seed,
        backend=backend,
        out=out,
        progress=False,
    )
    return _SweepResult(df=df, manifest=_load_manifest(out))


def _entry_reproducibility_view(entry: ManifestEntry) -> dict[str, Any]:
    """Strip volatile fields from a ``ManifestEntry`` for cross-backend ``==``."""
    payload = entry.to_dict()
    return {k: v for k, v in payload.items() if k not in _ENTRY_VOLATILE_FIELDS}


def _assert_sweep_actually_succeeded(result: _SweepResult, label: str) -> None:
    """Fail fast if the sweep had zero ``ok`` runs.

    Without this guard, an all-failed sweep would trivially pass the
    cross-backend equivalence check — every backend reproduces the same
    NaN-row, status="failed" payload by construction. This catches the
    "fixture script can't even be parsed by GMAT on this runner" class of
    bug at the first sweep, instead of hiding it behind a green test.
    """
    ok_runs = sum(1 for entry in result.manifest.entries if entry.status == "ok")
    assert ok_runs > 0, (
        f"{label}: no runs completed with status=ok — fixture or backend setup "
        f"is broken (every cross-backend equivalence check would pass trivially). "
        f"Inspect worker.log under {result.manifest.entries[0].log_path}."
    )


def _assert_equivalent(reference: _SweepResult, candidate: _SweepResult) -> None:
    """Assert ``candidate`` matches ``reference`` on the cross-backend contract.

    DataFrame equality is enforced exactly (``check_exact=True`` —
    bit-equal floats, identical dtypes, identical MultiIndex). Manifest
    equality is enforced on the reproducibility-bearing fields only:
    ``parameter_spec``, ``sweep_seed``, the per-``run_id`` ``overrides``,
    and the run set's ``run_id`` cardinality. Per-entry timing and
    ``output_paths`` (which embed per-sweep ``tmp_path`` roots) are
    excluded. The ``backend`` header is asserted to *differ* — proves the
    two sweeps actually went through different pools.
    """
    _assert_sweep_actually_succeeded(reference, "reference sweep")
    _assert_sweep_actually_succeeded(candidate, "candidate sweep")

    pd.testing.assert_frame_equal(reference.df, candidate.df, check_exact=True)

    ref_m = reference.manifest
    cand_m = candidate.manifest

    assert ref_m.backend != cand_m.backend, (
        f"backends are not distinct: both manifests report backend={ref_m.backend!r}"
    )
    assert ref_m.parameter_spec == cand_m.parameter_spec
    assert ref_m.sweep_seed == cand_m.sweep_seed
    assert ref_m.run_count == cand_m.run_count

    ref_entries = {e.run_id: _entry_reproducibility_view(e) for e in ref_m.entries}
    cand_entries = {e.run_id: _entry_reproducibility_view(e) for e in cand_m.entries}
    assert ref_entries == cand_entries


# ---- reference fixtures (one local-backend sweep per shape, cached) ------


def _build_reference_pool() -> Pool:
    return build_pool(REFERENCE_BACKEND, workers=_WORKERS)


@pytest.fixture(scope="module")
def reference_grid(tmp_path_factory: pytest.TempPathFactory) -> _SweepResult:
    pool = _build_reference_pool()
    try:
        return _run_grid(pool, tmp_path_factory.mktemp("ref-grid"))
    finally:
        pool.close()


@pytest.fixture(scope="module")
def reference_monte_carlo(tmp_path_factory: pytest.TempPathFactory) -> _SweepResult:
    pool = _build_reference_pool()
    try:
        return _run_monte_carlo(pool, tmp_path_factory.mktemp("ref-mc"))
    finally:
        pool.close()


@pytest.fixture(scope="module")
def reference_latin_hypercube(tmp_path_factory: pytest.TempPathFactory) -> _SweepResult:
    pool = _build_reference_pool()
    try:
        return _run_latin_hypercube(pool, tmp_path_factory.mktemp("ref-lh"))
    finally:
        pool.close()


# ---- candidate-vs-reference cross-backend equivalence --------------------


@pytest.mark.parametrize("backend_name", CANDIDATE_BACKENDS)
def test_grid_sweep_matches_reference_backend(
    backend_name: Backend, reference_grid: _SweepResult, tmp_path: Path
) -> None:
    pool = build_pool(backend_name, workers=_WORKERS)
    try:
        candidate = _run_grid(pool, tmp_path / backend_name)
    finally:
        pool.close()
    _assert_equivalent(reference_grid, candidate)


@pytest.mark.parametrize("backend_name", CANDIDATE_BACKENDS)
def test_monte_carlo_sweep_matches_reference_backend(
    backend_name: Backend, reference_monte_carlo: _SweepResult, tmp_path: Path
) -> None:
    pool = build_pool(backend_name, workers=_WORKERS)
    try:
        candidate = _run_monte_carlo(pool, tmp_path / backend_name)
    finally:
        pool.close()
    _assert_equivalent(reference_monte_carlo, candidate)


@pytest.mark.parametrize("backend_name", CANDIDATE_BACKENDS)
def test_latin_hypercube_sweep_matches_reference_backend(
    backend_name: Backend, reference_latin_hypercube: _SweepResult, tmp_path: Path
) -> None:
    pool = build_pool(backend_name, workers=_WORKERS)
    try:
        candidate = _run_latin_hypercube(pool, tmp_path / backend_name)
    finally:
        pool.close()
    _assert_equivalent(reference_latin_hypercube, candidate)


# ---- same-backend determinism (v0.2 contract restated for every backend) -


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_monte_carlo_same_backend_repeatable(backend_name: Backend, tmp_path: Path) -> None:
    """Two MC sweeps at the same seed on the same backend → bit-equal DataFrames.

    Restates the v0.2 ``LocalJoblibPool`` determinism contract pinned by
    :mod:`tests.test_monte_carlo_determinism` for the Dask and Ray paths.
    """
    pool_a = build_pool(backend_name, workers=_WORKERS)
    try:
        result_a = _run_monte_carlo(pool_a, tmp_path / "a")
    finally:
        pool_a.close()

    pool_b = build_pool(backend_name, workers=_WORKERS)
    try:
        result_b = _run_monte_carlo(pool_b, tmp_path / "b")
    finally:
        pool_b.close()

    _assert_sweep_actually_succeeded(result_a, f"first MC sweep on {backend_name}")
    _assert_sweep_actually_succeeded(result_b, f"second MC sweep on {backend_name}")
    pd.testing.assert_frame_equal(result_a.df, result_b.df, check_exact=True)


# ---- cross-process determinism (Dask path only) --------------------------


# A fresh driver-process Python that runs the same Monte Carlo sweep through
# DaskPool and serialises its aggregated DataFrame to Parquet. The in-process
# driver loads the Parquet back and asserts bit-equality with its own sweep.
# Module scope (``__main__`` guard absent) keeps the body inline-executable
# under ``python -c``.
_CROSS_PROCESS_SCRIPT = """
import sys
from pathlib import Path

import gmat_sweep
from gmat_sweep.backends.dask import DaskPool

inj_script = Path({inj_script!r})
out = Path({out!r})
n = {n!r}
seed = {seed!r}
workers = {workers!r}
perturb = {{
    "CoastTime.Value": ("normal", 600.0, 30.0),
    "Inj.Element1": ("normal", 1.0, 0.005),
    "Inj.Element2": ("normal", 0.0, 0.005),
    "Inj.Element3": ("normal", 0.0, 0.005),
}}

pool = DaskPool(n_workers=workers, threads_per_worker=1)
try:
    df = gmat_sweep.monte_carlo(
        inj_script,
        n=n,
        perturb=perturb,
        seed=seed,
        backend=pool,
        out=out,
        progress=False,
    )
finally:
    pool.close()

# reset_index for a clean Parquet round-trip; the in-process driver
# reverses this before comparing.
df.reset_index().to_parquet(out / "result.parquet")
"""


def test_monte_carlo_dask_cross_process_determinism(tmp_path: Path) -> None:
    """Driver-process restart between Dask MC sweeps must yield bit-equal DataFrames.

    Guards against any process-affected RNG state (an unseeded global
    ``np.random``, a worker-startup hook that perturbs draws) silently
    breaking Monte Carlo replay between an original sweep and a resumed
    sweep run from a fresh interpreter. Scoped to Dask per the issue's
    scope note — Ray's actor lifecycle makes the same fixture expensive to
    set up, and the regression class this guards is execution-backend
    agnostic.
    """
    in_process_out = tmp_path / "in-process"
    pool = build_pool("dask", workers=_WORKERS)
    try:
        in_process = _run_monte_carlo(pool, in_process_out)
    finally:
        pool.close()
    _assert_sweep_actually_succeeded(in_process, "in-process Dask MC sweep")

    cross_process_out = tmp_path / "cross-process"
    code = _CROSS_PROCESS_SCRIPT.format(
        inj_script=str(_INJ_SCRIPT),
        out=str(cross_process_out),
        n=_MC_N,
        seed=_SEED,
        workers=_WORKERS,
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"cross-process Dask sweep failed (rc={completed.returncode}): "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )

    cross_process_df = cast(
        pd.DataFrame,
        pd.read_parquet(cross_process_out / "result.parquet").set_index(["run_id", "time"]),
    )

    pd.testing.assert_frame_equal(in_process.df, cross_process_df, check_exact=True)

    # Also pin the per-run overrides cross-process — the assembled DataFrame
    # carries column values, but the manifest's ``overrides`` dict is the
    # canonical record of what each run was *asked* to do. A divergence here
    # would mean the spec-generation RNG itself drifted across processes,
    # which is a stricter signal than "the resulting numbers happened to
    # match".
    cross_process_manifest = _load_manifest(cross_process_out)
    in_process_overrides = {e.run_id: e.overrides for e in in_process.manifest.entries}
    cross_process_overrides = {e.run_id: e.overrides for e in cross_process_manifest.entries}
    assert in_process_overrides == cross_process_overrides
