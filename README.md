# gmat-sweep

[![CI](https://github.com/astro-tools/gmat-sweep/actions/workflows/ci.yml/badge.svg)](https://github.com/astro-tools/gmat-sweep/actions/workflows/ci.yml)
[![Docs](https://github.com/astro-tools/gmat-sweep/actions/workflows/docs.yml/badge.svg)](https://astro-tools.github.io/gmat-sweep/)
[![PyPI](https://img.shields.io/pypi/v/gmat-sweep.svg)](https://pypi.org/project/gmat-sweep/)
[![Python versions](https://img.shields.io/pypi/pyversions/gmat-sweep.svg)](https://pypi.org/project/gmat-sweep/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel from Python.

## What this is

A parallel orchestrator on top of [`gmat-run`](https://github.com/astro-tools/gmat-run)'s
single-run primitive. Point `gmat-sweep` at a working `.script` and either a parameter
grid, an explicit run table, or a perturbation distribution, and it fans the run set
across subprocess workers, aggregates each run's `ReportFile` (and any `EphemerisFile`
or `ContactLocator` outputs) into multi-indexed pandas DataFrames, and writes a JSON
Lines manifest alongside the results so any sweep is reproducible bit-for-bit. Killed
sweeps reload from the manifest and re-run only the missing or failed runs.

The four entry points cover the common shapes:

- [`sweep(grid=...)`](https://astro-tools.github.io/gmat-sweep/parameter-spec/#full-factorial-expansion)
  — full-factorial grid over one or more dotted-path fields.
- [`sweep(samples=DataFrame)`](https://astro-tools.github.io/gmat-sweep/parameter-spec/#explicit-row-sweeps)
  — explicit-row sweep where you pre-build the run set (Halton, Sobol, custom design).
- [`monte_carlo(perturb=...)`](https://astro-tools.github.io/gmat-sweep/monte-carlo/)
  — stochastic dispersion with named distributions and a deterministic seed contract.
- [`latin_hypercube(perturb=...)`](https://astro-tools.github.io/gmat-sweep/parameter-spec/#monte-carlo-vs-latin-hypercube)
  — stratified sampling for variance reduction at small `n`.

## What this is not

- **Not** a single-run runner — that's [`gmat-run`](https://github.com/astro-tools/gmat-run);
  every `gmat-sweep` worker calls into it.
- **Not** a way to build GMAT missions from scratch in Python — see
  [`gmatpyplus`](https://github.com/weasdown/gmatpyplus).
- **Not** a `.script` text generator — see [`pygmat`](https://pypi.org/project/pygmat/).
- **Not** an optimiser. Gradient-, Bayesian-, and population-based optimisation
  (CasADi, pagmo2, scikit-optimize) is a different problem; `gmat-sweep` may serve as the
  parallel evaluator inside one, but it ships no optimiser of its own.
- **Not** a workflow engine. `gmat-sweep` runs homogeneous parametric sweeps of
  one mission; Snakemake / Nextflow / Hamilton manage DAGs of heterogeneous tasks.
  A workflow engine can schedule a `gmat-sweep` step; the converse is not interesting.

## Requirements

- Python 3.10, 3.11, or 3.12.
- [`gmat-run`](https://github.com/astro-tools/gmat-run) ≥ 0.3 — installed as a transitive
  dependency from PyPI. `gmat-sweep` never imports `gmatpy` directly; the import happens
  inside each worker subprocess on first call.
- A local GMAT install. `gmat-sweep` does not ship GMAT binaries; it relies on `gmat-run`'s
  install discovery, which honours `$GMAT_ROOT` or finds a build under a conventional path.
  Download GMAT from the
  [SourceForge release page](https://sourceforge.net/projects/gmat/files/GMAT/) — see
  [`gmat-run`'s install guide](https://astro-tools.github.io/gmat-run/install-gmat/) for the
  unpack-and-discover steps.

### Supported GMAT versions

| GMAT release | Status | CI |
|---|---|---|
| R2026a | Primary development target | Exercised on every PR (Ubuntu + Windows + macOS, Python 3.10/3.11/3.12) |
| R2025a | Supported | Exercised on every PR (Ubuntu + Windows + macOS, Python 3.10/3.11/3.12) |

R2023a and R2024a were never released by the upstream GMAT project; R2025a and R2026a are
the only releases supported.

## Installation

```bash
pip install gmat-sweep
```

The `[examples]` extra pulls in matplotlib for the example notebooks:

```bash
pip install gmat-sweep[examples]
```

## Quick start

```python
from gmat_sweep import LocalJoblibPool, sweep

df = sweep(
    "mission.script",
    grid={"Sat.SMA": [7000, 7100, 7200]},
    backend=LocalJoblibPool(max_workers=8),
)
print(df)
```

That call runs `mission.script` three times — once per `Sat.SMA` value — each in a fresh
subprocess, and returns a `(run_id, time)`-MultiIndexed `pandas.DataFrame` containing
the rows from every run's `ReportFile` plus a `__status` column flagging
`ok` / `failed` / `skipped`. A single failed run lands as a `failed` row with the captured
GMAT stderr in the manifest — never as a silent zero-row DataFrame and never as an
unhandled exception that aborts the whole sweep.

For a stochastic dispersion, swap [`sweep`](https://astro-tools.github.io/gmat-sweep/api/#gmat_sweep.sweep)
for [`monte_carlo`](https://astro-tools.github.io/gmat-sweep/monte-carlo/) and pass a
`perturb` mapping of named distributions:

```python
from gmat_sweep import LocalJoblibPool, monte_carlo

df = monte_carlo(
    "mission.script",
    n=1000,
    perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
    backend=LocalJoblibPool(max_workers=8),
    seed=42,
)
```

Returns the same DataFrame shape as `sweep()`. Per-run sub-seeds derive from `seed` via
`numpy.random.SeedSequence.spawn`, so the draw is bit-reproducible and a resumed sweep
samples the same values for any given `run_id`. See the
[Monte Carlo guide](https://astro-tools.github.io/gmat-sweep/monte-carlo/) for the full
determinism contract and [`latin_hypercube`](https://astro-tools.github.io/gmat-sweep/parameter-spec/#monte-carlo-vs-latin-hypercube)
for the stratified-sampling variant.

By default the per-run Parquet files and the manifest land in a temporary directory
whose lifetime is tied to the returned DataFrame. Pass `out=Path(...)` to keep them —
that's also what enables [resuming a killed sweep](https://astro-tools.github.io/gmat-sweep/resume/)
via `Sweep.from_manifest(...).resume()` or `gmat-sweep resume <manifest>`.

For multi-host sweeps, swap the local pool for `DaskPool` or `RayPool` — same
`sweep()` / `monte_carlo()` / `latin_hypercube()` call shape, different `backend=`:

```python
from gmat_sweep import sweep
from gmat_sweep.backends import DaskPool

with DaskPool(n_workers=8) as pool:
    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000, 7100, 7200]},
        backend=pool,
    )
```

`DaskPool` and `RayPool` ship behind `pip install gmat-sweep[dask]` /
`gmat-sweep[ray]`. See the [backends page](https://astro-tools.github.io/gmat-sweep/backends/)
for the full set of pool patterns and the
[cluster recipes](https://astro-tools.github.io/gmat-sweep/recipes/) for
Slurm / Kubernetes / Ray autoscaling wiring.

A `gmat-sweep` console script is also installed for shell-script and CI use:

```bash
gmat-sweep run         --grid Sat.SMA=7000:7200:3 --workers 8 --out ./sweep mission.script
gmat-sweep run         --grid Sat.SMA=7000:7200:3 --backend dask --workers 8 --out ./sweep mission.script
gmat-sweep monte-carlo --n 1000 --perturb 'Sat.SMA=normal:7100:50' --seed 42 --out ./mc mission.script
gmat-sweep resume      ./mc/manifest.jsonl mission.script --workers 8
gmat-sweep show        ./sweep/manifest.jsonl
```

See the [CLI reference in the docs](https://astro-tools.github.io/gmat-sweep/cli/)
for every subcommand and the full mini-grammar.

## Outputs

Every sweep emits two artefacts:

- The returned **DataFrame** — `(run_id, time)`-MultiIndexed, one column per `ReportFile`
  channel plus the `__status` column. Built lazily from per-run Parquet files via
  pyarrow's dataset API, so a 10,000-run sweep does not have to fit in memory at once.
- A **JSON Lines manifest** (`manifest.jsonl`) — append-only, fsync'd after every entry.
  Records the canonical script SHA-256, software-version fingerprint, full parameter
  spec, and per-run status, timing, output paths, and captured stderr. A `Ctrl-C`
  mid-sweep leaves the manifest in a parseable state. See the
  [manifest schema](https://astro-tools.github.io/gmat-sweep/manifest-schema/) for the
  full contract.

## Documentation

Full docs at **<https://astro-tools.github.io/gmat-sweep/>**, including a
[getting-started guide](https://astro-tools.github.io/gmat-sweep/getting-started/),
the [parameter spec reference](https://astro-tools.github.io/gmat-sweep/parameter-spec/),
the [manifest schema](https://astro-tools.github.io/gmat-sweep/manifest-schema/),
the [supported-version matrix](https://astro-tools.github.io/gmat-sweep/supported-versions/),
the [FAQ](https://astro-tools.github.io/gmat-sweep/faq/),
and the [API reference](https://astro-tools.github.io/gmat-sweep/api/).

Runnable example notebooks:

- [Single-axis SMA scan](https://astro-tools.github.io/gmat-sweep/examples/01_sma_scan/) —
  fifty runs across `np.linspace(7000, 8000, 50)` of `Sat.SMA`, parallel-dispatched and
  overlaid on a single altitude-vs-time plot.
- [Two-axis epoch × time-of-flight grid](https://astro-tools.github.io/gmat-sweep/examples/02_epoch_arrival_grid/) —
  cartesian product over `Sat.Epoch` and a script-level `Variable TOF`, contoured by
  per-run miss distance.
- [Surviving a kill](https://astro-tools.github.io/gmat-sweep/examples/03_killed_sweep_recovery/) —
  launch a sweep, send `SIGINT` mid-run, walk through inspecting the partial manifest
  with `gmat-sweep show`, then complete the sweep with `Sweep.from_manifest(...).resume()`.
- [Monte Carlo dispersion](https://astro-tools.github.io/gmat-sweep/examples/04_monte_carlo_dispersion/) —
  1000-run Monte Carlo around a nominal injection burn over a four-axis perturbation
  cube, with arrival-miss histogram and a 3-σ covariance ellipse.
- [Latin hypercube vs Monte Carlo](https://astro-tools.github.io/gmat-sweep/examples/05_latin_hypercube/) —
  64-run Latin hypercube alongside a 64-run plain Monte Carlo on the same perturbation,
  pair-plotting the unit-cube samples to make the stratification visible.
- [Dask cluster recipe](https://astro-tools.github.io/gmat-sweep/examples/06_dask_cluster_recipe/) —
  100-run `Sat.SMA` grid dispatched through a `distributed.LocalCluster` with `DaskPool`,
  same flow as a real `dask.distributed` cluster.
- [Ray autoscaling recipe](https://astro-tools.github.io/gmat-sweep/examples/07_ray_autoscaling_recipe/) —
  100-run Monte Carlo dispatched through `RayPool` against a local `ray.init()`, same
  task model as a real autoscaling Ray cluster.
- [Sobol sensitivity](https://astro-tools.github.io/gmat-sweep/examples/08_sobol_sensitivity/) —
  Saltelli design via `sobol_sample`, run through `sweep(samples=...)`, reduced to
  first/total-order Sobol indices via `sobol_analyze` with 95 % bootstrap CIs.
- [Archive bundle](https://astro-tools.github.io/gmat-sweep/examples/09_archive_bundle/) —
  pack a finished sweep into a self-describing `.zip` via `Sweep.archive()`, inspect
  the layout, and re-aggregate the per-run DataFrame from the unzipped tree.
- [Extending a Monte Carlo](https://astro-tools.github.io/gmat-sweep/examples/10_extending_monte_carlo/) —
  anchor a 100-run `monte_carlo`, append 200 more via `monte_carlo_extend(n=200)`,
  and assert that the original 100 `run_id`s are preserved bit-for-bit.

## Roadmap

| Release | Scope |
|---|---|
| **v0.3** *(current)* | `DaskPool` (extra `[dask]`) and `RayPool` (extra `[ray]`) join `LocalJoblibPool` behind a single `Pool` ABC. CLI `--backend {local,dask,ray}` flag and rich `gmat-sweep show --detail` / `--run` modes. Three cluster-recipe pages (Slurm with `srun`, Kubernetes pod-per-worker, Ray autoscaling). Benchmark page comparing backends on a 1000-run reference sweep, with a per-backend throughput floor enforced in CI. Manifest header gains a `backend` field; `reuse_gmat_context` exposes the bootstrap-amortisation choice on every pool. |
| **v0.4** *(next)* | `KubernetesJobPool` (extra `[k8s]`) — every run becomes one `batch/v1` Job, no Dask or Ray middleware. Notebook-friendly `__repr_html__` for `Sweep` / `RunOutcome`. Optional plotting helpers (`sweep_corner`, `sweep_heatmap`) behind a `[plot]` extra pulling matplotlib. Sobol sensitivity indices via SALib (extra `[sensitivity]`) — `sobol_sample` builds the Saltelli design, `sobol_analyze` returns first/total/second-order indices with confidence intervals. A docs cookbook page on integrating sweep outputs into downstream consumers. Smoke-canary cell against the canonical `ghcr.io/astro-tools/gmat` image. |

Past releases live in [`CHANGELOG.md`](CHANGELOG.md).

## Development

To work on `gmat-sweep` itself:

```bash
git clone https://github.com/astro-tools/gmat-sweep.git
cd gmat-sweep
uv sync --all-groups
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full branch / PR / test workflow.

## Licence

MIT. See [LICENSE](LICENSE).
