# gmat-sweep

[![CI](https://github.com/astro-tools/gmat-sweep/actions/workflows/ci.yml/badge.svg)](https://github.com/astro-tools/gmat-sweep/actions/workflows/ci.yml)
[![Docs](https://github.com/astro-tools/gmat-sweep/actions/workflows/docs.yml/badge.svg)](https://astro-tools.github.io/gmat-sweep/)
[![PyPI](https://img.shields.io/pypi/v/gmat-sweep.svg)](https://pypi.org/project/gmat-sweep/)
[![Python versions](https://img.shields.io/pypi/pyversions/gmat-sweep.svg)](https://pypi.org/project/gmat-sweep/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel from Python.

## What this is

A parallel orchestrator on top of [`gmat-run`](https://github.com/astro-tools/gmat-run)'s
single-run primitive. Point `gmat-sweep` at a working `.script`, declare a parameter grid,
and it fans the cartesian product across subprocess workers, aggregates each run's
`ReportFile` into a single `(run_id, time)`-MultiIndexed pandas DataFrame, and writes a
JSON Lines manifest alongside the results so any sweep is reproducible bit-for-bit.

## What this is not

- **Not** a single-run runner — that's [`gmat-run`](https://github.com/astro-tools/gmat-run);
  every `gmat-sweep` worker calls into it.
- **Not** a way to build GMAT missions from scratch in Python — see
  [`gmatpyplus`](https://github.com/weasdown/gmatpyplus).
- **Not** a `.script` text generator — see [`pygmat`](https://pypi.org/project/pygmat/).
- **Not** an optimiser. Gradient-, Bayesian-, and population-based optimisation
  (CasADi, pagmo2, scikit-optimize) is a different problem; `gmat-sweep` may serve as the
  parallel evaluator inside one, but it ships no optimiser of its own.
- Monte Carlo dispersion (`monte_carlo`), Latin hypercube sampling (`latin_hypercube`),
  and programmatic resume of partial sweeps land in **v0.2** — see the roadmap below. v0.1
  ships the full-factorial grid path and the durability contract those features build on.

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
from gmat_sweep import sweep

df = sweep(
    "mission.script",
    grid={"Sat.SMA": [7000, 7100, 7200]},
    workers=8,
)
print(df)
```

That call runs `mission.script` three times — once per `Sat.SMA` value — each in a fresh
subprocess, and returns a `(run_id, time)`-MultiIndexed `pandas.DataFrame` containing
the rows from every run's `ReportFile` plus a `__status` column flagging
`ok` / `failed` / `skipped`. A single failed run lands as a `failed` row with the captured
GMAT stderr in the manifest — never as a silent zero-row DataFrame and never as an
unhandled exception that aborts the whole sweep.

By default the per-run Parquet files and the manifest land in a temporary directory
whose lifetime is tied to the returned DataFrame. Pass `out=Path(...)` to keep them.

A `gmat-sweep` console script is also installed for shell-script and CI use:

```bash
gmat-sweep run --grid Sat.SMA=7000:7200:3 --workers 8 --out ./sweep mission.script
gmat-sweep show ./sweep/manifest.jsonl
```

See the [CLI reference in the docs](https://astro-tools.github.io/gmat-sweep/parameter-spec/#cli-mini-grammar)
for the full grid grammar.

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
  launch a sweep, send `SIGINT` mid-run, and walk through inspecting the partial manifest
  with `gmat-sweep show` before reloading the partial DataFrame from disk.

## Roadmap

| Release | Scope |
|---|---|
| **v0.1** *(current)* | Full-factorial `sweep(grid=...)`. `LocalJoblibPool` default backend with subprocess isolation per run. Lazy `(run_id, time)` aggregation from per-run Parquet. JSON Lines manifest with append/fsync durability. `gmat-sweep run`/`show` CLI. Ubuntu + Windows CI on R2025a + R2026a × Python 3.10/3.11/3.12. |
| **v0.2** *(next)* | `monte_carlo()` and `latin_hypercube()` plus explicit-row `samples=DataFrame` sweeps. Programmatic resume via `Sweep.from_manifest(...).resume()`. Ephemeris and contact aggregation across runs. Manifest format frozen as a stable v1 schema. Coverage gate raised to 85%. |

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
