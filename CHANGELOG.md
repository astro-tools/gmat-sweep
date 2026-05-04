# Changelog

All notable changes to gmat-sweep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-04

Initial public release. The MVP slice of the charter: full-factorial parameter
sweeps over a `gmat-run` mission, parallelised across subprocess workers,
aggregated into a `(run_id, time)`-MultiIndexed `pandas.DataFrame`, and backed
by a durable JSON Lines manifest.

### Added

- `sweep(mission, grid=..., workers=...)` — public entry point for
  full-factorial parameter grids. Materialises the cartesian product into one
  [`RunSpec`][gmat_sweep.RunSpec] per cell, dispatches through the default
  [`LocalJoblibPool`][gmat_sweep.Pool] backend, and returns the aggregated
  multi-indexed DataFrame (#9).
- `gmat_sweep.full_factorial` and `gmat_sweep.expand_grid_to_run_specs` —
  cartesian-product expansion with deterministic, lexicographic key ordering
  and zero-based `run_id` assignment, the contract every manifest and future
  resume flow depends on (#4).
- [`Pool`][gmat_sweep.Pool] ABC with the `subprocess_isolated` invariant
  enforced at class-definition time, plus the default
  [`LocalJoblibPool`][gmat_sweep.Pool] implementation backed by joblib's loky
  executor — every run spawns a fresh Python interpreter that imports
  `gmatpy` once, sidestepping the well-known reinit limitation (#7).
- Single-run worker callable that imports `gmat_run`, applies overrides via the
  dotted-path setter, runs the mission, serialises every `ReportFile` to
  Parquet, and converts every exception into a `RunOutcome.failed(...)` so a
  single bad run never aborts the sweep (#6).
- Lazy `(run_id, time)`-MultiIndex aggregation from per-run Parquet via
  pyarrow's dataset API. Failed and skipped runs surface as one row with the
  `__status` column populated (#8).
- JSON Lines manifest with one header line and one fsync'd
  [`ManifestEntry`][gmat_sweep.ManifestEntry] per run. `Manifest.load`
  tolerates a single torn last line (a `Ctrl-C`'d sweep leaves a parseable
  file). Header captures canonical script SHA-256, `gmat_sweep` /
  `gmat_run` / GMAT install / Python / OS versions, the materialised parameter
  spec, and the run count. Includes `find_failed` / `find_missing` lookup
  helpers for the v0.2 resume flow (#5).
- Typed exception hierarchy under `gmat_sweep.errors` rooted at
  [`GmatSweepError`][gmat_sweep.GmatSweepError] —
  [`SweepConfigError`][gmat_sweep.SweepConfigError],
  [`RunFailed`][gmat_sweep.RunFailed],
  [`BackendError`][gmat_sweep.BackendError], and
  [`ManifestCorruptError`][gmat_sweep.ManifestCorruptError] — alongside
  JSON-serialisable [`RunSpec`][gmat_sweep.RunSpec] /
  [`SweepSpec`][gmat_sweep.SweepSpec] / [`RunOutcome`][gmat_sweep.RunOutcome]
  dataclasses (#3).
- `gmat-sweep` console script with `run` and `show` subcommands. `run`
  accepts repeated `--grid name=lo:hi:count` linspace and `--grid
  name=v1,v2,v3` explicit-list axes; `show` prints a one-line summary of an
  existing manifest (#10).
- Validation test suite — per-run round-trip, parameter-spec round-trip
  (float / int / datetime / vector / str enum), 16-run reference-sweep
  regression, failure-mode coverage (invalid override, divergent solver, bad
  script, OOM), and a manifest-replay contract test that pins the v0.1
  manifest format ahead of v0.2's resume flow (#11).
- MkDocs-Material documentation site auto-deployed to GitHub Pages on tag
  pushes, with mkdocstrings-driven API reference, parameter-spec and
  manifest-schema reference pages, supported-versions table, and FAQ
  (#12, #28).
- Three runnable example notebooks rendered into the docs site —
  single-axis SMA scan, two-axis epoch × time-of-flight grid, and a
  surviving-a-kill walkthrough demonstrating manifest durability (#13, #29).
- CI on Ubuntu + Windows × Python 3.10 / 3.11 / 3.12 × R2025a / R2026a (12
  cells) via `astro-tools/setup-gmat`, with both unit and integration suites
  enabled. Coverage gates: ≥ 80 % overall and ≥ 95 % each on `grids.py`,
  `distributions.py`, `manifest.py`, and `aggregate.py` (#2).
- Release workflow: `uv build` → PyPI trusted publishing →
  `gh release create --generate-notes` on `v*` tags (#2).

[0.1.0]: https://github.com/astro-tools/gmat-sweep/releases/tag/v0.1.0
