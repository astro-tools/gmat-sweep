# Changelog

All notable changes to gmat-sweep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-05

The stochastic-sweep release: `monte_carlo()` and `latin_hypercube()`
land alongside explicit-row sweeps and a programmatic resume flow,
ephemeris and contact outputs aggregate across runs the same way
reports do, and the manifest format is frozen as a stable v1 schema.
macOS joins the CI matrix.

### Added

- `monte_carlo(mission, *, n, perturb, seed=...)` — stochastic dispersion
  sweep. Per-parameter sub-seeds are derived from the parameter *name*
  via `numpy.random.SeedSequence.spawn`, so adding a perturbed parameter
  does not change the draws of any other parameter at any `run_id` (#33).
- `latin_hypercube(mission, *, n, perturb, seed=...)` — stratified
  sampling backed by [`scipy.stats.qmc.LatinHypercube`][scipy-lh],
  ppf-transformed through each user-supplied distribution (#35).
- `sweep(samples=DataFrame)` — explicit-row sweep variant. The
  DataFrame's columns are dotted-path parameter names; one row per
  run, `run_id` = row index. `grid=` and `samples=` are mutually
  exclusive (#34).
- `Sweep.from_manifest(manifest_path, script_path, *, backend,
  allow_script_drift=False)` — rebuild a Sweep from an existing
  manifest. Dispatches on `parameter_spec["_kind"]` and validates the
  recorded `script_sha256` (#36).
- `Sweep.resume()` — re-submits only the union of `find_failed()` and
  `find_missing(...)`, appending new entries with the same `run_id` so
  successful runs' Parquet files are reused. Monte Carlo and Latin
  hypercube resumes draw bit-equal values to the original sweep (#36).
- `gmat_sweep.distributions` — `DistSpec` shorthand types, `to_rv_frozen`
  coercion with strict up-front validation, `derive_run_seeds` (the
  Monte Carlo replay contract: per-run sub-seeds via
  `numpy.random.SeedSequence.spawn`), `derive_param_seed`, and
  `sample` (#31, #33).
- `lazy_ephemerides(manifest, output_dir, *, name=None)` and
  `lazy_contacts(manifest, output_dir, *, name=None)` — mirror
  `lazy_multiindex` for `EphemerisFile` and `ContactLocator` outputs
  with `(run_id, time)` and `(run_id, interval_id)` index shapes
  respectively. Failed/skipped runs and ok runs that did not produce
  the requested output kind surface as one NaN row per run carrying
  `__status` (#32).
- `Sweep.to_dataframe(name=...)`, `Sweep.to_ephemerides(name=...)`,
  `Sweep.to_contacts(name=...)` — manifest-bound convenience wrappers
  for the lazy aggregators (#32).
- `gmat-sweep monte-carlo`, `gmat-sweep latin-hypercube`,
  `gmat-sweep explicit`, and `gmat-sweep resume` CLI subcommands. The
  stochastic subcommands take repeatable
  `--perturb 'name=tag:p1:p2'` flags; `explicit` loads a CSV/Parquet
  sample design; `resume` requires `--script PATH` and validates the
  canonical hash unless `--allow-script-drift` is set (#37).
- Manifest header carries `schema_version=1`; `MANIFEST_SCHEMA_VERSION`
  is exported as the running writer's emitted value and the maximum it
  accepts on load. Untagged v0.1 manifests load as `schema_version=1`
  for backwards compatibility (#39).
- `parameter_spec` carries a `_kind` discriminator
  (`"grid"` / `"explicit"` / `"monte_carlo"` / `"latin_hypercube"`).
  Untagged v0.1 grid manifests are dispatched as `"grid"` (#39, #34, #33).
- New documentation pages: `docs/monte-carlo.md` (#33), `docs/resume.md`
  (#36), `docs/cli.md` covering all six subcommands (#37). Manifest
  schema page rewritten as the canonical v1 reference with a
  **Compatibility policy** enumerating which schema changes require a
  version bump (#39).
- New runnable example notebooks rendered into the docs site:
  `04_monte_carlo_dispersion.ipynb` (1000-run MC over a four-axis
  injection-burn perturbation, miss-distance histogram, 3-σ covariance
  ellipse, determinism-contract recipe) and `05_latin_hypercube.ipynb`
  (64-run LH alongside a 64-run plain MC, unit-cube pair plot, stacked
  miss-distance histogram). The kill-recovery notebook now closes
  with a real `Sweep.from_manifest(...).resume().to_dataframe()`
  call (#41, #36).
- macOS (Apple Silicon) joined the CI matrix; `test` job now covers
  `{ubuntu-latest, windows-latest, macos-latest} × {3.10, 3.11, 3.12} ×
  {R2025a, R2026a}` = 18 cells (#38).
- Validation suite for the v0.2 surface: Monte Carlo determinism
  (including cross-process bit-equality), Latin hypercube
  stratification, ephemeris/contact aggregation, resume round-trip,
  and explicit-row round-trip. Replaces the v0.1 forward-only
  manifest-replay placeholder; the unknown-extra-header-fields
  forward-compat assertion ports into `tests/test_manifest.py` (#40).

### Changed

- The worker writes per-run Parquet outputs with kind-prefixed basenames
  (`report__<name>.parquet`, `ephemeris__<name>.parquet`,
  `contact__<name>.parquet`); `output_paths` keys carry the same prefix
  so the aggregator can dispatch on output kind without reading the
  file. **Breaking:** v0.1 manifests are not readable by v0.2
  aggregators because their `output_paths` entries lack the prefix —
  re-run any sweep you need to re-aggregate. Documented in
  `docs/aggregation.md` (#32).
- `Manifest.load` folds duplicate `run_id`s last-wins, so a resume
  appends new entries for re-run rows without rewriting the file. The
  on-disk file remains append-only (#36).
- Overall coverage gate raised from ≥ 80 % to ≥ 85 %. The four per-file
  95 % gates on `grids.py`, `distributions.py`, `manifest.py`, and
  `aggregate.py` are unchanged.

[0.2.0]: https://github.com/astro-tools/gmat-sweep/releases/tag/v0.2.0
[scipy-lh]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html

## [0.1.0] — 2026-05-04

Initial public release. The MVP slice: full-factorial parameter sweeps over
a `gmat-run` mission, parallelised across subprocess workers, aggregated
into a `(run_id, time)`-MultiIndexed `pandas.DataFrame`, and backed by a
durable JSON Lines manifest.

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
