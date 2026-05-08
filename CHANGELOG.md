# Changelog

All notable changes to gmat-sweep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-05-07

The cluster-backends release: `DaskPool` and `RayPool` join the
`Pool` ABC behind opt-in extras, the CLI gains a `--backend` flag
and a rich `show --detail` / `show --run` mode, three cluster-recipe
pages and two example notebooks document the multi-host story
end-to-end, and a 1000-run benchmark sweep with a per-backend
throughput floor lands in CI. Trove classifier moves from
`Development Status :: 3 - Alpha` to `4 - Beta`.

### Added

- `DaskPool` (`gmat-sweep[dask]`) — distributed pool over
  `dask.distributed`. Spawns a `LocalCluster` by default; `client=`
  accepts an existing `distributed.Client` so a sweep plugs into an
  already-running cluster (Slurm, Kubernetes, a long-lived dev
  scheduler). Imported lazily — a minimal install never imports
  `distributed` (#57).
- `RayPool` (`gmat-sweep[ray]`) — distributed pool over Ray. Calls
  `ray.init` for you by default; `address=` connects to a pre-existing
  cluster (`"auto"`, `"ray://host:port"`, or a raw GCS address).
  Owns the runtime only when its own `__init__` initialised it; an
  externally-initialised runtime is left alone on `close()` (#58).
- `backend=` keyword on [`sweep`][gmat_sweep.sweep],
  [`monte_carlo`][gmat_sweep.monte_carlo], and
  [`latin_hypercube`][gmat_sweep.latin_hypercube]. Replaces the
  former `workers=` shorthand: pass any constructed `Pool` to
  control execution. The `backend=None` default still constructs
  a fresh `LocalJoblibPool` over every available core (#56).
- `reuse_gmat_context: bool = True` keyword on every `Pool`
  constructor. `True` (the default) lets a worker process import
  `gmat_run` once and reuse the bootstrap across many tasks —
  paid once per worker. `False` spawns a fresh Python interpreter
  per task via `python -m gmat_sweep._run_subprocess`. Same flag,
  same semantics on `LocalJoblibPool`, `DaskPool`, and `RayPool` (#78).
- Manifest header carries a new `backend` field — the pool's
  `__class__.__name__` (e.g. `"LocalJoblibPool"`, `"DaskPool"`,
  `"RayPool"`). Additive within `schema_version=1`: manifests
  written before this field landed load with `backend == "unknown"`.
- `_run_subprocess` module — `python -m gmat_sweep._run_subprocess
  <spec.json> <outcome.json>` runs one `RunSpec` in the calling
  interpreter and writes the resulting `RunOutcome` back. Internal
  surface; the subprocess hop the `reuse_gmat_context=False` path
  uses on every backend (#55).
- CLI `--backend {local,dask,ray}` flag and repeatable
  `--backend-arg KEY=VALUE` escape hatch on every sweep-running
  subcommand (`run`, `monte-carlo`, `latin-hypercube`, `explicit`)
  and on `resume`. `--workers N` maps onto each backend in the
  natural way (`workers` / `n_workers` / `num_cpus`). Missing
  extras exit with code `4` and a `pip install gmat-sweep[…]`
  message on stderr (#59).
- `gmat-sweep show --detail` — per-run table sorted with `failed`
  first, then `skipped`, then `ok`, plus the existing one-line
  summary trailer. `gmat-sweep show --run N` prints `run_id=N`'s
  full record: header fields, override dict, full unsuppressed
  `stderr`. `--filter STATUS` narrows the table to a single
  bucket. `--detail` and `--run` are mutually exclusive (#60).
- New documentation pages: [`docs/backends.md`](docs/backends.md)
  (the three pools, the `reuse_gmat_context` contract, and the
  backend-equivalence guarantee), three cluster-recipe pages —
  [`recipes/slurm.md`](docs/recipes/slurm.md),
  [`recipes/kubernetes.md`](docs/recipes/kubernetes.md),
  [`recipes/ray-autoscaling.md`](docs/recipes/ray-autoscaling.md) —
  and [`docs/benchmarks.md`](docs/benchmarks.md) (1000-run
  reference sweep numbers per backend with reproduce commands)
  (#63, #61).
- New runnable example notebooks rendered into the docs site:
  `06_dask_cluster_recipe.ipynb` (100-run grid through a
  `distributed.LocalCluster` with `DaskPool`) and
  `07_ray_autoscaling_recipe.ipynb` (100-run Monte Carlo through
  `RayPool` against a local `ray.init()`) (#64).
- Backend-equivalence validation suite — `tests/test_backend_equivalence.py`
  runs a 16-run grid sweep, a 32-run Monte Carlo sweep, and a 16-run
  Latin hypercube sweep on each of `LocalJoblibPool` / `DaskPool` /
  `RayPool` and asserts pairwise bit-equality on the aggregated
  DataFrame and the manifest's reproducibility-bearing fields. Cross-
  process determinism on `DaskPool` is pinned in the same suite.
  Gated as `integration and slow`; runs on a dedicated CI cell on
  every PR (#62).
- Per-backend throughput regression — `tests/test_backend_throughput.py`
  runs a 50-run benchmark on each of the three backends and asserts
  measured throughput meets the floor in
  `tests/data/throughput_floor.json`. The 1000-run docs numbers and
  the 50-run CI gate share a single sweep-fixture definition
  (`tests/data/benchmark_sweep.py`) so the two cannot drift (#61).
- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` set at backend-package import
  time (`gmat_sweep.backends.__init__`). Disables Ray's auto-`uv`
  `runtime_env` hook before any `import ray`, which would otherwise
  rebuild the worker venv from the project's *base* dependencies
  under `uv run` and fail worker startup with `ModuleNotFoundError:
  No module named 'ray'`. `setdefault` respects an explicit user
  opt-in (#76).

### Changed

- The `workers=N` keyword on [`sweep`][gmat_sweep.sweep],
  [`monte_carlo`][gmat_sweep.monte_carlo], and
  [`latin_hypercube`][gmat_sweep.latin_hypercube] is replaced by
  `backend=`. **Breaking:** a v0.2 caller passing `workers=8` must
  now pass `backend=LocalJoblibPool(workers=8)`. The migration is
  one line at every call site (#56).
- `DaskPool` and `RayPool` default to `reuse_gmat_context=True` —
  a worker process imports `gmat_run` once and reuses the bootstrap
  across many tasks. Safe only when every spec dispatched through
  the pool loads the same script (the common case). Callers that
  compose a single Dask or Ray pool across calls that load different
  scripts must pass `reuse_gmat_context=False` (#78).
- Coverage gate held at ≥ 85 % (the v0.2 number); the four per-file
  95 % gates on `grids.py`, `distributions.py`, `manifest.py`, and
  `aggregate.py` are unchanged. A bump toward the v1.0 ≥ 90 %
  charter target is deferred to a follow-up so the cut PR stayed
  focused on release mechanics.

[0.3.0]: https://github.com/astro-tools/gmat-sweep/releases/tag/v0.3.0

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
