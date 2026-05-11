# Changelog

All notable changes to gmat-sweep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-10

The quality-of-life and ecosystem release: three new execution
backends join `KubernetesJobPool` (`MPIPool`,
`ProcessPoolExecutorPool`, `DebugPool`), six DataFrame helpers land
on top of the aggregator (`sweep_summary`, `sweep_diff`,
`mc_convergence`, `lazy_fused_reports`, plus `monte_carlo_extend` /
`latin_hypercube_extend`), Sobol sensitivity and matplotlib plot
helpers ship behind opt-in extras, an opt-in Polars output engine
runs alongside the pandas default, `Sweep.archive` packs a finished
sweep into a deterministic `.zip` for Zenodo / JOSS deposits, and
notebook-friendly `_repr_html_` rendering lands on `Sweep` /
`RunOutcome` / `ManifestEntry`. CI gains a smoke-canary cell against
the canonical `ghcr.io/astro-tools/gmat` image; per-PR cells trim
from 18 to 10 with the heavy integration work moving to a scheduled
`integration.yml`. Coverage gate raises from ≥ 85% to ≥ 90%.

A late-cycle hardening pass closes the v0.4-review punch list across
the manifest, aggregator, worker / orchestration, every backend
pool, the CLI, and the plot helpers: streaming manifest scans, an
fsync cadence knob, a streaming grid generator + `Pool.imap`,
Welford's online variance, uniform `RunOutcome.failed` folding on
worker-side transport failures, lazy `gmat-sweep.cli` cold start,
and monotonic-clock `duration_s`.

### Added

- `KubernetesJobPool` (`gmat-sweep[k8s]`) — every run becomes one
  `batch/v1` Job; every Pod is a fresh interpreter; completions
  drained via a `kubernetes.watch.Watch` loop. Spec/outcome handoff
  via the same shared PVC the worker Pods mount. No changes to
  `RunSpec` or the `Pool` ABC. New
  [`recipes/kubernetes-jobpool.md`](docs/recipes/kubernetes-jobpool.md)
  walks the pod-per-run wiring (#101).
- `MPIPool` (`gmat-sweep[mpi]`) — wraps
  `mpi4py.futures.MPIPoolExecutor` and works under both
  dynamic-spawn (plain `python …`) and pre-allocated-rank
  (`mpirun -n K python -m mpi4py.futures …`) launches without any
  launcher detection on this side. CLI gains `--backend mpi` with
  `--backend-arg max_workers=N` as the dynamic-spawn knob (#102).
- `ProcessPoolExecutorPool` (stdlib, Python ≥ 3.11) — wraps
  `concurrent.futures.ProcessPoolExecutor(max_tasks_per_child=1)`,
  no `joblib` / `loky` dependency. Coexists with `LocalJoblibPool`
  rather than replacing it (#104).
- `DebugPool` — in-process, single-run dispatch so a `breakpoint()`
  reached during a run drops into the driver's debugger. Two opt-in
  flags (`subprocess_isolated="debug"` on the `Pool` ABC + an
  explicit `debug=True` on the constructor) gate the deliberate
  isolation violation (#105).
- `gmat_sweep.plotting` module behind `gmat-sweep[plot]`.
  `sweep_corner(df, params=..., metric=..., manifest=...)` is a pair
  plot of perturbed dotted-paths coloured by a per-run scalar
  metric; `sweep_heatmap(df, x=..., y=..., metric=...)` is a
  contour-grade heatmap on a 2-axis grid sweep. `matplotlib` is
  imported lazily inside each helper, so `import gmat_sweep.plotting`
  succeeds without the extra (#109).
- `gmat_sweep.sensitivity` module behind `gmat-sweep[sensitivity]`
  (`salib>=1.5`). `sobol_sample(perturb, n, *, seed=None,
  calc_second_order=True)` builds the Saltelli/Sobol design as an
  explicit-row DataFrame; `sobol_analyze(df, perturb, metric, *,
  calc_second_order=True, seed=None)` returns first/total/
  second-order indices with bootstrap confidence intervals. New
  [`docs/sensitivity.md`](docs/sensitivity.md) walks the design and
  the metric-callable contract (#121).
- `engine="polars"` kwarg across the DataFrame-returning surface —
  [`sweep`][gmat_sweep.sweep], [`monte_carlo`][gmat_sweep.monte_carlo],
  [`latin_hypercube`][gmat_sweep.latin_hypercube],
  [`monte_carlo_extend`][gmat_sweep.monte_carlo_extend],
  `Sweep.to_dataframe` / `to_ephemerides` / `to_contacts` plus a
  `Sweep.to_polars()` shortcut, and the standalone
  `lazy_multiindex` / `lazy_ephemerides` / `lazy_contacts` /
  `sweep_diff` / `mc_convergence`. Pandas remains the default; the
  `(run_id, time)` index translates to a column-pair on the polars
  side. Behind `gmat-sweep[polars]` (#120).
- [`Sweep.archive(out, *, include_logs=False) -> Path`][gmat_sweep.Sweep.archive]
  and a matching `gmat-sweep archive MANIFEST --out bundle.zip
  [--include-logs] [--allow-script-drift]` CLI subcommand pack a
  finished sweep — script, manifest, per-run Parquet outputs, a
  generated reproduce-recipe README, and a `sha256sum`-compatible
  `MANIFEST.hash` — into one deterministic `.zip` ready for Zenodo
  or a JOSS deposit. Bundled manifest `output_paths` /
  `log_path` are rewritten to bundle-relative form (#111).
- `Sweep` / [`RunOutcome`][gmat_sweep.RunOutcome] /
  [`ManifestEntry`][gmat_sweep.ManifestEntry] gain `_repr_html_`
  methods rendering as compact HTML key/value tables in Jupyter,
  instead of the default `<gmat_sweep.sweep.Sweep object at 0x…>`
  placeholder. `__repr__` is unchanged (#108).
- [`lazy_fused_reports`][gmat_sweep.lazy_fused_reports] and
  `Sweep.to_fused_reports` fuse N `ReportFile` outputs per run into
  one DataFrame with column-level `MultiIndex` keyed by
  `(report_name, column)`. Per-run merge anchors on the first name;
  subsequent reports join via `pd.merge_asof` with a user-supplied
  `tolerance`, or via inner join on `time` when `tolerance` is `None`
  (#107).
- [`sweep_summary(df, *, by="time", q=(0.05, 0.5, 0.95),
  include=("mean", "std"), dropna=True)`][gmat_sweep.sweep_summary]
  collapses a `(run_id, time)`-MultiIndexed sweep DataFrame into
  per-`by` statistics with a 2-level `("statistic", "field")`
  column MultiIndex. Default 5/50/95 quantiles plus mean/std feed
  directly into the matching plot helper; `count_ok` reports the
  per-group non-NaN run count (#114).
- [`monte_carlo_extend(manifest, script, *, n, backend, ...)`][gmat_sweep.monte_carlo_extend]
  and a sibling `latin_hypercube_extend` run only the new `n`
  samples on top of an existing sweep; the original `run_id`s'
  draws are bit-equal to the same indices of a fresh
  `monte_carlo(n=old_n + n, seed=...)`. Per-run sub-seeds derive
  from `numpy.random.SeedSequence.spawn`, so the determinism
  contract holds across extensions (#115).
- [`mc_convergence(df, metric, *, terminal_only=False)`][gmat_sweep.mc_convergence]
  returns per-prefix running mean / std / SE-of-mean across
  `run_id` prefixes — answers "did my Monte Carlo converge?" for
  a chosen metric. `metric` accepts a column name (with
  `terminal_only=True` for the per-run last value, or
  `terminal_only=False` for one running curve per `time` step) or
  a callable `(per_run_subframe) -> scalar` (#117).
- [`sweep_diff(df_a, df_b, *, on=None, how="both",
  tolerance=None)`][gmat_sweep.sweep_diff] for comparing two
  same-shape sweep DataFrames into a `(run_id, time)`- (or
  `run_id`-) indexed diff frame. Per shared numeric column emits
  `<col>__diff = b - a` and/or `<col>__rel = (b - a) / a`;
  `on="run_id"` first collapses each side to its per-run
  final-step row (#119).
- New [`docs/cookbook.md`](docs/cookbook.md) page documenting four
  patterns for turning sweep outputs into downstream-consumer
  inputs: visualisation export (CCSDS-OEM 502.0-B-3 KVN, CZML for
  Cesium), cross-tool validation (manifest + multi-indexed frame
  as harness inputs), external-tool wrapping (`out=` directory +
  `manifest.jsonl` + per-run Parquets as today's portable bundle),
  and archival deposit (Zenodo / JOSS via `Sweep.archive`) (#110).
- New runnable example notebooks rendered into the docs site:
  `08_sobol_sensitivity.ipynb` (Saltelli/Sobol design + first/
  total-order indices with 95% CIs), `09_archive_bundle.ipynb`
  (pack a sweep with `Sweep.archive()`, inspect + re-aggregate
  from the unzipped bundle), `10_extending_monte_carlo.ipynb`
  (100-run base + `monte_carlo_extend(n=200)` + bit-for-bit
  determinism assertion on the original 100). `sweep_summary`,
  `sweep_corner`, and `sweep_heatmap` cells appear inline across
  the existing sweep / Monte Carlo notebooks (#124).
- `docs/benchmarks.md` extended to report `ProcessPoolExecutorPool`
  and `MPIPool` alongside the existing `LocalJoblibPool` /
  `DaskPool` / `RayPool` rows. `KubernetesJobPool` and `DebugPool`
  are noted under an "Excluded backends" section with rationale
  (k8s = per-Pod scheduling dominates wall-clock; Debug =
  single-spec by construction) (#125).
- Smoke-canary CI cell (`smoke-canary-image`) under
  `integration.yml` runs the four-line vision-snippet sweep and a
  4-run Monte Carlo against the canonical
  `ghcr.io/astro-tools/gmat:R2025a` and `:R2026a` images on every
  push to `main`, nightly, and on demand (#112).
- New `integration.yml` workflow consolidates
  `backend-throughput`, `backend-equivalence`, `backend-k8s`, and
  `smoke-canary-image (R2025a, R2026a)`. Runs on every push to
  `main`, nightly at 07:00 UTC, and on `workflow_dispatch` (#126).
- `CITATION.cff` at the repo root so GitHub's "Cite this
  repository" UI and academic tooling resolve a canonical
  citation for the project (#116).
- [`Pool.imap(specs, *, in_flight=None)`][gmat_sweep.Pool.imap]
  ABC method with a chunked default bounded to
  `4 * max_workers`; every backend overrides the new
  `Pool.max_workers` property to expose its actual count.
  `grids.iter_grid_run_specs` + `full_factorial_size` give lazy
  grid expansion; `Sweep.__init__` accepts an `Iterable[RunSpec]`
  plus `expected_run_count`; `api.sweep(grid=...)` streams the
  cartesian product end-to-end. A 10⁵-row factorial no longer
  pins 10⁵ specs + 10⁵ futures in driver memory (#140).
- [`Manifest.iter_entries(path)`][gmat_sweep.Manifest.iter_entries]
  lazily yields parsed entries from disk; `Manifest.find_failed`
  and `Manifest.find_missing` become classmethods taking a `path`
  and consume `iter_entries` so `gmat-sweep resume` on a 10k-run
  manifest no longer pays 10k JSON parses against an entry list it
  never reads. `Manifest.load` stays eager for the consumers that
  need every entry (#137).
- [`Manifest.fsync_each: bool = True`][gmat_sweep.Manifest] plus
  `Manifest.fsync_batch: int = 50` — fsyncs amortise across
  batches when `fsync_each=False`, with `Manifest.close()`
  flushing the tail on successful sweep exit. Plumbed through
  `Sweep`, the four public API entry points, and every
  sweep-running CLI subcommand as `--fsync-each / --no-fsync-each`
  and `--fsync-batch N`. The default preserves v0.3 strict-
  per-entry durability; `KeyboardInterrupt` deliberately skips
  `close()` so the resume flow picks up the lost-tail slice (#137).
- [`Manifest.total_run_count`][gmat_sweep.Manifest.total_run_count]
  property — live total derived from the entries
  (`run_count + extension_run_count`). The header's `run_count`
  field stays frozen on disk to preserve the documented append-
  only / torn-tail tolerance (#140).
- `Manifest._migrate_header(data, from_version, path)` schema-
  migration shim — no-op for v1 → v1 today; the ladder gives a
  future v2 bump a single place to hang per-version migration
  steps (#137).
- [`ManifestCorruptError.line_number: int | None`][gmat_sweep.ManifestCorruptError]
  — 1-indexed line that failed to parse, `None` for whole-file
  failures. `Manifest.load` and `iter_entries` set it on every
  line-localised raise; the CLI's top-level error handler prints
  `path:line` so the offending line is locatable without
  re-grepping (#137).
- `KubernetesJobPool` gains
  `job_deadline_seconds: int = 3600` (validated `>= 1`) to hang-
  protect stuck Jobs — the drain loop checks each in-flight Job's
  elapsed monotonic time and folds an expired Job into a
  synthetic `RunOutcome.failed` with a background-propagation
  delete. `close()` now tracks `_in_flight_jobs: set[str]` and
  issues a background-propagation delete for each on shutdown,
  swallowing per-Job `ApiException` so one delete failure (TTL
  race, RBAC, transient network) does not orphan the rest.
  `_submit_job` wraps `create_namespaced_job` in a try/finally so
  the spec JSON written to the PVC is unlinked on every create-
  failure path (typed `ApiException`, network errors,
  `KeyboardInterrupt`) (#135).
- `sweep_corner` gains a `kind="scatter"|"hexbin"|"auto"` kwarg;
  `"auto"` (the default) flips to hexbin past 2000 runs to avoid
  silent overplot saturation (#139).

### Changed

- Overall coverage gate on the Ubuntu / Python 3.12 / R2026a cell
  raised from `>=85%` to `>=90%`, the v0.4 charter line. The four
  per-file 95% gates on `grids.py`, `distributions.py`,
  `manifest.py`, and `aggregate.py` are unchanged (#113).
- Per-PR CI trimmed from 18 to 10 cells: R2026a × 3 OS × 3 Python
  = 9, plus a single R2025a Linux/py3.12 sentinel. The full
  3 × 3 × 2 = 18 matrix expands on `main` push. Heavy backend and
  canary cells move to the new `integration.yml` (push:main +
  nightly + on-demand). No coverage loss post-merge (#126).
- Docs alignment pass: `cli.md` gains the `archive` subcommand
  (it was wired in `cli.py` and used in `cookbook.md` but missing
  from the CLI reference); `api.md` backfills the public exports
  the auto-generated reference was missing — the four
  `expand_*_to_run_specs` helpers, `latin_hypercube_samples`,
  `sweep_corner` / `sweep_heatmap`, and `DebugPool`. Version-pegged
  copy across `manifest-schema.md`, `parameter-spec.md`,
  `supported-versions.md`, and a handful of source comments is
  scrubbed (#127).
- Every backend pool now folds worker-side transport failures
  (loky / `ProcessPoolExecutor` / Dask / MPI rank / Ray worker
  death, `RayTaskError`, pickling fault, …) into a synthetic
  `RunOutcome.failed` at the drain site instead of letting
  `.result()` raise and abort the sweep. `LocalJoblibPool.close()`
  cancels parked futures *before* exiting the `Parallel` context
  (the old order let loky compute an outcome on the way out and
  silently drop it). `RayPool` uses `ray.wait(..., fetch_local=True)`
  so the per-ref `ray.get` is in-store, dropping the second network
  trip per outcome. `DaskPool` switches to
  `distributed.as_completed(..., with_results=True, raise_errors=False)`
  so the completion event, the result, and worker-side exception
  folding share one round-trip. `ProcessPoolExecutorPool` dispatches
  `run_one` directly on both `reuse_gmat_context` values —
  `max_tasks_per_child=1` already gives every task a fresh
  interpreter, so the old `run_spec_in_subprocess` hop was
  double-paying the subprocess cost (#138).
- `LocalJoblibPool` deprecates `workers=` in favour of
  `max_workers=` (kept as alias with `DeprecationWarning`). Docs,
  README, notebooks, CLI, and internal callsites switched (#138).
- `Manifest.find_failed()` / `Manifest.find_missing(...)` become
  classmethods taking a `path` argument. Pre-1.0; the only
  public doc snippet that called the old form has been updated (#137).
- `_aggregate` replaces the per-fragment `pd.DataFrame`
  accumulator + `pd.concat` with a single
  `pa.concat_tables(...).to_pandas()` — peak Python-heap allocation
  no longer scales linearly with run count (measured ~8× the
  final-frame size on a 1000-run fixture; the old path was ~67×).
  `lazy_fused_reports` replaces the O(N · runs) per-(report,
  `run_id`) boolean mask with one `groupby('run_id')` per report,
  and asserts anchor sortedness on `time` before
  `pd.merge_asof` (#136).
- `import gmat_sweep.cli` is now lazy — `pandas` / `pyarrow` /
  `tqdm` / `joblib` are not pulled until first use.
  `gmat-sweep --help` cold start drops from ~300–800 ms to ~120 ms
  on a warm cache. Pinned by
  `test_cold_start_does_not_load_heavy_dependencies` in
  `tests/test_import.py` (#139).
- `_parse_grid_value` / `_parse_backend_arg` reject `nan` / `inf`
  / `-inf`; linspace bounds get the same check. `gmat-sweep show
  --filter STATUS` rejects without `--detail` before stat'ing the
  manifest. `--backend mpi` paired with a non-default `--workers`
  exits 2 with a spelled-out MPI exception instead of silently
  ignoring the flag. `monte-carlo --help` description mirrors the
  actual determinism contract (#139).
- `_gather_param_values` prefers `manifest.parameter_spec`
  overrides (design values) over `df.columns` (measured values)
  for params in the spec, warning on disagreement.
  `sweep_heatmap` validates pivot-axis dtype before
  `np.asarray(..., dtype=float)`; a string axis raises a typed
  `SweepConfigError` pointing at `sweep_corner` (#139).
- The polars output engine is marked **experimental** in
  `docs/aggregation.md` and on the canonical `lazy_multiindex`
  engine docstring (`sweep_diff` / `mc_convergence` defer to it) (#136).

### Fixed

- `sweep()` / `monte_carlo()` / `latin_hypercube()` and
  `Sweep.from_manifest` now resolve `out=` (or the manifest path)
  to absolute before per-run directories are seeded. A relative
  `out=` was forwarded into `gmat_run.Mission.run(working_dir=...)`
  and on into the GMAT API, which resolves a relative `working_dir`
  against its installed `OUTPUT_PATH` rather than the caller's
  CWD — per-run Parquet files landed under the GMAT install
  directory and the aggregator failed to find them. Existing
  absolute-`out=` callers see no behavioural change (#112).
- Running-stats variance switches from the cumulative
  sum-of-squares identity to Welford's online recurrence — fixes
  catastrophic cancellation on km-magnitude metrics where
  `np.clip(var, 0, None)` was reporting zero std. Regression test:
  5000 samples from `N(7.1e3, 50²)` now match `np.std(ddof=1)` to
  1e-6 relative (#136).
- `sweep_summary` rejects duplicate `q` / `include` entries up
  front with `SweepConfigError` instead of producing a non-unique
  column MultiIndex. `sweep_diff(on='run_id')` sorts before
  `groupby().last()` so the terminal-row collapse is deterministic
  under shuffled input. `mc_convergence` adds an explicit
  `sort_values(['time', 'n'])` at the bottom of
  `_running_stats_per_time`, pinning the docstring's stated
  ordering contract (#136).
- `canonical_script_sha256` strips a leading UTF-8 BOM before
  line-ending normalisation, so a `.script` saved from a
  BOM-emitting Windows editor and the same script saved without
  a BOM hash equal (#137).
- `_run_subprocess` probes `--outcome` writability *before* the
  `run_one` call, so a bad output path no longer burns a
  30-minute GMAT run. On a post-run write failure the outcome is
  printed to stdout; `backends._subprocess.run_spec_in_subprocess`
  recovers it before falling back to a synthetic
  `RunOutcome.failed`. `worker.run_one`'s `mkdir` / logger /
  `FileHandler` setup now sits inside the `try` block, so a
  `FileHandler` init failure folds into `RunOutcome.failed` instead
  of escaping the worker (#140).
- `duration_s` is now sourced from `time.monotonic()`; an NTP step
  mid-run can no longer drive duration negative.
  `RunOutcome.from_dict` validates `status in {ok, failed, skipped}`;
  `RunSpec.from_dict` validates field types explicitly. Both raise
  `ValueError` naming the bad field, which
  `manifest._parse_entry_line` wraps to `ManifestCorruptError` and
  `_run_subprocess.py` maps to `_EXIT_BAD_SPEC=3` (#140).
- `Sweep._repr_html_`'s status counter buckets unknown statuses
  under `"other"` instead of raising `KeyError`. `RunSpec.to_dict`
  / `SweepSpec.to_dict` drop their per-call `dict(...)` defensive
  copies — `json.dumps` doesn't mutate (#140).
- `latin_hypercube_extend` is removed from `__all__` and
  `docs/api.md`. The function is a typed-refusal sentinel —
  raises `SweepConfigError` with a substantive explanation (LH
  extension changes stratification of every existing sample, so
  there is no bit-equal slice) — and remains importable from
  `gmat_sweep.api` for generic wrappers that want the typed
  error (#140).
- Doc drift swept out by walking every page against the current
  source and re-executing all ten example notebooks: two
  `Sweep.from_manifest(Path(...))` examples in
  [`docs/cookbook.md`](docs/cookbook.md) were missing the required
  `script_path` positional and `backend=` keyword (would raise
  `TypeError` as written); rewritten with `LocalJoblibPool`.
  `docs/manifest-schema.md` described `duration_s` as
  `(ended_at - started_at).total_seconds()`; rewritten to match
  the actual `time.monotonic` bookend pair.
  `docs/recipes/slurm.md` and `docs/recipes/ray-autoscaling.md`
  referenced a non-existent `_worker_entrypoint`; rewritten to
  point at `gmat_sweep.worker.run_one` (reuse path) and
  `gmat_sweep.backends._subprocess.run_spec_in_subprocess`
  (fresh-bootstrap path). Pool-inventory enumerations across
  `docs/index.md`, `docs/getting-started.md`, `docs/faq.md`,
  `docs/recipes/index.md`, `docs/backends.md`, and `docs/cli.md`
  refreshed to the seven shipped pools; the missing `mpi` entry in
  `cli.md`'s "Choosing a backend" table is back, and the deprecated
  `LocalJoblibPool(workers=...)` snippet swapped for
  `max_workers=...` (#141).

[0.4.0]: https://github.com/astro-tools/gmat-sweep/releases/tag/v0.4.0

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
