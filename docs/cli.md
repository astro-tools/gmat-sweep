# CLI reference

`gmat-sweep` ships a `gmat-sweep` console script. Every Python entry point on
the public surface has a matching subcommand, so you can drive a sweep,
inspect a manifest, or resume a partial run without writing Python.

```text
gmat-sweep <subcommand> [options] [SCRIPT]
```

The subcommands:

- [`run`](#run) — full-factorial grid sweep.
- [`monte-carlo`](#monte-carlo) — stochastic dispersion sweep.
- [`latin-hypercube`](#latin-hypercube) — Latin hypercube sweep.
- [`explicit`](#explicit) — explicit-row sweep from a CSV or Parquet design.
- [`resume`](#resume) — re-run only the failed and missing entries from a
  prior manifest.
- [`extend`](#extend) — append more bit-deterministic Monte Carlo runs to an
  existing sweep.
- [`show`](#show) — print a one-line summary of a manifest.
- [`archive`](#archive) — pack a finished sweep into a portable `.zip` for
  archival deposit.

`gmat-sweep --help` lists them. Each subcommand has its own `--help`.

## Common options

The sweep-running subcommands (`run`, `monte-carlo`, `latin-hypercube`,
`explicit`) share three flags:

| Flag         | Default | Meaning                                                                      |
|--------------|---------|------------------------------------------------------------------------------|
| `--workers N`| `-1`    | Number of subprocess workers. `-1` uses every available core. Not accepted on `--backend mpi` (the launcher fixes the rank count); passing it there exits `2`. |
| `--out PATH` | —       | Required. Output directory for per-run artefacts and `manifest.jsonl`.       |
| `SCRIPT`     | —       | Required positional. Path to the GMAT `.script` every run loads.             |

The post-hoc subcommands (`extend`, `resume`, `archive`) take the script as
a `--script PATH` flag instead — the manifest is the primary positional
input on those, and the script is a parameter pointing at the previously-
loaded mission file.

Each of those four subcommands writes a `manifest.jsonl` under `--out` and
prints a one-line summary to stdout when the sweep finishes:

```text
N runs (A ok[, B failed][, C skipped]) in T.TT s — output: PATH
```

## Choosing a backend

The four sweep-running subcommands and `resume` accept `--backend`. `show`
does not — it never runs anything.

| Value           | Pool                                   | Extras                           |
|-----------------|----------------------------------------|----------------------------------|
| `local` (default) | [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] over loky workers | none |
| `dask`          | [`DaskPool`][gmat_sweep.backends.DaskPool] over a `LocalCluster` | `pip install gmat-sweep[dask]` |
| `ray`           | [`RayPool`][gmat_sweep.backends.RayPool] over a local Ray runtime | `pip install gmat-sweep[ray]` |
| `mpi`           | [`MPIPool`][gmat_sweep.backends.MPIPool] over `mpi4py.futures` | `pip install gmat-sweep[mpi]` (plus a system MPI install) |

`--workers N` maps onto each backend in the natural way: `LocalJoblibPool`
takes it as `max_workers`, `DaskPool` as `n_workers`, `RayPool` as `num_cpus`.
The default `-1` means "let the pool pick" (every available core for the
local pool, `os.cpu_count()` for Dask, Ray's own auto-detect for Ray).
`--workers` is not supported on `--backend mpi` — the rank count is fixed
by the MPI launcher (`mpirun -n K`) or by `--backend-arg max_workers=N`
under dynamic spawn; passing `--workers` there exits `2`.

`--backend-arg KEY=VALUE` is an escape hatch for less-common pool
constructor kwargs. It is repeatable, values are coerced int → float → str,
and the parsed pairs are forwarded as `**kwargs` to the chosen pool.
Examples:

```bash
gmat-sweep run --backend dask \
    --backend-arg threads_per_worker=2 \
    --grid Sat.SMA=7000:7200:3 \
    --out ./sweep mission.script

gmat-sweep run --backend ray \
    --backend-arg address=ray://head:10001 \
    --grid Sat.SMA=7000:7200:3 \
    --out ./sweep mission.script
```

`--backend-arg` is rejected with `--backend local` (the local pool has no
extra kwargs to forward). Missing extras (`[dask]` / `[ray]` / `[mpi]` not
installed) exit with code `4` and a "pip install gmat-sweep[…]" message
on stderr. Unknown kwargs surface the same way — they reach the pool
constructor and are rejected there.

## Manifest fsync cadence

Every sweep-running subcommand (`run`, `monte-carlo`, `latin-hypercube`,
`explicit`, `extend`, `resume`) accepts two flags that control how often
the manifest is fsynced:

| Flag | Default | Meaning |
|------|---------|---------|
| `--fsync-each` / `--no-fsync-each` | `--fsync-each` | When set, the manifest is fsynced after every appended entry — strict per-run durability. Pass `--no-fsync-each` to amortise the fsync cost across batches. |
| `--fsync-batch N` | `50` | Fsync interval (in entries) when `--no-fsync-each` is set. Ignored otherwise. |

`--no-fsync-each` is a measurable speedup for sub-second runs at large
counts (Monte Carlo dispersion sweeps, parameter scans where each run
takes <100 ms) where the per-entry fsync would otherwise dominate the
driver thread. The tradeoff: a host crash between fsync boundaries can
leave up to `--fsync-batch - 1` recently-completed entries missing from
the on-disk manifest. The per-run Parquet outputs and the script hash
are unaffected, and `gmat-sweep resume` re-runs only the missing slice.
See [Manifest schema § Fsync cadence and durability](manifest-schema.md#fsync-cadence-and-durability)
for the full contract.

## Exit codes

| Code | Meaning                                                                        |
|------|--------------------------------------------------------------------------------|
| `0`  | success                                                                        |
| `1`  | any other [`GmatSweepError`][gmat_sweep.GmatSweepError]                        |
| `2`  | [`SweepConfigError`][gmat_sweep.SweepConfigError] or argparse usage error      |
| `3`  | [`ManifestCorruptError`][gmat_sweep.ManifestCorruptError] or missing manifest  |
| `4`  | [`BackendError`][gmat_sweep.BackendError]                                      |

## `run`

Run a full-factorial grid sweep. Each `--grid` flag adds one axis; multiple
flags combine into the cartesian product.

```bash
gmat-sweep run \
    --grid 'Sat.SMA=7000:8000:5' \
    --grid 'Sat.DryMass=100,200,300' \
    --workers 4 \
    --out ./sweep-out \
    mission.script
```

`--grid SPEC` accepts two forms:

- `name=lo:hi:count` — `count` evenly-spaced points from `lo` to `hi`
  inclusive (numpy linspace). `count` must be ≥ 2.
- `name=v1,v2,v3` — explicit comma-separated values, each coerced via
  int → float → str fallback.

Repeated `--grid` flags for the same axis name exit with code `2`.

## `monte-carlo`

Run `n` independent stochastic samples by sampling each `--perturb`
parameter from its own distribution. With `--seed` set, two runs at the
same `(--perturb, --n, --seed)` produce bit-equal DataFrames; without
`--seed` the draws fall back to OS entropy and are not reproducible.

```bash
gmat-sweep monte-carlo \
    --n 1000 \
    --perturb 'Sat.SMA=normal:7100:50' \
    --perturb 'Sat.INC=uniform:0:90' \
    --seed 42 \
    --workers 8 \
    --out ./mc-out \
    mission.script
```

`--perturb SPEC` takes one of three shorthands:

| Form                         | scipy equivalent                            |
|------------------------------|---------------------------------------------|
| `name=normal:mu:sigma`       | `scipy.stats.norm(loc=mu, scale=sigma)`     |
| `name=uniform:lo:hi`         | `scipy.stats.uniform(loc=lo, scale=hi-lo)`  |
| `name=lognormal:mu:sigma`    | `scipy.stats.lognorm(s=sigma, scale=exp(mu))`|

Per-parameter sub-seeds are derived from the parameter's *name*, so adding a
`--perturb` flag to an existing sweep does not change the draws of any other
parameter at any `run_id` — see [Monte Carlo](monte-carlo.md) for the full
determinism contract.

`--seed` is optional; without it, the draw set falls back to OS entropy and
is not reproducible. An unknown distribution tag (e.g. `triangular`) exits
with code `2`.

## `latin-hypercube`

Draw `n` Latin hypercube points stratified across each `--perturb` axis,
mapping each axis through the user's distribution. Same `--perturb` syntax as
`monte-carlo`.

```bash
gmat-sweep latin-hypercube \
    --n 100 \
    --perturb 'Sat.SMA=normal:7100:50' \
    --seed 42 \
    --workers 4 \
    --out ./lhs-out \
    mission.script
```

Latin hypercube sampling typically beats plain Monte Carlo when `n` is small
relative to the problem's dimensionality, because the per-axis coverage is
enforced by construction.

## `explicit`

Run one mission per row of a pre-built sample design. Column names are
[dotted-path field names](parameter-spec.md#dotted-path-keys); the row index
becomes `run_id`.

```bash
gmat-sweep explicit \
    --samples ./samples.csv \
    --workers 4 \
    --out ./explicit-out \
    mission.script
```

`--samples PATH` accepts `.csv` (loaded via `pandas.read_csv`) and `.parquet`
(loaded via `pandas.read_parquet`). Other suffixes exit with code `2`. The
loaded DataFrame must use a default `RangeIndex(start=0)`, have unique string
column names, and contain no all-NaN columns — any violation surfaces as a
[`SweepConfigError`][gmat_sweep.SweepConfigError] (exit code `2`) before any
runs start.

Use `explicit` when you have already built a sampling design (Halton, Sobol,
custom optimisation results) and want to hand it in directly.

## `resume`

Re-run only the failed and never-recorded entries from an existing
`manifest.jsonl`. Successful runs' Parquet files are reused from disk.

```bash
gmat-sweep resume ./sweep-out/manifest.jsonl \
    --script mission.script \
    --workers 4
```

Required positional: `MANIFEST` — the existing `manifest.jsonl`.

Required flag: `--script PATH` — the same GMAT `.script` the original sweep
loaded. Its canonical SHA-256 must equal the manifest's `script_sha256`; see
[Resume § Script drift](resume.md#script-drift) for the full contract. Add
`--allow-script-drift` to proceed past a hash mismatch (emits a
`RuntimeWarning`).

A missing `MANIFEST` exits with code `3`; a missing `--script` or a hash
mismatch exits with code `2`.

## `extend`

Append `N` more bit-deterministic Monte Carlo runs to an existing sweep.
The base manifest's `seed` and `perturb` mapping are reused so the new
draws are bit-equal to the same indices of a fresh
`monte_carlo(n=old_n + N)` call.

```bash
gmat-sweep extend ./mc-out/manifest.jsonl \
    --n 1000 \
    --script mission.script \
    --workers 8
```

Required positional: `MANIFEST` — an existing Monte Carlo
`manifest.jsonl`. Grid, explicit-row, and Latin hypercube manifests
exit with code `2` (their stochastic semantics don't admit clean
extension).

Required flags:

- `--n N` — number of additional stochastic runs to append, ≥ 1.
- `--script PATH` — the same GMAT `.script` the original sweep loaded.
  Same hash-drift contract as `resume`; add `--allow-script-drift` to
  proceed past a mismatch.

`extend` refuses if the base sweep has any `failed` or missing runs in
its original `[0, n)` range — the underlying error names them and
points at `gmat-sweep resume`. Run `resume` first, then `extend`.

A missing `MANIFEST` exits with code `3`; a non-Monte-Carlo manifest,
an incomplete base sweep, a missing `--script`, or a hash mismatch
exits with code `2`. See
[Monte Carlo § Extending an existing sweep](monte-carlo.md#extending-an-existing-sweep)
for the full determinism contract.

## `show`

Inspect a manifest produced by any of the sweep-running subcommands. Three
modes:

- default — one-line summary.
- `--detail` — per-run table sorted with `failed` first, then `skipped`, then
  `ok`, plus the same one-line summary at the bottom.
- `--run N` — full record for `run_id=N`: header fields, override dict, and
  the unsuppressed `stderr`.

`--detail` and `--run` are mutually exclusive.

```bash
gmat-sweep show ./sweep-out/manifest.jsonl
```

```text
5 runs (4 ok, 1 failed) in 53.41 s — output: ./sweep-out
```

### `--detail`

```bash
gmat-sweep show --detail ./sweep-out/manifest.jsonl
```

```text
run_id  status   duration_s  stderr_summary                  log_path
1       failed   0.21        ValueError: Sat.SMA out of...   ./sweep-out/run-1/worker.log
0       ok       12.43       —                               ./sweep-out/run-0/worker.log
2       ok       11.97       —                               ./sweep-out/run-2/worker.log
3       ok       14.02       —                               ./sweep-out/run-3/worker.log
4       ok       14.78       —                               ./sweep-out/run-4/worker.log
5 runs (4 ok, 1 failed) in 53.41 s — output: ./sweep-out
```

`stderr_summary` is the first line of the run's captured `stderr`, truncated
to 60 characters with a `...` ellipsis. `ok` rows show `—` (no captured
`stderr`).

`--filter STATUS` narrows the table to one of `ok`, `failed`, `skipped`. The
trailing summary line still reflects the full manifest. `--filter` requires
`--detail`.

### `--run N`

```bash
gmat-sweep show --run 1 ./sweep-out/manifest.jsonl
```

Prints `run_id=1`'s full record (status, duration, timestamps, log path,
override dict, full unsuppressed `stderr`). Exit code `0`. If `N` is not in
the manifest, exits with code `3` and a `gmat-sweep: run_id N not found in
manifest` message on stderr.

### Exit codes

A missing or unparseable manifest, or a `--run N` for an `N` not in the
manifest, exits with code `3`. An unparseable manifest's error message
includes the file path and the 1-indexed line number that failed to
parse — e.g. `gmat-sweep: manifest entry on line 42 is malformed: ...
(./sweep/manifest.jsonl:42)` — so the offending line can be located
without re-parsing the file by hand. Whole-file failures (empty file,
header-line truncation) print just the path.

## `archive`

Pack a finished sweep — script, manifest, per-run Parquet outputs — into a
single self-describing `.zip` suitable for archival deposit (Zenodo, JOSS
supplementary material) or internal handoff.

```bash
gmat-sweep archive ./sweep-out/manifest.jsonl \
    --script mission.script \
    --out ./sma-scan-2026q2.zip
```

Required positional: `MANIFEST` — the existing `manifest.jsonl`.

Required flags:

- `--script PATH` — the same GMAT `.script` the sweep loaded. Its canonical
  SHA-256 must equal the manifest's `script_sha256`; same hash-drift contract
  as `resume` and `extend`. Add `--allow-script-drift` to proceed past a
  mismatch — the bundle still records the manifest's original hash so a
  downstream verifier can spot the drift.
- `--out PATH` — destination `.zip` path. Parent directories are created on
  demand.

`--include-logs` bundles every per-run `worker.log` and keeps the manifest's
`log_path` field pointing at it (rewritten to bundle-relative form). The
default drops the logs and sets `log_path` to `null` in the bundled manifest,
keeping the archive small.

The bundled manifest's `output_paths` are rewritten to be bundle-relative
(`runs/run-<id>/<basename>`), and the bundle is byte-deterministic — two
archives of the same manifest produce identical bytes. See
[Cookbook § Pattern 4 — Archival deposit](cookbook.md#pattern-4-archival-deposit-zenodo-joss)
for the full layout and the reproduce-from-bundle recipe.

A missing `MANIFEST` exits with code `3`; a missing `--script`, or a hash
mismatch without `--allow-script-drift`, exits with code `2`.
