# FAQ

## Why does each run go in its own subprocess?

GMAT is not safe to reuse for multiple `.script` loads inside a single
Python process. Two reasons stack:

1. **Bootstrap cost is paid once per process anyway.** The first
   `import gmat_run` plus `Mission.load(...)` in any process pays the
   `gmatpy` SWIG bring-up. Subsequent calls in the same process are cheap,
   but you only get that benefit if it is safe to *use* the same process for
   the next run — and it is not (point 2).
2. **GMAT relies on process-global singletons that do not survive a second
   `Mission.load`.** Reusing a Python process across runs that load
   different scripts (or even the same script with different overrides)
   produces inconsistent state — sometimes silently wrong, sometimes a hard
   crash deep inside the engine.

The [`Pool`][gmat_sweep.Pool] ABC enforces this at class-definition time
via its `subprocess_isolated` class attribute: every `Pool` subclass must
implement both bootstrap-amortisation modes correctly, and any subclass
that tries to opt out is rejected when its module is imported, before a
sweep can start.

The two modes are exposed via the `reuse_gmat_context` keyword on every
pool constructor:

- **`reuse_gmat_context=True` — the default.** A worker process imports
  `gmat_run` once and reuses the loaded state across many tasks. Bootstrap
  cost is paid once per worker, then amortised. Safe **only when every
  task dispatched through the pool loads the same script** — GMAT relies
  on process-global singletons that cannot be reused across runs that
  load different scripts. This is the right choice for the common case
  (one mission, many parameter combinations) and is what every notebook
  and recipe in the docs assumes.
- **`reuse_gmat_context=False` — the isolation path.** Every task spawns
  a fresh Python interpreter that bootstraps `gmatpy` from scratch via
  `python -m gmat_sweep._run_subprocess` (an internal CLI module that
  runs one [`RunSpec`][gmat_sweep.RunSpec] and emits the resulting
  [`RunOutcome`][gmat_sweep.RunOutcome] as JSON). Slower per task but
  supports arbitrary heterogeneous scripts on a single pool. Reach for
  it when you compose one Dask or Ray pool across calls that load
  different `.script` files.

The contract is uniform across every shipped pool — `LocalJoblibPool`,
`ProcessPoolExecutorPool`, `DaskPool`, `RayPool`, `KubernetesJobPool`,
`MPIPool`, and `DebugPool` — and the `Pool` ABC enforces it on any
third-party subclass. Under the default, a sweep of N runs with W
workers pays roughly W bootstraps total, not 1 and not N — for a small
sweep that's overhead worth knowing about; for a meaningfully large
sweep it is a rounding error. `ProcessPoolExecutorPool` is the
exception by construction: `max_tasks_per_child=1` gives every task a
fresh interpreter, so it pays N bootstraps regardless of mode.

## Why does `gmat-sweep` depend on `gmat-run`?

`gmat-sweep` is the parallel orchestrator. `gmat-run` is the single-run
primitive — it owns GMAT install discovery, the `gmatpy` bootstrap, the
`Mission` API, the dotted-path setter, the `ReportFile` parser, and the
`Results` object. `gmat-sweep` would have to re-implement all of that just
to launch a single run, so it does not — every worker subprocess calls into `gmat_run.Mission`.

That layering also keeps `gmat-sweep`'s scope narrow. The package's job is:
expand a grid into [`RunSpec`][gmat_sweep.RunSpec]s, fan them out to a
[`Pool`][gmat_sweep.Pool], record outcomes in the
[manifest](manifest-schema.md), and aggregate the per-run Parquet outputs
into one DataFrame. Anything below the per-run line — script parsing,
field setters, GMAT engine invocation — is
[`gmat-run`'s](https://astro-tools.github.io/gmat-run/) problem.

## Where do I get GMAT?

GMAT is distributed by NASA's General Mission Analysis Tool project on
SourceForge:
[https://sourceforge.net/projects/gmat/files/GMAT/](https://sourceforge.net/projects/gmat/files/GMAT/).
R2026a is the primary development target; R2025a is also supported. See
[supported versions](supported-versions.md) for the full matrix.

Once unpacked, [`gmat-run`'s install
guide](https://astro-tools.github.io/gmat-run/install-gmat/) covers
discovery — typically `gmat-run` finds the install via `$GMAT_ROOT`, a
conventional install path, or a same-prefix sibling directory.

If you are wiring a CI job, use
[`astro-tools/setup-gmat`](https://github.com/astro-tools/setup-gmat) — it
caches the install across runs and exposes the same discovery hooks
`gmat-run` uses locally.
