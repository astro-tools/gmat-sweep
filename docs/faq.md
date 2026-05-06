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
via its `subprocess_isolated` class attribute. Any `Pool` subclass that
tries to set it to anything other than `True` is rejected when its module
is imported, before a sweep can start. Each worker subprocess imports
`gmat_run` once on its first run and reuses that import for the rest of the
runs that worker handles.

Backends that reuse worker processes honour the contract via
`python -m gmat_sweep._run_subprocess` — an internal CLI module that
runs one [`RunSpec`][gmat_sweep.RunSpec] in a freshly-spawned interpreter
and emits the resulting [`RunOutcome`][gmat_sweep.RunOutcome] as JSON.
Each task body in such a backend invokes the entrypoint via
`subprocess.run`, so even when the surrounding worker process is
long-lived, the GMAT run itself gets a fresh interpreter. The entrypoint
is internal infrastructure; callers go through a `Pool`.

The trade-off is the bootstrap cost amortising over batches — for a sweep
of N runs with W workers, you pay roughly W bootstraps total, not 1 and not
N. For a small sweep that's overhead worth knowing about; for a meaningfully
large sweep it is a rounding error.

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
