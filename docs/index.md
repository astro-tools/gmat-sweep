# gmat-sweep

Run parameter sweeps and Monte Carlo dispersions over [GMAT](https://software.nasa.gov/software/GSC-17177-1)
missions in parallel from Python.

!!! warning "Pre-alpha"
    The public surface is not yet usable for production work. The
    [v0.1 milestone](https://github.com/astro-tools/gmat-sweep/milestone/1)
    tracks the work needed to ship the first PyPI release.

## What it does

`gmat-sweep` takes one GMAT `.script` file and runs it many times under
different parameter overrides, in parallel subprocesses, then aggregates the
per-run reports into a single multi-indexed `pandas.DataFrame`. The single
public entry point is [`sweep()`][gmat_sweep.sweep]: pass it a script, a
parameter grid, and (optionally) a worker count.

```python
from gmat_sweep import sweep

df = sweep(
    "mission.script",
    grid={
        "Sat.SMA": [7000, 7100, 7200],
        "Sat.DryMass": [100, 200],
    },
)
```

That call runs the cartesian product (six runs in this example), one fresh
GMAT subprocess per run, and returns a `(run_id, time)`-MultiIndexed
DataFrame with one row per (run, time-step) pair plus a `__status` column
flagging `ok` / `failed` / `skipped` runs.

## Where to go next

- **[Getting started](getting-started.md)** — install and the four-line
  vision snippet.
- **[Parameter spec](parameter-spec.md)** — how grids, dotted-path overrides,
  and the CLI's mini-grammar work.
- **[Manifest schema](manifest-schema.md)** — the JSON Lines manifest each
  sweep writes alongside its outputs.
- **[Supported versions](supported-versions.md)** — GMAT × Python × OS matrix.
- **[FAQ](faq.md)** — subprocess isolation, the `gmat-run` dependency, and
  where to get GMAT.
- **[API reference](api.md)** — auto-generated from docstrings.

## Scope

`gmat-sweep` is deliberately narrow: run an existing `.script` N times under
N different overrides via [`gmat-run`](https://github.com/astro-tools/gmat-run),
in parallel, and aggregate the results. It is not an optimiser, not a
script-builder, and does not own GMAT installation.

For the full scope rationale and pointers to neighbouring tools see
[`CONTRIBUTING.md`](https://github.com/astro-tools/gmat-sweep/blob/main/CONTRIBUTING.md#scope-discipline).
