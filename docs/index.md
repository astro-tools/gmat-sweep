# gmat-sweep

Run parameter sweeps and Monte Carlo dispersions over [GMAT](https://software.nasa.gov/software/GSC-17177-1)
missions in parallel from Python.

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

For stochastic studies — launch dispersions, sensitivity analyses,
margin sweeps — reach for [`monte_carlo()`][gmat_sweep.monte_carlo] or
[`latin_hypercube()`][gmat_sweep.latin_hypercube]. They take the same
mission script and a `perturb` mapping of named distributions, and
return the same DataFrame shape:

```python
from gmat_sweep import monte_carlo

df = monte_carlo(
    "mission.script",
    n=500,
    perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
    seed=42,
)
```

See [Monte Carlo](monte-carlo.md) for the determinism and
order-independence contracts.

## Where to go next

- **[Getting started](getting-started.md)** — install and the four-line
  vision snippet.
- **[Parameter spec](parameter-spec.md)** — how grids, dotted-path overrides,
  and the CLI's mini-grammar work.
- **[Monte Carlo](monte-carlo.md)** — stochastic dispersion sweeps with
  named distributions and a determinism contract.
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
