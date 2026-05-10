# Sensitivity analysis

Sobol sensitivity indices answer the canonical *which input drives the
output* question for a stochastic sweep. `gmat_sweep.sensitivity`
([`sobol_sample`][gmat_sweep.sobol_sample] + [`sobol_analyze`][gmat_sweep.sobol_analyze])
wraps [SALib](https://salib.readthedocs.io/) so you can produce a
Saltelli sample design, run the sweep through the existing `sweep(samples=...)`
path, and read the indices off without assembling SALib's `problem`
dict by hand.

## Install

The dependency is gated behind an optional extra so the default install
stays slim:

```bash
pip install gmat-sweep[sensitivity]
```

## Workflow

The three steps are: build the sample design, run the sweep, analyse.

```python
from gmat_sweep import sweep, sobol_sample, sobol_analyze

perturb = {
    "CoastTime":    ("uniform", 0.0, 1200.0),
    "Inj.Element1": ("normal", 1.0, 0.05),
    "Inj.Element2": ("normal", 0.0, 0.05),
    "Inj.Element3": ("normal", 0.0, 0.05),
}

# 1. Saltelli/Sobol sample design — n*(2D+2) rows for D=4, so n=128 → 1280 runs.
samples = sobol_sample(perturb, n=128, seed=42)

# 2. Run the sweep through the explicit-row entry point.
df = sweep("injection_dispersion.script", samples=samples)

# 3. Sobol indices on the metric of interest.
indices = sobol_analyze(df, perturb, metric="Sat.X", seed=42)
print(indices)
```

`sobol_analyze` returns a tidy long DataFrame with columns
`kind` / `param_a` / `param_b` / `value` / `conf`:

| kind | param_a       | param_b       | value | conf |
|------|---------------|---------------|------:|-----:|
| S1   | CoastTime     | NaN           | 0.41  | 0.03 |
| S1   | Inj.Element1  | NaN           | 0.18  | 0.02 |
| ...  |               |               |       |      |
| ST   | CoastTime     | NaN           | 0.55  | 0.04 |
| ...  |               |               |       |      |
| S2   | CoastTime     | Inj.Element1  | 0.07  | 0.05 |

`kind="S1"` is the first-order index (parameter alone), `"ST"` is the
total-order index (parameter plus all its interactions), and `"S2"` is
the pairwise second-order index. `conf` is SALib's 95 % bootstrap
confidence half-width — interpret index values as `value ± conf`.

## Sample size

The Saltelli design generates `n * (2*D + 2)` runs at
`calc_second_order=True` and `n * (D + 2)` runs at
`calc_second_order=False`, where `D = len(perturb)`. Pick `n` as a
power of two; SALib's authors recommend `n ≥ 1024` for stable estimates
on most engineering problems.

If the second-order pairwise indices aren't part of the question,
trade them away to halve the run count:

```python
samples = sobol_sample(perturb, n=128, seed=42, calc_second_order=False)
df = sweep("...", samples=samples)
indices = sobol_analyze(df, perturb, metric="Sat.X", seed=42, calc_second_order=False)
```

The flag must match between the two calls — `sobol_analyze` doesn't
re-derive it from the row count.

## Reducing per-run output to a scalar

`metric` collapses the `(run_id, time)`-MultiIndexed sweep result to one
scalar per run:

* `metric="Sat.X"` — convenience for *value of `Sat.X` at each run's
  final time-step* (the typical end-of-mission state).
* `metric=callable` — receives the full DataFrame, returns a
  `pd.Series` indexed by `run_id`. Use this for derived quantities
  like miss distance, total Δv, or any quantile across the run's
  trajectory.

```python
import numpy as np

def terminal_radius(df):
    last = df.groupby(level="run_id").tail(1)
    return np.sqrt(last["Sat.X"]**2 + last["Sat.Y"]**2 + last["Sat.Z"]**2)

indices = sobol_analyze(df, perturb, metric=terminal_radius, seed=42)
```

## Distribution shapes

`sobol_sample` accepts the same `perturb` mapping as
[`monte_carlo`](monte-carlo.md): the three shorthand tuples
(`("normal", mu, sigma)`, `("uniform", lo, hi)`, `("lognormal", mu, sigma)`)
and any pre-frozen `scipy.stats.rv_frozen`. The unit-cube design from
SALib is lifted into each parameter's marginal via the distribution's
`ppf` — generalising past SALib's own `dists` knob to anything
`scipy.stats` can freeze.

## Failure modes

`sobol_analyze` rejects a sweep result that contains any `__status != "ok"`
rows: SALib cannot ingest the NaN-padded values that
failed/skipped runs leave behind. Filter explicitly first:

```python
df_ok = df[df["__status"] == "ok"]
indices = sobol_analyze(df_ok, perturb, metric="Sat.X", seed=42)
```

Be aware that filtering breaks the Saltelli design's row balance — if
runs failed in the middle, the indices on the survivors are no longer
unbiased estimates. For a clean answer, re-launch the failed rows
(`gmat-sweep resume`) before analysing.

## Determinism

Two `sobol_sample(..., seed=K)` calls produce bit-equal DataFrames at
the same `(perturb, n, seed, calc_second_order)`. Two
`sobol_analyze(..., seed=K)` calls produce bit-equal indices and
bootstrap confidence intervals against the same `Y` vector. Saltelli
seeding is independent of [`monte_carlo`](monte-carlo.md)'s
per-run-name seeding contract — they are separate samplers.

## What SALib methods are wrapped

`gmat_sweep.sensitivity` ships Sobol only. SALib also exposes Morris,
FAST, RBD-FAST, and DGSM samplers and analyses; if your work needs one
of those, the same recipe (`sobol_sample`'s unit-cube + `ppf` lift,
applied to the SALib sampler, then handed to `sweep(samples=...)`)
generalises in three lines — open an issue if you'd like the wrappers
upstreamed.
