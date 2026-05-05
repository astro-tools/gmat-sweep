# Monte Carlo

[`monte_carlo()`][gmat_sweep.monte_carlo] runs a stochastic dispersion sweep
over a GMAT mission: each parameter is drawn from its own distribution, and
the same `(n, perturb, seed)` reproduces the same draws bit-for-bit on any
machine.

```python
from gmat_sweep import monte_carlo

df = monte_carlo(
    "mission.script",
    n=100,
    perturb={
        "Sat.SMA": ("normal", 7100.0, 50.0),
        "Sat.INC": ("uniform", 0.0, 90.0),
    },
    seed=42,
)
```

That call spawns 100 runs, samples one value per `perturb` entry per run,
and returns a `(run_id, time)`-MultiIndexed DataFrame containing every
run's `ReportFile` rows plus a `__status` column flagging
`ok` / `failed` / `skipped`.

## Distribution specs

The `perturb` mapping accepts the same shapes as the rest of the
stochastic-sweep surface — see
[Parameter spec → Stochastic specs](parameter-spec.md#stochastic-specs)
for the full table. The shorthands cover the cases that come up day-to-day
in dispersion analyses; reach for a pre-frozen
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) rv
when you need a different shape:

```python
import math
from scipy import stats

monte_carlo(
    "mission.script",
    n=500,
    perturb={
        "Sat.SMA":      ("normal", 7100.0, 50.0),
        "Sat.DryMass":  ("lognormal", math.log(100.0), 0.05),
        "Sat.RAAN":     ("uniform", 0.0, 360.0),
        "Sat.INC":      stats.truncnorm(a=-2, b=2, loc=0.0, scale=10.0),
    },
    seed=42,
)
```

## Determinism contract

Two `monte_carlo(..., seed=42)` calls with the same `n`, the same
`perturb`, and the same script produce DataFrames whose recorded per-run
overrides are identical at every `run_id`. Two calls at `seed=42` and
`seed=43` produce different draws.

The contract is process-independent — the per-run sub-seeds come from
[`numpy.random.SeedSequence`][gmat_sweep.distributions.derive_run_seeds],
so a fresh Python process given the same inputs reconstructs the same
draws.

`seed=None` falls back to OS entropy and is **not** reproducible.

## Order independence

Adding a perturbed parameter to an existing `perturb` dict does not change
the draws of any other parameter at any `run_id` — regardless of where
the new parameter falls in lexicographic order. Per-parameter sub-seeds
are derived from the parameter *name*, not its position in the mapping:

```python
# First sweep: one perturbed axis.
df_one = monte_carlo("mission.script", n=20, seed=42, perturb={
    "Sat.SMA": ("normal", 7100.0, 50.0),
})

# Add a second axis whose name sorts BEFORE Sat.SMA.
df_two = monte_carlo("mission.script", n=20, seed=42, perturb={
    "Aaa.X":   ("uniform", 0.0, 1.0),
    "Sat.SMA": ("normal", 7100.0, 50.0),
})

# Sat.SMA's draw at every run_id is unchanged.
```

This matters when an analysis grows: extending a 1-D `perturb` to a 4-D
one mid-investigation should not invalidate the 1-D results.

## Worked example: launch dispersion

A typical injection-error analysis perturbs the post-burn state vector
around its nominal value and asks how the final orbit's miss distance
distributes:

```python
import math
import pandas as pd

from gmat_sweep import monte_carlo

df = monte_carlo(
    "transfer_porkchop.script",
    n=500,
    perturb={
        "Sat.SMA":     ("normal", 24500.0, 25.0),    # ±25 km 1-σ
        "Sat.ECC":     ("normal", 0.7300, 1e-4),     # ±0.0001 1-σ
        "Sat.INC":     ("normal", 28.5, 0.05),       # ±0.05° 1-σ
        "Sat.RAAN":    ("normal", 0.0, 0.5),         # ±0.5° 1-σ
        "Sat.DryMass": ("lognormal", math.log(1200.0), 0.02),
    },
    seed=20260504,
    workers=8,
    out="./launch-dispersion",
)

# Final-step rows of every run, joined with status.
final = df.groupby("run_id").tail(1).reset_index()
print(final[["run_id", "MissDistance", "__status"]].describe())
```

The manifest at `./launch-dispersion/manifest.jsonl` records the seed and
the `perturb` dict, so the analysis is reproducible from disk alone — see
[Manifest schema](manifest-schema.md) for the full header layout.

## Failed runs

A run that raises during override application or mission execution lands
as a single NaN-filled row with `__status="failed"` and the captured
worker stderr in the manifest entry — same contract as
[`sweep()`][gmat_sweep.sweep]. A bad draw never aborts the sweep:

```python
ok = df[df["__status"] == "ok"]
failed = df[df["__status"] == "failed"]
```

## See also

- [Parameter spec → Stochastic specs](parameter-spec.md#stochastic-specs)
  — distribution shorthand surface and validation rules.
- [API reference: `monte_carlo`](api.md#gmat_sweep.monte_carlo)
- [API reference: `latin_hypercube`](api.md#gmat_sweep.latin_hypercube)
  — stratified-sampling alternative for low-`n`, higher-dimensional
  studies.
