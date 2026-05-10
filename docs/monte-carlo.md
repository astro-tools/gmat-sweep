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
`numpy.random.SeedSequence`, so a fresh Python process given the same
inputs reconstructs the same draws.

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

## Extending an existing sweep

A 1000-run dispersion that turned out to need 2000 doesn't have to start
over. [`monte_carlo_extend()`][gmat_sweep.monte_carlo_extend] runs only
the new 1000 against the original sweep's manifest and returns the full
2000-run aggregated DataFrame:

```python
from gmat_sweep import monte_carlo, monte_carlo_extend

# Original sweep.
df_1000 = monte_carlo(
    "mission.script",
    n=1000,
    perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
    seed=42,
    out="./dispersion",
)

# Decide later that 2000 was the right size after all.
df_2000 = monte_carlo_extend(
    "./dispersion/manifest.jsonl",
    "mission.script",
    n=1000,
)
```

The first 1000 `run_id`s in `df_2000` are bit-equal to `df_1000` —
[`numpy.random.SeedSequence.spawn`][seed-spawn] is position-deterministic,
so per-run sub-seeds at indices `0..999` are independent of how many
total samples were requested. Equivalently, `df_2000` is bit-equal to a
fresh `monte_carlo(n=2000, seed=42, ...)` call: extend on top of `n=1000`
is indistinguishable from running `n=2000` from scratch.

[seed-spawn]: https://numpy.org/doc/stable/reference/random/parallel.html

The original `perturb` mapping and `seed` are read from the manifest
header — the caller does not (and cannot) change them. Adding new
perturbed parameters mid-sweep would break determinism; if the analysis
needs a different distribution shape, run a fresh sweep instead.

`monte_carlo_extend` refuses if the base sweep has any `failed` or
missing runs in its original `[0, n)` range, naming them and pointing at
[`Sweep.resume()`][gmat_sweep.Sweep.resume]. Mixing extension over an
unfinished base would produce a manifest with gaps in the original
range that downstream readers couldn't interpret — fill them in first,
then extend.

The on-disk header's `parameter_spec.n` stays frozen at the original
sweep's size (manifest headers are append-only). Use
[`Manifest.extension_run_count`][gmat_sweep.Manifest.extension_run_count]
to read the cumulative count of extension runs, or the simpler
`max(e.run_id for e in manifest.entries) + 1` for the total run count
on disk.

### Why Latin hypercube can't be extended

[`latin_hypercube_extend()`][gmat_sweep.latin_hypercube_extend] exists
to refuse the operation cleanly. Extending a Latin hypercube sweep
would change the per-axis stratification of every sample (the `n` bins
under [`scipy.stats.qmc.LatinHypercube`][scipy-lh] repartition when `n`
changes), so there is no slice of a larger LH draw that reproduces the
original `n` samples bit-for-bit. If you need more samples for an LH
study, run a fresh `latin_hypercube(n=old_n + new)` from scratch.

[scipy-lh]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html

## Worked example: launch dispersion

A typical injection-error analysis perturbs the post-burn state vector
around its nominal value and asks how the final orbit's miss distance
distributes:

```python
import math
import pandas as pd

from gmat_sweep import LocalJoblibPool, monte_carlo

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
    backend=LocalJoblibPool(workers=8),
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
