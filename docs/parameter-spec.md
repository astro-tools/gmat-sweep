# Parameter spec

The `grid` argument to [`sweep()`][gmat_sweep.sweep] is a mapping from
**dotted-path field names** to sequences of values. `gmat-sweep` builds the
cartesian product of those sequences and runs the mission script once per
combination.

```python
from gmat_sweep import sweep

df = sweep(
    "mission.script",
    grid={
        "Sat.SMA": [7000, 7100, 7200],   # 3 values
        "Sat.DryMass": [100, 200],       # 2 values
    },
)
# 3 × 2 = 6 runs
```

## Dotted-path keys

Every key in the `grid` mapping is a dotted-path that addresses one field on
one resource in the loaded `.script`, exactly as `gmat-run`'s
`Mission.__setitem__` consumes it. Examples:

| Key                       | What it sets                                    |
|---------------------------|-------------------------------------------------|
| `Sat.SMA`                 | `Sat`'s semi-major axis                         |
| `Sat.DryMass`             | `Sat`'s dry mass                                |
| `Prop.Type`               | the propagator's integrator type (string)       |
| `MainBurn.Element1`       | the first impulsive-burn delta-V component      |

The first dotted segment is the resource name as it appears in the script
(`Create Spacecraft Sat;` → `Sat`); the rest is the field path.
`gmat-sweep` does not validate the path itself — `gmat-run` raises at run
time if the path or value is rejected by GMAT, and that one run lands as a
`failed` row in the result DataFrame.

For a tour of the dotted-path syntax see `gmat-run`'s
[Mission reference](https://astro-tools.github.io/gmat-run/reference/mission/).

## Valid override types

Override values cross a process boundary as JSON, so they must be
JSON-encodable. In practice that means:

- **`int`** and **`float`** — most numeric fields (lengths, masses,
  durations, etc.).
- **`str`** — enum-style fields (e.g. `Prop.Type = "RungeKutta89"`).
- **`bool`** — flags.
- **`list`** of the above — for vector fields like burn elements.

`numpy` scalars and arrays should be cast to native Python types before
being passed in. `gmat-run` itself accepts numpy on the receive side, but
`gmat-sweep`'s [`RunSpec`][gmat_sweep.RunSpec] is the worker-boundary
serialisation surface and JSON does not encode numpy scalars natively.

## Stochastic specs

For Monte Carlo and Latin hypercube sweeps the per-axis value is a
*distribution* rather than a sequence. `gmat-sweep` accepts three
shorthand tuples and a pass-through for any pre-frozen
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
distribution:

| Spec                          | Equivalent `scipy.stats` call                       |
|-------------------------------|------------------------------------------------------|
| `("normal", mu, sigma)`       | `scipy.stats.norm(loc=mu, scale=sigma)`              |
| `("uniform", lo, hi)`         | `scipy.stats.uniform(loc=lo, scale=hi - lo)`         |
| `("lognormal", mu, sigma)`    | `scipy.stats.lognorm(s=sigma, scale=exp(mu))`        |
| any `scipy.stats.*` frozen rv | the rv itself (passes through unchanged)             |

```python
import math
from scipy import stats

# Three shorthand forms.
("normal", 7000.0, 5.0)            # SMA, ±5 km 1-σ
("uniform", 0.0, 360.0)             # RAAN, anywhere on the equator
("lognormal", math.log(100), 0.1)   # dry mass around 100 kg, log-normal spread

# Anything else: hand in a pre-frozen distribution directly.
stats.triang(c=0.5, loc=-1, scale=2)
```

The shorthands cover the cases that come up day-to-day in dispersion
analyses; reach for the pre-frozen path when you need a different shape
(triangular, beta, a truncated normal via `scipy.stats.truncnorm`, etc.).

`mu` and `sigma` for `lognormal` are the parameters of the *underlying*
normal — the mean and standard deviation of `log(X)`, not of `X`. This
matches `scipy.stats.lognorm`'s convention but trips up callers expecting
to specify the lognormal's own mean directly.

Validation is strict and happens up front:

- Tag must be one of `"normal"`, `"uniform"`, `"lognormal"`.
- `sigma` must be `> 0` for `normal` and `lognormal`.
- `hi` must be `> lo` for `uniform`.
- All numeric parameters must be finite (no `nan` or `inf`).

Any violation raises [`SweepConfigError`][gmat_sweep.SweepConfigError]
before any run starts.

### Monte Carlo vs Latin hypercube

[`monte_carlo()`][gmat_sweep.monte_carlo] draws each sample independently
from each distribution; the empirical coverage of any single axis is only
as good as the law of large numbers makes it for the chosen `n`.

[`latin_hypercube()`][gmat_sweep.latin_hypercube] uses
[`scipy.stats.qmc.LatinHypercube`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)
to stratify each axis into `n` equal-probability bins before transforming
through the distribution's quantile function. One sample per bin per axis
is guaranteed by construction, so the marginal coverage of every axis is
uniform at any `n`.

The rule of thumb: prefer LH when `n` is small relative to the number of
perturbed parameters (so each axis only gets a handful of samples and the
stratification visibly improves coverage). Plain Monte Carlo is the
right call when `n` is large, when you want the joint distribution to
match the product of marginals exactly (no LH-induced anti-correlation),
or when you intend to extend the sample count incrementally — LH samples
do not append cleanly across runs the way independent Monte Carlo draws
do.

## Full-factorial expansion

[`sweep()`][gmat_sweep.sweep] uses the full-factorial expansion in
[`full_factorial()`][gmat_sweep.full_factorial]:

- Keys are emitted in **lexicographic order**, so the iteration order of
  the input `grid` dict does not matter.
- The cartesian product enumerates in `itertools.product` order over the
  materialised input iterables. The lexicographically-first key varies
  *slowest*; the last key varies *fastest*.

For `{"a": [1, 2], "b": [10, 20, 30]}` the six override dicts come out as:

```text
(a=1, b=10), (a=1, b=20), (a=1, b=30),
(a=2, b=10), (a=2, b=20), (a=2, b=30)
```

`run_id` values are assigned in this order starting at `0`, and that
ordering is what the manifest and the [resume flow](resume.md) rely on.

The expansion is byte-for-byte deterministic across processes and machines:
two runs serialised through `json.dumps(..., sort_keys=True)` produce
identical bytes. This is the property the
[`canonical_script_sha256()`][gmat_sweep.canonical_script_sha256] hash and
the manifest header are designed against.

Empty grids (`grid={}`) are valid and produce a single run with no
overrides — the cartesian product of nothing has one element. Empty
*values* (e.g. `{"a": []}`) are rejected with
[`SweepConfigError`][gmat_sweep.SweepConfigError].

## Explicit-row sweeps

Pass a pre-built `pandas.DataFrame` as `samples=` instead of `grid=` when you
want full control over which rows run. The DataFrame's columns are
dotted-path field names, its rows are the run set, and its
`pd.RangeIndex(start=0, …)` becomes the `run_id` axis on the result frame.

```python
import pandas as pd
from gmat_sweep import sweep

samples = pd.DataFrame(
    {
        "Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0],
        "Sat.ECC": [0.001, 0.002, 0.003, 0.004],
    }
)
df = sweep("mission.script", samples=samples)
# 4 runs, run_ids 0..3, one per DataFrame row
```

`grid=` and `samples=` are **mutually exclusive** — passing both raises
[`SweepConfigError`][gmat_sweep.SweepConfigError], and so does passing
neither.

This is the underlying primitive the [`monte_carlo`][gmat_sweep.monte_carlo]
and [`latin_hypercube`][gmat_sweep.latin_hypercube] wrappers build on. Use
those when you want stochastic draws from named distributions; reach for
`samples=` directly when you have already built a custom design (Halton,
Sobol, a hand-curated edge-case grid) and want to hand it in unchanged.

### Worked example: a 64-point Latin hypercube

For Latin hypercube sampling against named distributions, reach for
[`latin_hypercube()`][gmat_sweep.latin_hypercube] directly — it builds the
unit-cube design, maps each axis through the user's distribution, and
delegates to the same explicit-row primitive:

```python
from gmat_sweep import latin_hypercube

df = latin_hypercube(
    "mission.script",
    n=64,
    perturb={
        "Sat.SMA": ("uniform", 6900.0, 7400.0),
        "Sat.ECC": ("uniform", 0.0005, 0.005),
    },
    seed=42,
    out="./lhs-sweep",
)
```

The wrapper records `{"_kind": "latin_hypercube", "perturb": ..., "n": ...,
"seed": ...}` on the manifest header so the design is reproducible from
the seed alone.

If you want a different stratified design (Halton, Sobol, a custom-built
DataFrame) hand it in via `samples=` and bypass the wrapper:

```python
import pandas as pd
from scipy.stats import qmc

from gmat_sweep import sweep

sampler = qmc.Halton(d=2, seed=42)
unit = sampler.random(n=64)
scaled = qmc.scale(unit, l_bounds=[6900.0, 0.0005], u_bounds=[7400.0, 0.005])
samples = pd.DataFrame(scaled, columns=["Sat.SMA", "Sat.ECC"])
df = sweep("mission.script", samples=samples, out="./halton-sweep")
```

The DataFrame must have:

- A default `pd.RangeIndex(start=0, stop=N)` — call `samples.reset_index(drop=True)`
  if you sliced or filtered rows. A non-default index raises
  [`SweepConfigError`][gmat_sweep.SweepConfigError].
- Unique, all-`str` column names — duplicate columns would silently lose
  data when each row is converted to an override dict.
- No fully-NaN columns. A NaN inside a single cell is fine and is
  forwarded to `gmat-run` as-is — `gmat-run` decides whether NaN is a
  valid value for a given dotted path.

### Manifest serialisation

Every parameter-spec shape carries a `_kind` discriminator on the
manifest header so a later loader can dispatch without inferring the
sweep kind from which keys are present. The explicit-row shape:

```json
{
  "parameter_spec": {
    "_kind":   "explicit",
    "columns": ["Sat.SMA", "Sat.ECC"],
    "rows":    [[7000.0, 0.001], [7100.0, 0.002], …]
  }
}
```

Reconstructing the DataFrame after the fact is one line:

```python
from gmat_sweep import Manifest

m = Manifest.load("./lhs-sweep/manifest.jsonl")
samples = pd.DataFrame(m.parameter_spec["rows"], columns=m.parameter_spec["columns"])
```

Grid sweeps emit `_kind: "grid"` alongside the materialised axes:

```json
{
  "parameter_spec": {
    "_kind":   "grid",
    "Sat.SMA": [7000.0, 7100.0],
    "Sat.ECC": [0.001, 0.002]
  }
}
```

Older manifests that omit `_kind` on grid sweeps keep loading:
[`Manifest.load`][gmat_sweep.Manifest.load] treats the absent
discriminator as `"grid"`. See [Manifest schema → `parameter_spec`
shapes](manifest-schema.md#parameter_spec-shapes) for the full
enumeration.

## CLI mini-grammar

The `gmat-sweep run` CLI accepts the same grids via repeated `--grid`
flags. The right-hand side of each `--grid` argument has two forms:

### Linspace: `name=lo:hi:count`

`count` evenly-spaced values from `lo` to `hi` *inclusive*, via
`numpy.linspace`. `lo` and `hi` are floats, `count` is an integer ≥ 2.

```bash
gmat-sweep run --grid Sat.SMA=7000:7400:5 --out ./sweep mission.script
# Sat.SMA ∈ {7000.0, 7100.0, 7200.0, 7300.0, 7400.0}
```

### Explicit list: `name=v1,v2,v3`

A comma-separated list. Each token is coerced to `int` if possible, then
`float`, then left as a `str`:

```bash
gmat-sweep run \
  --grid Sat.SMA=7000,7100,7200 \
  --grid Sat.DryMass=100,200 \
  --grid Prop.Type=RungeKutta89,PrinceDormand78 \
  --out ./sweep \
  mission.script
```

That sweep runs 3 × 2 × 2 = 12 cells.

A grid name may only appear once per CLI invocation; passing the same name
twice raises [`SweepConfigError`][gmat_sweep.SweepConfigError]. There is no
way to mix linspace and explicit notation on the same axis — pick one.

## Choosing a backend

By default [`sweep()`][gmat_sweep.sweep],
[`monte_carlo()`][gmat_sweep.monte_carlo], and
[`latin_hypercube()`][gmat_sweep.latin_hypercube] dispatch runs through a
fresh [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] over every available
core. Pass an explicit pool via `backend=` to:

- **Cap parallelism** for a long-running sweep so the box stays responsive:

    ```python
    from gmat_sweep import LocalJoblibPool, sweep

    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=LocalJoblibPool(workers=4),
        out="./sweep",
    )
    ```

- **Share one pool across several sweeps** — the worker bootstrap cost
  (fresh interpreter + `gmatpy` import per worker) is paid once instead of
  once per `sweep()` call:

    ```python
    from gmat_sweep import LocalJoblibPool, monte_carlo, sweep

    with LocalJoblibPool(workers=8) as pool:
        baseline = sweep(
            "mission.script",
            grid={"Sat.DryMass": [100.0, 150.0, 200.0]},
            backend=pool,
            out="./baseline",
        )
        dispersion = monte_carlo(
            "mission.script",
            n=200,
            perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
            seed=42,
            backend=pool,
            out="./dispersion",
        )
    ```

- **Use a different execution backend** — any [`Pool`][gmat_sweep.Pool]
  subclass works, including third-party ones. The pool's class name is
  recorded on the manifest header so a later loader can tell which
  backend produced the sweep.

When you supply `backend=`, you own the pool's lifecycle —
[`sweep()`][gmat_sweep.sweep] will not call
[`Pool.close()`][gmat_sweep.Pool.close] on it. Wrap it in a `with` block,
or call `close()` yourself when the pool is done. The `backend=None`
default closes the pool it built when `sweep()` returns.

## See also

- [API reference: `sweep`](api.md#gmat_sweep.sweep)
- [Manifest schema](manifest-schema.md) — how the resulting manifest
  records the materialised grid.
