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
ordering is what the manifest and any future resume flow rely on.

The expansion is byte-for-byte deterministic across processes and machines:
two runs serialised through `json.dumps(..., sort_keys=True)` produce
identical bytes. This is the property the
[`canonical_script_sha256()`][gmat_sweep.canonical_script_sha256] hash and
the manifest header are designed against.

Empty grids (`grid={}`) are valid and produce a single run with no
overrides — the cartesian product of nothing has one element. Empty
*values* (e.g. `{"a": []}`) are rejected with
[`SweepConfigError`][gmat_sweep.SweepConfigError].

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

## See also

- [API reference: `sweep`](api.md#gmat_sweep.sweep)
- [Manifest schema](manifest-schema.md) — how the resulting manifest
  records the materialised grid.
