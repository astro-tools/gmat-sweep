# Aggregating sweep outputs

`gmat-sweep` writes each run's outputs as one Parquet file per GMAT
output kind, then assembles them on demand into multi-indexed
`pandas.DataFrame`s. Three entry points cover the three kinds gmat-run
surfaces:

| Function | GMAT output kind | Index |
|----------|------------------|-------|
| [`lazy_multiindex`][gmat_sweep.lazy_multiindex] | `ReportFile` | `(run_id, time)` |
| [`lazy_ephemerides`][gmat_sweep.lazy_ephemerides] | `EphemerisFile` (OEM, STK, SPK) | `(run_id, time)` |
| [`lazy_contacts`][gmat_sweep.lazy_contacts] | `ContactLocator` | `(run_id, interval_id)` |

The matching [`Sweep`][gmat_sweep.Sweep] convenience methods —
`Sweep.to_dataframe`, `Sweep.to_ephemerides`, `Sweep.to_contacts` —
delegate to these with the sweep's manifest and output directory already
bound. [`sweep()`][gmat_sweep.sweep] returns
`Sweep.to_dataframe(name=None)` directly for the common single-report
case.

## The `name` selector

Every entry point accepts `name: str | None = None`. With one output of
the relevant kind across the sweep, `name=None` resolves to that sole
output automatically. With two or more, `name=None` raises
[`SweepConfigError`][gmat_sweep.SweepConfigError] listing the available
names — pass `name="..."` to pick one, and call the same function twice
to get two frames.

```python
from pathlib import Path

from gmat_sweep import Manifest, lazy_contacts, lazy_ephemerides, lazy_multiindex, sweep

# Single-report case: sweep() returns the report frame directly.
reports = sweep(
    "mission.script",
    grid={"Sat.SMA": [7000, 7100, 7200]},
    out=Path("./sweep"),
)

# Mixed outputs (report + ephemeris + contact): re-load the manifest and
# pull each frame independently.
manifest = Manifest.load(Path("./sweep/manifest.jsonl"))
ephemerides = lazy_ephemerides(manifest, Path("./sweep"))
contacts = lazy_contacts(manifest, Path("./sweep"))

# Two ReportFiles (e.g. spacecraft state + maneuvers): pass `name=` to pick.
states = lazy_multiindex(manifest, Path("./sweep"), name="StateReport")
burns = lazy_multiindex(manifest, Path("./sweep"), name="BurnReport")
```

## Failed and skipped runs

Failed and skipped runs surface as one row per run with the data columns
NaN-filled and the `__status` column set to `"failed"` or `"skipped"`.
The secondary index level carries a kind-appropriate missing value:

- `lazy_multiindex` / `lazy_ephemerides` — `time = NaT` (`datetime64[ns]`).
- `lazy_contacts` — `interval_id = pd.NA` (nullable `Int64`).

An `ok` run that ran successfully but did not produce the requested
output kind (e.g. asking for ephemerides on a sweep where one specific
run only emitted reports) lands the same way, with `__status="ok"` so
it remains distinguishable from a true failure.

## Index shapes

### Reports and ephemerides — `(run_id, time)`

The worker copies the first datetime column of each frame to a column
literally named `time` before writing Parquet — so user column names
(`Sat.UTCGregorian`, `Epoch`, …) round-trip into the aggregated frame
unchanged, while the aggregator gets the consistent `time` level it
needs. SPK, STK-TimePosVel, and CCSDS-OEM ephemeris frames all expose
their epoch as `Epoch`, so the same synthesis covers every gmat-run
ephemeris format.

### Contacts — `(run_id, interval_id)`

Contact frames are intervals, not point samples — one row per
visibility interval. The worker assigns a fresh `interval_id` column
(`range(len(df))`, so `0..K-1` per run) at write time. Use `interval_id`
the same way you'd use a per-run row position; the actual visibility
times are still in the data columns (`Start`, `Stop`, `Duration`, etc.,
depending on the `ContactLocator.ReportFormat` setting).

## Fusing multiple reports per run

When a sweep produces several `ReportFile` outputs and you want them
side-by-side on a shared timeline,
[`lazy_fused_reports`][gmat_sweep.lazy_fused_reports] (and
`Sweep.to_fused_reports`) reshape them into one wide DataFrame with a
column-level `pandas.MultiIndex` keyed by `(report_name, column)`. The
first name in `names` is the merge anchor; subsequent reports are
joined onto it per `run_id`.

| `tolerance` | Merge per run | When it fits |
|-------------|---------------|--------------|
| `"exact"` (literal) | inner join on `time` | every report shares the same step setting and you only want rows present in all of them |
| `pd.Timedelta(...)` | `pd.merge_asof` (`backward` direction, default) | reports use different cadences and a "nearest within window" match per anchor row is what you want |

```python
from pathlib import Path

import pandas as pd
from gmat_sweep import Manifest, lazy_fused_reports

manifest = Manifest.load(Path("./sweep/manifest.jsonl"))

# Two reports on the same step setting → exact inner join.
exact = lazy_fused_reports(
    manifest, Path("./sweep"), names=["StateReport", "BurnReport"], tolerance="exact",
)

# Reports on different cadences (e.g. 1 Hz state + 10 s maneuvers) → asof merge
# with a per-row tolerance window.
fused = lazy_fused_reports(
    manifest,
    Path("./sweep"),
    names=["StateReport", "BurnReport"],
    tolerance=pd.Timedelta(seconds=2),
)

# Anchor's columns under (StateReport, *), right-side under (BurnReport, *).
fused[("StateReport", "Sat.X")]
fused[("BurnReport", "Sat.Tank.Mass")]
```

### Column shape

| Column | Meaning |
|--------|---------|
| `(report_name, column)` | data column from that report |
| `(report_name, "__status")` | per-report status — preserves the [`lazy_multiindex`][gmat_sweep.lazy_multiindex] contract for each report independently (e.g. one report failed-for-this-run while others succeeded) |
| `("__status", "")` | run-level status, mirrored from the manifest entry |

A run whose anchor failed (`status != "ok"` for the first name in
`names`, or its parquet was missing) lands as a single `time=NaT` row
with all data NaN. The per-report `__status` columns still surface every
report's individual state for that run, but the other reports' data is
not merged in — anchor-failure shadows the rest of the row. Pick the
report most likely to be present as the first entry of `names`.

`tolerance` is required (no default). See the
[`pandas.merge_asof` reference](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html)
for the `direction` (`backward` by default), `allow_exact_matches`, and
accepted `tolerance` types.

## Memory: streaming vs. eager reads

`lazy_multiindex` and `lazy_ephemerides` accept `spool: bool = True`.
With `spool=True` (default) each per-run Parquet is streamed through
pandas one record batch at a time, so peak conversion memory is one
batch rather than one full sweep. `spool=False` reads each Parquet
eagerly in one shot — simpler control flow, higher peak memory, useful
on small sweeps. The result frame is identical either way.

`lazy_contacts` does not take a `spool` flag — `ContactLocator` outputs
are typically tiny (one row per pass) and the streaming overhead is not
worth the knob.

## Summarising across runs

`gmat-sweep` returns one row per `(run_id, time)` — every run kept,
every time step kept. The canonical next step for dispersion analysis
is to collapse across runs at each time step into per-time
statistics: median, 5th and 95th percentile, mean, std, and a count of
ok contributions. [`sweep_summary`][gmat_sweep.sweep_summary] does
exactly that and pairs with
[`sweep_band_plot`][gmat_sweep.plotting.sweep_band_plot] (gated on the
`[plot]` extra) for the matching figure.

```python
from pathlib import Path

from gmat_sweep import Manifest, lazy_multiindex, sweep_summary
from gmat_sweep.plotting import sweep_band_plot

manifest = Manifest.load(Path("./sweep/manifest.jsonl"))
df = lazy_multiindex(manifest, Path("./sweep"))

# Default: per-time-step 5/50/95 + mean + std across runs.
summary = sweep_summary(df)

# (time, q=0.5) slice — the median Sat.X over time across all runs.
median_x = summary[("q0.5", "Sat.X")]

# Median + 5–95% band for Sat.X.
ax = sweep_band_plot(summary, "Sat.X")
ax.figure.savefig("sat_x_band.png")
```

### Output shape

The result is a single DataFrame whose row index is the unique values
of the `by` level and whose column index is a two-level
`pandas.MultiIndex`:

| Column | Meaning |
|--------|---------|
| `(statistic, field)` | one column per `(statistic, original-column)` pair |

Statistic labels are exactly the entries of `include` followed by
`f"q{q_val}"` for each requested quantile — e.g. `"mean"`, `"std"`,
`"q0.05"`, `"q0.5"`, `"q0.95"` for the defaults. `count_ok` counts
non-NaN values per group; useful for spotting time steps where many
runs produced NaNs.

### `by="time"` vs `by="run_id"`

- `by="time"` (default) — collapse across runs at each time step. The
  natural input for "median over time with a 5/95% band".
- `by="run_id"` — collapse across time steps within each run. The
  natural input for per-run summary metrics (e.g. mean Sat.X over the
  whole trajectory).

Other values raise `SweepConfigError`. Categorical groupings and
arbitrary `by=` keys are intentionally out of scope in this release.

### Failed and skipped runs

By default (`dropna=True`) `sweep_summary` filters rows where
`__status != "ok"` before aggregating, so failed and skipped runs are
excluded from every statistic. Pass `dropna=False` to keep them — the
NaT marker rows from non-ok runs land as a NaT-keyed group in the
output (mostly NaN, with `count_ok` reflecting the contribution).

## Comparing two sweeps

Once you have two sweep DataFrames of the same shape — baseline vs.
perturbed, before vs. after a `.script` edit, two backends on the same
sweep — [`sweep_diff`][gmat_sweep.sweep_diff] turns them into a single
diff frame ready for plotting.

```python
from pathlib import Path

from gmat_sweep import Manifest, lazy_multiindex, sweep_diff

baseline = lazy_multiindex(Manifest.load(Path("./baseline/manifest.jsonl")), Path("./baseline"))
perturbed = lazy_multiindex(Manifest.load(Path("./perturbed/manifest.jsonl")), Path("./perturbed"))

# Per-row absolute and relative diff for every shared numeric column.
diff = sweep_diff(baseline, perturbed)

# (run_id, time) → Sat.SMA shifted by exactly +50 km, everywhere.
diff[["Sat.SMA__diff", "Sat.SMA__rel"]].head()
```

For each numeric column shared between the two inputs, `sweep_diff`
emits `<col>__diff = b - a` and/or `<col>__rel = (b - a) / a`. Columns
present on only one side, and shared columns whose dtype is non-numeric,
are silently dropped — the contract is "compare the comparable".

### Output shape

| Column | Meaning |
|--------|---------|
| `<col>__diff` | absolute difference, `b - a` |
| `<col>__rel` | relative difference, `(b - a) / a` (NaN where `a == 0`) |
| `__status_diff` | `"ok"` when both sides are `__status="ok"`; otherwise `"<a_status>/<b_status>"` (e.g. `"failed/ok"`). Omitted when neither input has a `__status` column. |

The row index matches the inputs' (aligned) index — typically
`(run_id, time)` or just `run_id` after the `on="run_id"` collapse below.

### `how` selects which suffixes appear

| `how` | Columns |
|-------|---------|
| `"absolute"` | only `<col>__diff` |
| `"relative"` | only `<col>__rel` |
| `"both"` (default) | both, interleaved as `<col>__diff`, `<col>__rel`, `<col2>__diff`, `<col2>__rel`, … |

### `on=None` vs `on="run_id"`

- `on=None` (default) — align on the existing index (typically
  `(run_id, time)`). The diff is per-row, per-time-step.
- `on="run_id"` — collapse each side to its per-run **final-step row**
  via `groupby(level="run_id").last()`, then diff. Output is indexed by
  `run_id`. The natural shape for "did the dispersion of the final
  state change?" comparisons.

### Tolerance masking

`tolerance=` masks every diff whose absolute value is strictly below
the cutoff to NaN — both `__diff` and the matching `__rel` are masked
at the same positions, so the surviving non-NaN entries highlight the
meaningful changes only.

```python
# Single cutoff applied to every column.
diff = sweep_diff(baseline, perturbed, tolerance=1e-6)

# Per-column cutoffs — useful when columns carry mixed units.
def cutoff(col: str) -> float:
    return 1e-3 if col.endswith(".SMA") else 1e-6  # km vs. km/s

diff = sweep_diff(baseline, perturbed, tolerance=cutoff)
```

### Failed and skipped runs

`__status_diff` records the per-row pair so a downstream filter or plot
can drop or annotate them:

```python
clean = diff.loc[diff["__status_diff"] == "ok"]
mismatched = diff.loc[diff["__status_diff"] != "ok"]
```

A row where one side failed and the other succeeded surfaces as
`"failed/ok"` (or the symmetric `"ok/failed"`), with the data columns
NaN — failed runs do not produce numeric outputs to subtract.

### Index alignment

`sweep_diff` aligns the two inputs on the intersection of their
indexes. Keys present on only one side are silently dropped. The
function does **not** reshape across `parameter_spec` shapes — diffing a
grid sweep against a Monte Carlo sweep is the user's responsibility to
align (e.g. via `df.reset_index().set_index([...])`) before calling.

## Polars output engine

Pandas is the default and only return type with no extra installed.
With the `[polars]` extra, every flat-column DataFrame-returning entry
point in this module accepts an `engine="polars"` keyword that returns
a [`polars.DataFrame`][polars-df] instead.

```bash
pip install gmat-sweep[polars]
```

The MultiIndex on `lazy_multiindex`/`lazy_ephemerides` (`(run_id, time)`)
and `lazy_contacts` (`(run_id, interval_id)`) is flattened into two
sorted leading columns; row order, row count, and the non-index column
set match the pandas-engine equivalent. Polars carries the typed nulls
across — `NaT` becomes a polars `null` in `Datetime[ns]`, the nullable
`Int64` `interval_id` round-trips to a polars `Int64` with `null`, and
`NaN` in numeric columns becomes a `null` in `Float64`.

```python
from gmat_sweep import sweep

# pandas (default) — returns a (run_id, time)-MultiIndexed pandas DataFrame.
df = sweep("mission.script", grid={"Sat.SMA": [7000, 7100]}, out=...)

# polars — returns a polars.DataFrame with run_id/time as leading columns.
plf = sweep("mission.script", grid={"Sat.SMA": [7000, 7100]}, out=..., engine="polars")
plf.filter(plf["__status"] == "ok").group_by("run_id").agg(...)
```

The `engine="polars"` knob is available on:

- The top-level entry points: [`sweep`][gmat_sweep.sweep],
  [`monte_carlo`][gmat_sweep.monte_carlo],
  [`latin_hypercube`][gmat_sweep.latin_hypercube], and
  [`monte_carlo_extend`][gmat_sweep.monte_carlo_extend].
- The `Sweep` orchestrator methods:
  [`Sweep.to_dataframe`][gmat_sweep.Sweep.to_dataframe],
  [`Sweep.to_ephemerides`][gmat_sweep.Sweep.to_ephemerides],
  [`Sweep.to_contacts`][gmat_sweep.Sweep.to_contacts]. The
  `Sweep.to_polars()` shortcut is equivalent to
  `Sweep.to_dataframe(engine="polars")`.
- The standalone aggregators:
  [`lazy_multiindex`][gmat_sweep.lazy_multiindex],
  [`lazy_ephemerides`][gmat_sweep.lazy_ephemerides],
  [`lazy_contacts`][gmat_sweep.lazy_contacts],
  [`mc_convergence`][gmat_sweep.mc_convergence], and
  [`sweep_diff`][gmat_sweep.sweep_diff].

Two helpers stay pandas-only because their output carries a column-level
`MultiIndex` and polars has no native equivalent:
[`sweep_summary`][gmat_sweep.sweep_summary] (`(statistic, field)`
columns) and
[`lazy_fused_reports`][gmat_sweep.lazy_fused_reports] /
[`Sweep.to_fused_reports`][gmat_sweep.Sweep.to_fused_reports]
(`(report_name, column)` columns). Convert by hand when you need a
flat polars frame:

```python
import polars as pl

summary = sweep_summary(df)
flat = summary.copy()
flat.columns = [f"{stat}__{field}" for stat, field in flat.columns]
plf_summary = pl.from_pandas(flat.reset_index())
```

[polars-df]: https://docs.pola.rs/api/python/stable/reference/dataframe/index.html

## Migrating from v0.1

v0.1 used a bare `<name>.parquet` per-run layout and keyed
`output_paths` by `<name>`; v0.2 prepends the kind prefix
(`report__<name>.parquet`, key `report__<name>`). v0.1 manifests are not
readable by v0.2 aggregators — the aggregator dispatch is keyed on the
prefix and v0.1 entries lack one. Re-run any sweep you need to
re-aggregate under v0.2.
