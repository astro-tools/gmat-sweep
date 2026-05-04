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

## Migrating from v0.1

v0.1 used a bare `<name>.parquet` per-run layout and keyed
`output_paths` by `<name>`; v0.2 prepends the kind prefix
(`report__<name>.parquet`, key `report__<name>`). v0.1 manifests are not
readable by v0.2 aggregators — the aggregator dispatch is keyed on the
prefix and v0.1 entries lack one. Re-run any sweep you need to
re-aggregate under v0.2.
