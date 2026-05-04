# Getting started

## Install

```bash
pip install gmat-sweep
```

The `[examples]` extra pulls in `matplotlib` for the example notebooks:

```bash
pip install gmat-sweep[examples]
```

To work on `gmat-sweep` itself, clone and sync the dev groups:

```bash
git clone https://github.com/astro-tools/gmat-sweep.git
cd gmat-sweep
uv sync --all-groups
```

You also need a working **GMAT install** on the same machine. `gmat-sweep`
does not bundle GMAT binaries; it depends on
[`gmat-run`](https://github.com/astro-tools/gmat-run) for the single-run
primitive, and `gmat-run` discovers your local install at runtime. See
[`gmat-run`'s install guide](https://astro-tools.github.io/gmat-run/install-gmat/)
for download and configuration steps. R2025a and R2026a are the supported
releases; see [Supported versions](supported-versions.md) for the full matrix.

## The four-line vision snippet

Once GMAT is installed and `gmat-sweep` is importable:

```python
from gmat_sweep import sweep

df = sweep("mission.script", grid={"Sat.SMA": [7000, 7100, 7200]})
print(df)
```

That call runs `mission.script` three times — once per `Sat.SMA` value —
each in a fresh subprocess, and returns a `(run_id, time)`-MultiIndexed
`pandas.DataFrame` containing the rows from every run's `ReportFile`,
labelled with a `__status` column.

The snippet is runnable against any of the GMAT-shipped sample scripts (e.g.
`samples/Tut_GettingStarted.script`) — pick one that defines a Spacecraft
named `Sat` and a `ReportFile`, or change the grid key to match your script.

## What just happened

Behind that single call:

1. The grid is materialised into a cartesian product of override dicts.
2. Each combination becomes a [`RunSpec`][gmat_sweep.RunSpec] with a unique
   `run_id` and its own per-run output directory.
3. A [`Pool`][gmat_sweep.Pool] (the default `LocalJoblibPool`, backed by
   `joblib`'s loky executor) fans the specs out to subprocess workers. Each worker imports `gmat_run` once, loads `mission.script`,
   applies its overrides via the dotted-path setter, runs the mission, and
   writes each `ReportFile` as a Parquet file under the per-run directory.
4. As workers complete, the orchestrator appends one
   [`ManifestEntry`][gmat_sweep.ManifestEntry] per run to a `manifest.jsonl`
   in the sweep's output directory — fsync'd line by line so a `Ctrl-C`
   leaves a parseable file containing every run that finished.
5. Once the pool drains, the per-run Parquet files are stitched into the
   returned DataFrame.

By default the output directory is a `tempfile.TemporaryDirectory` whose
lifetime is tied to the returned DataFrame. To keep the per-run Parquet
files and the manifest, pass an explicit `out=Path(...)` to
[`sweep()`][gmat_sweep.sweep].

## CLI alternative

The same call from a shell:

```bash
gmat-sweep run --grid Sat.SMA=7000:7200:3 --out ./sweep mission.script
```

`--grid name=lo:hi:count` produces `count` evenly spaced points from `lo` to
`hi` inclusive; `--grid name=v1,v2,v3` uses an explicit list. Repeat
`--grid` for additional axes; the cartesian product is run. See
[Parameter spec](parameter-spec.md) for the full grammar.

## Next steps

- [Parameter spec](parameter-spec.md) — what overrides are valid and how
  the cartesian product is laid out.
- [Manifest schema](manifest-schema.md) — what's in `manifest.jsonl` and
  how to load it back.
- [API reference](api.md) — every public symbol, auto-generated.
