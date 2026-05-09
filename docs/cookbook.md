# Cookbook: integrating sweep outputs into downstream consumers

A sweep produces one canonical artifact set: a `manifest.jsonl`, a set of
per-run Parquet files, and the multi-indexed
[`pandas.DataFrame`][pandas-dataframe] the aggregator builds from them.
Most downstream consumers — visualisers, validation harnesses, archival
formats, external mission tools — want a *different* shape on the way in.
This page walks three patterns for getting from one to the other.

The recipes here are integration-side patterns, not new `gmat-sweep`
APIs. Every helper shown is a few lines of caller-side code; nothing
below ships in the package itself. The intent is to document the shape
of the contracts so you can drop the snippets into your own scripts.

[pandas-dataframe]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

## Pattern 1 — Visualisation export

Browser viewers and ground-segment tools rarely consume Parquet
directly. The two formats that cover most of the field are
[**CCSDS-OEM**](https://public.ccsds.org/Pubs/502x0b3e1.pdf) (Orbit
Ephemeris Message — what most ground stations and trajectory tools
ingest) and [**CZML**](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Structure)
(Cesium's time-dynamic packet format for browser viewers).

[`Sweep.to_ephemerides()`][gmat_sweep.Sweep.to_ephemerides] gives you
the `(run_id, time)`-indexed frame; the export step is a deterministic
transform on top.

### CCSDS-OEM 502.0-B-3 (KVN)

The KVN form is the one GMAT itself emits and the one most consumers
accept. A complete OEM file has a header, one or more `META`/`DATA`
segment pairs, and CRLF line endings. Per the standard, position is in
km and velocity (when present) is in km/s, in the declared
`REF_FRAME` / `TIME_SYSTEM`.

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CRLF = "\r\n"


def to_oem(
    df: pd.DataFrame,
    run_id: int,
    out: Path,
    *,
    object_name: str,
    object_id: str,
    ref_frame: str = "EME2000",
    time_system: str = "UTC",
    columns: tuple[str, ...] = ("X", "Y", "Z"),
) -> Path:
    """Write one run's ephemeris to a CCSDS-OEM 502.0-B-3 KVN file.

    `df` is the frame returned by `Sweep.to_ephemerides()`; `columns`
    are the position (or position+velocity) column names to emit per
    sample. Velocity is emitted iff len(columns) == 6.
    """
    run = df.xs(run_id, level="run_id")
    if "X" not in run.columns and len(columns) == 3:
        # Some EphemerisFile formats prefix columns with the spacecraft
        # name (e.g. "Sat.X"). The aggregator preserves them verbatim.
        raise KeyError(f"expected position columns {columns} in {list(run.columns)}")

    epochs = run.index.get_level_values("time")
    start = pd.Timestamp(epochs.min()).to_pydatetime().replace(tzinfo=timezone.utc)
    stop = pd.Timestamp(epochs.max()).to_pydatetime().replace(tzinfo=timezone.utc)
    creation = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE  = {creation}",
        "ORIGINATOR     = gmat-sweep",
        "META_START",
        f"OBJECT_NAME    = {object_name}",
        f"OBJECT_ID      = {object_id}",
        "CENTER_NAME    = EARTH",
        f"REF_FRAME      = {ref_frame}",
        f"TIME_SYSTEM    = {time_system}",
        f"START_TIME     = {start:%Y-%m-%dT%H:%M:%S.%f}",
        f"STOP_TIME      = {stop:%Y-%m-%dT%H:%M:%S.%f}",
        "META_STOP",
        "",
    ]
    for ts, row in run.iterrows():
        epoch_str = pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")
        values = " ".join(f"{row[c]:.9e}" for c in columns)
        lines.append(f"{epoch_str} {values}")

    out.write_text(CRLF.join(lines) + CRLF)
    return out
```

The translation is straightforward because the canonical schema —
`(run_id, time)` index, position columns in km, optional velocity in
km/s — already matches OEM's column ordering. Pick one `run_id` per
file: OEM is per-object, and consumers expect one segment per file.

To export every successful run in a sweep:

```python
from pathlib import Path

from gmat_sweep import Sweep

sweep = Sweep.from_manifest(Path("./sweep/manifest.jsonl"))
ephem = sweep.to_ephemerides()

out_dir = Path("./oem")
out_dir.mkdir(exist_ok=True)
for run_id in ephem.index.get_level_values("run_id").unique():
    to_oem(
        ephem,
        int(run_id),
        out_dir / f"run-{int(run_id):04d}.oem",
        object_name="Sat",
        object_id=f"SAT-{int(run_id):04d}",
    )
```

### CZML for Cesium viewers

CZML is a JSON document — an array of *packets*. The first packet is a
`document` packet declaring the clock; subsequent packets each describe
one entity (here, one trajectory). Time-dynamic position uses
`cartesian` samples laid out as
`[t0, x0, y0, z0, t1, x1, y1, z1, …]`, with positions in metres in the
`Inertial` (EME2000) frame.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def to_czml(
    df: pd.DataFrame,
    run_ids: list[int],
    out: Path,
    *,
    columns: tuple[str, str, str] = ("X", "Y", "Z"),
    interpolation_algorithm: str = "LAGRANGE",
    interpolation_degree: int = 5,
) -> Path:
    """Write one packet per ``run_id`` as a CZML document to ``out``."""
    epoch_iso = (
        pd.Timestamp(df.index.get_level_values("time").min())
        .tz_localize(None)
        .isoformat() + "Z"
    )
    document_packet: dict[str, Any] = {
        "id": "document",
        "name": "gmat-sweep",
        "version": "1.0",
        "clock": {
            "interval": _czml_interval(df),
            "currentTime": epoch_iso,
            "multiplier": 60,
        },
    }
    packets: list[dict[str, Any]] = [document_packet]

    base = pd.Timestamp(df.index.get_level_values("time").min())
    for run_id in run_ids:
        run = df.xs(run_id, level="run_id")
        seconds = (run.index - base).total_seconds().to_numpy()
        cartesian: list[float] = []
        for offset, (_, row) in zip(seconds, run.iterrows(), strict=True):
            cartesian.extend(
                [float(offset), float(row[columns[0]]) * 1000.0,
                 float(row[columns[1]]) * 1000.0,
                 float(row[columns[2]]) * 1000.0]
            )
        packets.append({
            "id": f"run-{run_id}",
            "name": f"Run {run_id}",
            "position": {
                "epoch": epoch_iso,
                "interpolationAlgorithm": interpolation_algorithm,
                "interpolationDegree": interpolation_degree,
                "referenceFrame": "INERTIAL",
                "cartesian": cartesian,
            },
        })

    out.write_text(json.dumps(packets, indent=2))
    return out


def _czml_interval(df: pd.DataFrame) -> str:
    epochs = df.index.get_level_values("time")
    start = pd.Timestamp(epochs.min()).tz_localize(None).isoformat() + "Z"
    stop = pd.Timestamp(epochs.max()).tz_localize(None).isoformat() + "Z"
    return f"{start}/{stop}"
```

The km→m unit conversion is the easy thing to miss: GMAT ephemerides
are in km, CZML's `cartesian` channel is in metres. The
`interpolationAlgorithm` knob is the consumer-side hint for how a
viewer should resample between the points you give it; `LAGRANGE` of
degree 5 is the typical choice for orbital state.

## Pattern 2 — Cross-tool validation

A common downstream task: re-run the same parameter set through a
second tool (a different propagator, a different mission-design suite,
an in-house simulator) and ask whether the two agree. The manifest +
multi-indexed frame is everything a generic harness needs.

### What a validation harness keys on

| Source | Field | Used for |
|---|---|---|
| Manifest header | `script_sha256` | refusing comparison if the input scripts differ |
| Manifest header | `parameter_spec` | recovering the run set the sweep expanded |
| Manifest header | `gmat_install_version`, `gmat_run_version` | provenance — flagging cross-version drift |
| `ManifestEntry.run_id` | per-row identity | aligning runs across tools (same `run_id` ↔ same overrides) |
| `ManifestEntry.overrides` | the dotted-path → value dict for that run | re-driving Tool B with the same parameters |
| `ManifestEntry.status` | `ok` / `failed` / `skipped` | filtering the comparable subset |
| Aggregated frame | `(run_id, time)` MultiIndex + state columns | per-sample residuals |

The first two header fields are the gate: if `script_sha256` differs
between the two manifests, the runs are not directly comparable and the
harness should refuse rather than silently produce nonsense numbers.
The remaining fields drive the per-run join.

### A worked harness against the 16-run SMA grid

The repo's [reference sweep][reference-sweep] is a 16-run SMA scan over
the LEO fixture — `Sat.SMA` swept linearly from 7000 km to 7300 km.
Treat that as Tool A; the harness below joins it against an arbitrary
Tool B manifest pointed at by `tool_b_root`.

[reference-sweep]: https://github.com/astro-tools/gmat-sweep/blob/main/tests/test_reference_sweep.py

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gmat_sweep import Manifest, Sweep


def compare_sweeps(
    tool_a_root: Path,
    tool_b_root: Path,
    *,
    state_columns: tuple[str, ...] = ("Sat.EarthMJ2000Eq.X",
                                      "Sat.EarthMJ2000Eq.Y",
                                      "Sat.EarthMJ2000Eq.Z"),
    tolerance: pd.Timedelta = pd.Timedelta(seconds=1),
) -> pd.DataFrame:
    """Per-run RMS position residual between two sweeps over the same script."""
    a_manifest = Manifest.load(tool_a_root / "manifest.jsonl")
    b_manifest = Manifest.load(tool_b_root / "manifest.jsonl")

    if a_manifest.script_sha256 != b_manifest.script_sha256:
        raise ValueError(
            "scripts differ — refusing to compare. "
            f"A: {a_manifest.script_sha256[:12]}, B: {b_manifest.script_sha256[:12]}"
        )

    a_df = Sweep.from_manifest(tool_a_root / "manifest.jsonl").to_dataframe()
    b_df = Sweep.from_manifest(tool_b_root / "manifest.jsonl").to_dataframe()

    a_ok = {e.run_id: e.overrides for e in a_manifest.entries if e.status == "ok"}
    b_ok = {e.run_id: e.overrides for e in b_manifest.entries if e.status == "ok"}
    common = sorted(set(a_ok) & set(b_ok))

    rows: list[dict[str, float | int]] = []
    for run_id in common:
        if a_ok[run_id] != b_ok[run_id]:
            # Same run_id, different overrides → not the same scenario.
            continue
        a_run = a_df.xs(run_id, level="run_id")[list(state_columns)]
        b_run = b_df.xs(run_id, level="run_id")[list(state_columns)]
        joined = pd.merge_asof(
            a_run.sort_index(),
            b_run.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            suffixes=("_a", "_b"),
            direction="nearest",
        )
        diffs = np.array([
            joined[f"{c}_a"].to_numpy() - joined[f"{c}_b"].to_numpy()
            for c in state_columns
        ])
        rms = float(np.sqrt(np.mean(np.sum(diffs**2, axis=0))))
        rows.append({"run_id": run_id, "rms_km": rms, "n_samples": len(joined)})

    return pd.DataFrame(rows).set_index("run_id")
```

The interesting pieces, in order:

1. **`script_sha256` gate.** The manifest's canonical hash is computed
   after line-ending and trailing-newline normalisation, so two clones
   of the same `.script` checked out under different settings still
   match. A real difference is a real signal: refuse rather than
   compare.
2. **Same-`run_id` ↔ same-`overrides` invariant.** Two sweeps over the
   same script with the same `parameter_spec` produce the same
   `run_id` → `overrides` mapping in the manifest. If the harness sees
   a `run_id` collision with disagreeing overrides, the sweeps were
   parameterised differently and the row should be dropped.
3. **`merge_asof` with a tolerance.** Two propagators rarely emit
   samples at exactly the same epochs, so the per-run join is nearest-
   match within a tolerance. `pd.Timedelta(seconds=1)` is right when
   both sides emit near-1-Hz state.
4. **Failure-state filtering.** Restrict the comparison to runs where
   *both* sides report `status == "ok"`. Runs that failed on one side
   are useful for a separate cross-tool error analysis but pollute the
   numerical-residual aggregate.

The output is a per-run summary indexed by `run_id`. Pivot it back
through the manifest's `overrides` if you want a parameter-keyed view
(e.g. residual vs. SMA).

## Pattern 3 — External-tool wrapping

Sometimes the consumer is a separate process — a Slurm job, a CI step,
a partner team's pipeline — that needs the *whole sweep* as a portable
artifact, not a Python object. Today the portable bundle is the sweep's
on-disk output directory, with three guarantees:

- **A single manifest** at the root: `manifest.jsonl` is the table of
  contents. Header carries the script hash and per-run schema; one
  entry per run carries `overrides`, `status`, and `output_paths`.
- **Self-describing per-run Parquets** under the same directory. Every
  successful run's outputs are recorded in `output_paths` keyed by
  prefixed name (`report__<name>`, `ephemeris__<name>`,
  `contact__<name>`).
- **Append-only durability.** The manifest is written entry-by-entry
  with `fsync` after each line, so a partial bundle from a `Ctrl-C`'d
  sweep is still parseable.

A receiving wrapper does not need any `gmat-sweep` API beyond
[`Manifest`][gmat_sweep.Manifest] and the per-run Parquet reads — those
are pandas calls — but using the high-level [`Sweep`][gmat_sweep.Sweep]
class is also fine when the consumer is in Python.

### Walking a bundle from outside

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
from gmat_sweep import Manifest, ManifestEntry


def iter_runs(
    bundle: Path,
    *,
    where: dict[str, object] | None = None,
    kind: str = "report",
) -> Iterator[tuple[ManifestEntry, pd.DataFrame]]:
    """Yield (entry, frame) for every run in ``bundle`` whose overrides match.

    ``where`` is a dict of dotted-path → expected value; runs whose
    ``overrides`` superset this dict are returned. ``None`` returns every
    successful run. ``kind`` is one of ``"report"``, ``"ephemeris"``,
    ``"contact"``.
    """
    manifest = Manifest.load(bundle / "manifest.jsonl")
    for entry in manifest.entries:
        if entry.status != "ok":
            continue
        if where and not all(entry.overrides.get(k) == v for k, v in where.items()):
            continue
        for prefixed_name, parquet_path in entry.output_paths.items():
            if not prefixed_name.startswith(f"{kind}__"):
                continue
            path = parquet_path if parquet_path.is_absolute() else bundle / parquet_path
            yield entry, pd.read_parquet(path)
```

This gives an external-tool wrapper exactly what it needs from a
sweep's bundle: filter by `overrides` to pick runs of interest, by
`kind` to pick the GMAT output type, and by `status` to skip failures —
then read each Parquet on demand. The consumer never has to know about
`Sweep`, `from_manifest`, or the aggregator: it just sees the bundle.

A typical usage from outside:

```python
from pathlib import Path

bundle = Path("/handoff/sma-scan-2026q2")
for entry, df in iter_runs(bundle, where={"Sat.SMA": 7100.0}):
    print(entry.run_id, entry.overrides, df.shape)
```

### Extending the bundle

The on-disk layout is the contract; nothing stops a consumer from
adding companion files alongside `manifest.jsonl` (a `README.md`, an
input summary, derived plots) before handing the bundle off. The
`script_sha256` field in the manifest header gives a downstream
verifier a way to check that the script in the bundle is the script
the manifest was actually written against.

## Pattern 4 — Archival deposit (Zenodo / JOSS)

Pattern 3 builds a hand-rolled wrapper around the on-disk bundle. When
the consumer is an *archival* deposit — a Zenodo record, JOSS
supplementary material, or an internal handoff that needs to survive
filesystem churn — `gmat-sweep` ships a packager that produces a single
self-describing `.zip` directly:
[`Sweep.archive`][gmat_sweep.Sweep.archive] and the matching
`gmat-sweep archive` CLI subcommand.

The bundle layout is:

```
bundle.zip
├── README.md            generated reproduce recipe + manifest summary
├── script/<name>        copy of the .script the manifest references
├── manifest.jsonl       paths rewritten to be bundle-relative
├── MANIFEST.hash        sha256sum-c compatible (every other member)
└── runs/run-<id>/...    per-run Parquet outputs (and worker.log if requested)
```

Two things are worth calling out about this layout, because they're what
make the bundle re-runnable on a fresh machine:

1. **Manifest paths are rewritten on the way in.** The on-disk manifest
   stores absolute `output_paths` pointing at the sweep's per-run
   directories. The bundled manifest carries `runs/run-<id>/<basename>`
   relative paths instead, which the aggregator resolves against the
   unzip directory without further plumbing.
2. **The bundle is byte-deterministic.** Two archives of the same
   manifest are identical at the byte level — fixed `ZipInfo` timestamps,
   sorted entries, stable hashes. Re-uploading to Zenodo from a
   different machine produces the same record.

### From a finished sweep

```python
from pathlib import Path

from gmat_sweep import Sweep
from gmat_sweep.backends import LocalJoblibPool

with LocalJoblibPool() as pool:
    sweep = Sweep.from_manifest(
        Path("./out/manifest.jsonl"),
        Path("./mission.script"),
        backend=pool,
    )
bundle = sweep.archive(Path("./sma-scan-2026q2.zip"))
```

`Sweep.archive` returns the resolved path to the `.zip`. By default it
drops every per-run `worker.log` (and sets the manifest's `log_path`
field to `null`) so the archive stays small. Pass `include_logs=True`
when you want them — useful for failure-analysis deposits where the
worker traces are part of the record.

### From the CLI

```bash
gmat-sweep archive ./out/manifest.jsonl \
    --script ./mission.script \
    --out ./sma-scan-2026q2.zip
```

Exits 2 if the script's canonical SHA-256 disagrees with the manifest's
recorded hash; pass `--allow-script-drift` to proceed anyway (the
bundle still records the manifest's original hash, so a downstream
verifier can spot the drift).

### Reproducing the sweep from a bundle

The generated `README.md` documents this for whoever downloads the
deposit, but the steps boil down to:

```bash
unzip sma-scan-2026q2.zip -d sma-scan-2026q2/
cd sma-scan-2026q2/

# Verify integrity:
sha256sum -c MANIFEST.hash

# One-line summary:
gmat-sweep show manifest.jsonl

# Re-run only the runs that failed or are missing on disk:
gmat-sweep resume manifest.jsonl --script script/mission.script
```

For an all-`ok` bundle, `gmat-sweep resume` is a no-op on the run side —
nothing failed, nothing's missing — and the aggregator reads the
existing per-run Parquets. The resulting DataFrame is bit-equal to the
one the original sweep produced.

## Where to from here

- The full set of fields and shapes the manifest carries is documented
  on the [Manifest schema](manifest-schema.md) page; that's the
  reference any cross-tool validator should be reading.
- The aggregator's three entry points
  ([`lazy_multiindex`][gmat_sweep.lazy_multiindex],
  [`lazy_ephemerides`][gmat_sweep.lazy_ephemerides],
  [`lazy_contacts`][gmat_sweep.lazy_contacts]) and the per-output index
  shapes are covered on [Aggregating sweep
  outputs](aggregation.md). Reach for those when a downstream consumer
  needs more than the default frame shape.
- For wiring a sweep onto cluster infrastructure rather than wrapping
  its outputs, see the [Cluster recipes](recipes/index.md).
