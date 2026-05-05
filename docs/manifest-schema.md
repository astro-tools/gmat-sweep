# Manifest schema

Every sweep writes a `manifest.jsonl` next to its per-run output
directories. It is the durable record of *what was run*, *with what
overrides*, *and how it turned out* — designed so a mid-sweep `Ctrl-C`
leaves a parseable file and so a future resume flow can rebuild the
unfinished tail of the sweep.

## On-disk format

The file is **JSON Lines**:

- **Line 1** — one JSON object: the **header**, written once by
  [`Manifest.save()`][gmat_sweep.Manifest.save] and never rewritten.
- **Lines 2..N** — one JSON object per run: a
  [`ManifestEntry`][gmat_sweep.ManifestEntry], appended one at a time by
  [`Manifest.append_entry()`][gmat_sweep.Manifest.append_entry] with `fsync`
  after each write.

Each line is a single complete JSON document with `sort_keys=True`, so the
file is bit-for-bit deterministic across processes and trivially
`grep`-friendly. The trailing newline on the final line is significant —
[`Manifest.load()`][gmat_sweep.Manifest.load] tolerates a single torn last
line by dropping it (a partial write loses one entry; the rest of the file
parses cleanly).

The header's `run_count` field is the *expected* run count at sweep launch
time. It is **not rewritten** as entries arrive, so the on-disk header may
report more runs than the file actually contains during and after a
`Ctrl-C`'d sweep. Read `len(manifest.entries)` for the actual count.

## Header fields

```json
{
  "schema_version":      1,
  "script_sha256":       "<hex>",
  "gmat_sweep_version":  "0.2.0",
  "gmat_run_version":    "0.4.x",
  "gmat_install_version": "R2026a",
  "python_version":      "3.12.x",
  "os_platform":         "Linux-6.x.x-...",
  "sweep_seed":          null,
  "parameter_spec":      { "_kind": "grid", "<dotted-path>": [<value>, ...], ... },
  "run_count":           <int>
}
```

| Field                  | What it carries                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------|
| `schema_version`       | Manifest schema version. `1` from v0.2 onward. v0.1 manifests omit the field entirely; [`Manifest.load`][gmat_sweep.Manifest.load] treats the absence as `1` for backwards compatibility. See [Compatibility policy](#compatibility-policy). |
| `script_sha256`        | SHA-256 of the `.script` after line-ending and trailing-newline normalisation. See below.        |
| `gmat_sweep_version`   | `gmat_sweep.__version__` at sweep time.                                                          |
| `gmat_run_version`     | `gmat_run.__version__`, or `"unknown"` if `gmat_run` is not importable.                          |
| `gmat_install_version` | The discovered GMAT install's version string (e.g. `"R2026a"`), or `"unknown"`.                  |
| `python_version`       | `platform.python_version()`.                                                                     |
| `os_platform`          | `platform.platform()` — same string `gmat-run` records.                                          |
| `sweep_seed`           | The seed passed to [`sweep(seed=...)`][gmat_sweep.sweep], [`monte_carlo(seed=...)`][gmat_sweep.monte_carlo], or [`latin_hypercube(seed=...)`][gmat_sweep.latin_hypercube], or `null`. |
| `parameter_spec`       | The run set the sweep expanded, tagged with a `_kind` discriminator. One of four shapes — see [`parameter_spec` shapes](#parameter_spec-shapes) below. |
| `run_count`            | The number of runs in the sweep at launch.                                                       |

### `parameter_spec` shapes

The `_kind` discriminator is one of four values, each with its own
payload shape:

| `_kind`           | Payload (alongside `_kind`) | Written by |
|-------------------|------------------------------|------------|
| `"grid"`          | `{"<dotted-path>": [<value>, ...], ...}` — the materialised cartesian product, every iterable expanded to a list, keys preserved verbatim. | [`sweep(grid=...)`][gmat_sweep.sweep] |
| `"explicit"`      | `{"columns": [<str>, ...], "rows": [[<value>, ...], ...]}` — the input DataFrame as column order plus row-major values. | [`sweep(samples=...)`][gmat_sweep.sweep] |
| `"monte_carlo"`   | `{"perturb": {<dotted-path>: <serialised dist>, ...}, "n": <int>, "seed": <int> \| null}` — the distribution descriptors plus the parent seed used to derive per-parameter sub-seeds. | [`monte_carlo`][gmat_sweep.monte_carlo] |
| `"latin_hypercube"` | Same shape as `"monte_carlo"` — the seed is forwarded to [`scipy.stats.qmc.LatinHypercube`][scipy-lh]. | [`latin_hypercube`][gmat_sweep.latin_hypercube] |

[scipy-lh]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html

See [Parameter spec](parameter-spec.md) for the user-facing semantics of
each shape and how to reconstruct the run set from a manifest.

#### v0.1 untagged grid headers

Manifests written by `gmat_sweep` 0.1.x omit `_kind` on grid sweeps and
present `parameter_spec` as the bare materialised grid:

```json
{ "parameter_spec": { "<dotted-path>": [<value>, ...], ... } }
```

These keep loading under v0.2+: the dispatch in [`Sweep.from_manifest`][gmat_sweep.Sweep.from_manifest]
treats a missing `_kind` as `"grid"`. New sweeps always tag the shape.

### Canonical script hash

`script_sha256` is computed by
[`canonical_script_sha256()`][gmat_sweep.canonical_script_sha256], which
normalises line endings (`\r\n` and lone `\r` → `\n`) and trims trailing
newlines to exactly one before hashing. Two clones of the same script
checked out under different line-ending settings produce identical hashes.

## Entry fields

```json
{
  "run_id":        0,
  "overrides":     { "<dotted-path>": <value>, ... },
  "status":        "ok" | "failed" | "skipped",
  "output_paths":  { "<report_name>": "<path>", ... },
  "started_at":    "<ISO-8601 datetime>",
  "ended_at":      "<ISO-8601 datetime>",
  "duration_s":    1.234,
  "stderr":        null,
  "log_path":      "<path>" | null
}
```

| Field          | What it carries                                                                                              |
|----------------|--------------------------------------------------------------------------------------------------------------|
| `run_id`       | Sequential integer assigned at grid-expansion time, starting at `0`. Unique within a sweep.                  |
| `overrides`    | The override dict applied for this run — exactly the slice of the grid that produced it.                     |
| `status`       | One of `"ok"`, `"failed"`, `"skipped"`. v0.1 only emits `"ok"` and `"failed"`.                                |
| `output_paths` | Map from the prefixed output basename (`report__<name>`, `ephemeris__<name>`, `contact__<name>`) to the per-run Parquet path. Empty `{}` for non-`ok` runs. The prefix encodes the GMAT output kind so [`lazy_multiindex`][gmat_sweep.lazy_multiindex] / [`lazy_ephemerides`][gmat_sweep.lazy_ephemerides] / [`lazy_contacts`][gmat_sweep.lazy_contacts] can dispatch without reading the file. |
| `started_at`   | UTC `datetime` the worker began this run, ISO-8601 with tz offset.                                           |
| `ended_at`     | UTC `datetime` the worker returned its outcome, ISO-8601.                                                    |
| `duration_s`   | `(ended_at - started_at).total_seconds()`. Computed once on the worker side; the three timing fields cannot disagree. |
| `stderr`       | `null` for successful runs. For failed runs: the formatted Python traceback, optionally followed by the captured GMAT engine log. |
| `log_path`     | Path to the worker log file (`worker.log` under the per-run output directory), or `null`. Present whether the run succeeded or failed. |

### `output_paths` invariant

For `status == "ok"` entries, `output_paths` is non-empty. Each key is
one of:

- `report__<name>` — a `ReportFile` resource named `<name>` in the script.
- `ephemeris__<name>` — an `EphemerisFile` resource (OEM, STK-TimePosVel,
  or SPK; the worker writes the parsed DataFrame either way).
- `contact__<name>` — a `ContactLocator` resource. The Parquet carries a
  fresh integer `interval_id` column (`0..K-1` per run) the aggregator
  uses as the secondary index.

A single sweep may produce any mix of the three kinds, and any number of
each. Whether a Parquet path is recorded as relative or absolute depends
on how the worker wrote it; the aggregator resolves relative paths
against the sweep's `output_dir`.

## Loading a manifest back

```python
from pathlib import Path
from gmat_sweep import Manifest

manifest = Manifest.load(Path("./sweep/manifest.jsonl"))
print(manifest.script_sha256, manifest.run_count, len(manifest.entries))

# Find runs that need attention:
failed_ids = manifest.find_failed()                       # [list of run_id]
missing_ids = manifest.find_missing(range(manifest.run_count))
```

## CLI summary

`gmat-sweep show` prints a one-line summary of an existing manifest
without re-running anything:

```bash
$ gmat-sweep show ./sweep/manifest.jsonl
6 runs (5 ok, 1 failed) in 12.34 s — output: sweep
```

## Append-only invariant

The manifest is written **append-only with fsync after every entry**:

- The header is written once, then never touched.
- Each [`Manifest.append_entry()`][gmat_sweep.Manifest.append_entry] call
  writes one line and `fsync`s the file (and, on POSIX, the parent
  directory) before returning.

A `Ctrl-C`, OOM kill, or `kill -9` can lose only the in-flight write —
every entry that returned successfully from `append_entry` is durable.
[`Manifest.load()`][gmat_sweep.Manifest.load] silently tolerates a single
torn last line; anything more corrupted raises
[`ManifestCorruptError`][gmat_sweep.ManifestCorruptError] with the offending
file's path attached.

## Last-wins merge on load

A resumed run appends a new entry with the same `run_id` as the
original failed entry, so the on-disk file may carry two (or more)
lines for that `run_id`. [`Manifest.load`][gmat_sweep.Manifest.load]
folds duplicate `run_id`s last-wins: the latest entry's content
survives, kept in the position of the first occurrence. The
in-memory `entries` list therefore has exactly one entry per
`run_id`, and [`find_failed`][gmat_sweep.Manifest.find_failed] reflects
the latest status. See [Resume](resume.md) for the resume flow that
relies on this.

## Compatibility policy

The on-disk shape is frozen as `schema_version=1` from `gmat_sweep` 0.2
onward. The exposed constant
[`gmat_sweep.MANIFEST_SCHEMA_VERSION`][gmat_sweep.MANIFEST_SCHEMA_VERSION]
is what the running `gmat-sweep` writes and the maximum it accepts on
load.

**Read rules.**

- A manifest with `schema_version <= MANIFEST_SCHEMA_VERSION` loads. A
  missing `schema_version` is treated as `1` for v0.1 backwards
  compatibility.
- A manifest with `schema_version > MANIFEST_SCHEMA_VERSION` is rejected
  with [`ManifestCorruptError`][gmat_sweep.ManifestCorruptError]: the
  reader is older than the writer and may have lost or changed semantics
  on fields the manifest carries.
- Unknown extra header fields are silently dropped on load. Older
  `gmat-sweep` versions can therefore read manifests written by newer
  versions whenever the new fields are purely additive.

**When to bump `schema_version`.**

| Change | Bump required? |
|--------|----------------|
| Adding a new header field | No (additive — older readers ignore it). |
| Adding a new per-entry field with a documented default | No (older readers ignore it; new readers fall back to the default when reading older manifests). |
| Removing a header or per-entry field | Yes. |
| Changing the semantics of an existing field, even if the JSON shape is unchanged | Yes. |
| Changing the JSON shape of an existing field (e.g. flat to nested) | Yes. |
| Adding a new `_kind` value to `parameter_spec` | No (additive — older readers will reject the unknown kind at dispatch time, which is the correct behavior; the manifest itself remains parseable). |

A `schema_version` bump is a coordinated change: the writer side
emits the new value and the reader side learns to interpret the new
shape. Older `gmat-sweep` versions stop accepting bumped manifests
on the read side, which is the point of the version field.
