# Manifest schema

Every sweep writes a `manifest.jsonl` next to its per-run output
directories. It is the durable record of *what was run*, *with what
overrides*, *and how it turned out* — designed so a mid-sweep `Ctrl-C`
leaves a parseable file and so the [resume flow](resume.md) can rebuild
the unfinished tail of the sweep.

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

The header's `run_count` field is the *expected* run count at the time of
first save, and is **frozen on disk for the life of the manifest** — the
header is append-only by design, so a torn last line costs exactly one
entry and the header stays valid. Consequences worth knowing:

- During a `Ctrl-C`'d sweep, `run_count` reports more runs than the file
  actually contains. Read `len(manifest.entries)` (or `manifest.find_missing(...)`)
  for the actual count and the gap.
- After [`Sweep.extend(n=K)`][gmat_sweep.Sweep.extend], `run_count` still
  reports the *original* size — it does not gain `K`. Read
  [`manifest.total_run_count`][gmat_sweep.Manifest.total_run_count]
  for the live total (original + extensions), or
  [`manifest.extension_run_count`][gmat_sweep.Manifest.extension_run_count]
  for just the extension delta.

## Header fields

```json
{
  "schema_version":      1,
  "script_sha256":       "<hex>",
  "gmat_sweep_version":  "<x.y.z>",
  "gmat_run_version":    "<x.y.z>",
  "gmat_install_version": "<R20yya>",
  "python_version":      "<x.y.z>",
  "os_platform":         "<platform.platform()>",
  "sweep_seed":          null,
  "parameter_spec":      { "_kind": "grid", "<dotted-path>": [<value>, ...], ... },
  "run_count":           <int>,
  "backend":             "<Pool subclass name>"
}
```

| Field                  | What it carries                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------|
| `schema_version`       | Manifest schema version. Currently `1`. Older manifests that omit the field are loaded as `1` for backwards compatibility. See [Compatibility policy](#compatibility-policy). |
| `script_sha256`        | SHA-256 of the `.script` after line-ending and trailing-newline normalisation. See below.        |
| `gmat_sweep_version`   | `gmat_sweep.__version__` at sweep time.                                                          |
| `gmat_run_version`     | `gmat_run.__version__`, or `"unknown"` if `gmat_run` is not importable.                          |
| `gmat_install_version` | The discovered GMAT install's version string (e.g. `"R2026a"`), or `"unknown"`.                  |
| `python_version`       | `platform.python_version()`.                                                                     |
| `os_platform`          | `platform.platform()` — same string `gmat-run` records.                                          |
| `sweep_seed`           | The seed passed to [`sweep(seed=...)`][gmat_sweep.sweep], [`monte_carlo(seed=...)`][gmat_sweep.monte_carlo], or [`latin_hypercube(seed=...)`][gmat_sweep.latin_hypercube], or `null`. |
| `parameter_spec`       | The run set the sweep expanded, tagged with a `_kind` discriminator. One of four shapes — see [`parameter_spec` shapes](#parameter_spec-shapes) below. |
| `run_count`            | The number of runs in the sweep at launch. Frozen on disk — does not change after [`Sweep.extend()`][gmat_sweep.Sweep.extend]; read [`Manifest.total_run_count`][gmat_sweep.Manifest.total_run_count] for the live total. |
| `backend`              | The execution backend's class name (`pool.__class__.__name__`) — e.g. `"LocalJoblibPool"`, `"DaskPool"`, `"RayPool"`, or any third-party `Pool` subclass. Optional on load: manifests written before this field landed report `"unknown"`. |

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

#### Untagged grid headers

Older manifests omit `_kind` on grid sweeps and present `parameter_spec`
as the bare materialised grid:

```json
{ "parameter_spec": { "<dotted-path>": [<value>, ...], ... } }
```

These keep loading: the dispatch in
[`Sweep.from_manifest`][gmat_sweep.Sweep.from_manifest] treats a missing
`_kind` as `"grid"`. New sweeps always tag the shape.

### Canonical script hash

`script_sha256` is computed by
[`canonical_script_sha256()`][gmat_sweep.canonical_script_sha256], which
normalises a leading UTF-8 byte-order mark (`﻿`), line endings
(`\r\n` and lone `\r` → `\n`), and trailing newlines (trimmed to exactly
one) before hashing. The same `.script` saved from a BOM-emitting
Windows editor and from a Linux editor without a BOM produces identical
hashes; same for two clones with different `core.autocrlf` settings.

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
| `status`       | One of `"ok"`, `"failed"`, `"skipped"`.                                                                      |
| `output_paths` | Map from the prefixed output basename (`report__<name>`, `ephemeris__<name>`, `contact__<name>`) to the per-run Parquet path. Empty `{}` for non-`ok` runs. The prefix encodes the GMAT output kind so [`lazy_multiindex`][gmat_sweep.lazy_multiindex] / [`lazy_ephemerides`][gmat_sweep.lazy_ephemerides] / [`lazy_contacts`][gmat_sweep.lazy_contacts] can dispatch without reading the file. |
| `started_at`   | UTC `datetime` the worker began this run, ISO-8601 with tz offset.                                           |
| `ended_at`     | UTC `datetime` the worker returned its outcome, ISO-8601.                                                    |
| `duration_s`   | Run duration in seconds, measured by the worker as a `time.monotonic` delta around the run body. Not equal to `(ended_at - started_at).total_seconds()` — measuring monotonically keeps `duration_s` non-negative across mid-run wall-clock corrections (NTP step), while `started_at` / `ended_at` remain wall-clock audit timestamps. |
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

[`Manifest.load`][gmat_sweep.Manifest.load] materialises every entry
into the returned manifest's `entries` list, deduplicated last-wins per
`run_id`. For tail-only operations on large manifests
(`gmat-sweep resume` against a 10k-run sweep, "what failed?" queries),
prefer the streaming primitives — they parse the file lazily and never
hold every entry in memory.

```python
from pathlib import Path
from gmat_sweep import Manifest

manifest_path = Path("./sweep/manifest.jsonl")

# Eager load: full entries list, deduplicated.
manifest = Manifest.load(manifest_path)
print(manifest.script_sha256, manifest.run_count, len(manifest.entries))

# Streaming tail-only scans (do not materialise the entry list):
failed_ids = Manifest.find_failed(manifest_path)
# Use total_run_count rather than the frozen header run_count when iterating
# expected ids on an extended manifest — see "Header fields" above.
missing_ids = Manifest.find_missing(manifest_path, range(manifest.total_run_count))

# Lazy iteration if you need each entry but not all at once:
for entry in Manifest.iter_entries(manifest_path):
    ...
```

## CLI summary

`gmat-sweep show` prints a one-line summary of an existing manifest
without re-running anything:

```bash
$ gmat-sweep show ./sweep/manifest.jsonl
6 runs (5 ok, 1 failed) in 12.34 s — output: sweep
```

## Append-only invariant

The manifest is written **append-only**:

- The header is written once, then never touched.
- Each [`Manifest.append_entry()`][gmat_sweep.Manifest.append_entry] call
  writes one line; whether the line is fsynced before the call returns
  depends on the manifest's [fsync cadence](#fsync-cadence-and-durability).

[`Manifest.load()`][gmat_sweep.Manifest.load] silently tolerates a single
torn last line; anything more corrupted raises
[`ManifestCorruptError`][gmat_sweep.ManifestCorruptError] with the offending
file's path attached, and a `line_number` attribute set to the 1-indexed
line that failed to parse (or `None` for whole-file failures such as an
empty file). `gmat-sweep show`'s error output surfaces both.

## Fsync cadence and durability

Two knobs on [`Manifest`][gmat_sweep.Manifest] (and forwarded by every
sweep-running entry point) control how often the manifest is fsynced:

| Knob | Default | Effect |
|------|---------|--------|
| `fsync_each` | `True` | Every appended entry is fsynced before `append_entry` returns. Strict per-run durability — a `Ctrl-C`, OOM kill, or `kill -9` can lose only the in-flight write. |
| `fsync_batch` | `50` | When `fsync_each=False`, the manifest is fsynced only every Nth entry (and once on [`Manifest.close()`][gmat_sweep.Manifest.close], called at end-of-sweep). |

The default (`fsync_each=True`) preserves the v0.3 strict-per-entry
behaviour. Opt into `fsync_each=False` when sub-second runs at large
counts make the per-entry fsync the dominant cost in the driver thread —
typical for 1000+ Monte Carlo or grid sweeps with cheap per-run work.

**Tradeoff.** With `fsync_each=False` and `fsync_batch=N`, a host crash
between fsync boundaries (power loss, kernel panic) can leave up to
`N - 1` recently-appended entries missing from the on-disk manifest.
The per-run Parquet files and the script hash are unaffected — the
[resume flow](resume.md) re-runs only the missing slice. `Ctrl-C`
mid-sweep deliberately skips the end-of-sweep `close()` so the same
recovery window applies; the resume flow handles the gap.

The CLI exposes the knob on every sweep-running subcommand as
`--fsync-each / --no-fsync-each` and `--fsync-batch N`. The Python API
accepts `fsync_each=` and `fsync_batch=` on
[`sweep`][gmat_sweep.sweep], [`monte_carlo`][gmat_sweep.monte_carlo],
[`latin_hypercube`][gmat_sweep.latin_hypercube], and
[`monte_carlo_extend`][gmat_sweep.monte_carlo_extend].

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

## Monte Carlo extensions

[`monte_carlo_extend()`][gmat_sweep.monte_carlo_extend] appends new
runs to an existing Monte Carlo manifest at `run_id` range
`[old_n, old_n + n)`. The header's `parameter_spec.n` is **not**
rewritten — it stays at the original sweep's size — and no new header
fields are added on disk. The cumulative count of extension runs is
recoverable from the entries themselves; the convenience accessor is:

```python
manifest = Manifest.load("./sweep/manifest.jsonl")
manifest.extension_run_count  # 0 for fresh sweeps; N after extend(n=N)
manifest.total_run_count      # original n + extension_run_count
```

`manifest.run_count` is the frozen header value (original size at first
save); `manifest.total_run_count` is the live total derived from the
entries, and is the right value to feed into
[`find_missing`][gmat_sweep.Manifest.find_missing] when iterating
expected run ids on an extended manifest.

The `_kind` of a Monte Carlo manifest stays `"monte_carlo"` after
extension; only Monte Carlo manifests support extension at all
(`latin_hypercube` and grid sweeps refuse — see
[Monte Carlo § Extending an existing sweep](monte-carlo.md#extending-an-existing-sweep)).

## Compatibility policy

The on-disk shape is frozen as `schema_version=1`. The exposed constant
[`gmat_sweep.MANIFEST_SCHEMA_VERSION`][gmat_sweep.MANIFEST_SCHEMA_VERSION]
is what the running `gmat-sweep` writes and the maximum it accepts on
load.

**Read rules.**

- A manifest with `schema_version <= MANIFEST_SCHEMA_VERSION` loads. A
  missing `schema_version` is treated as `1` for backwards compatibility
  with manifests written before the field was introduced.
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

**Migration ladder.** [`Manifest.load`][gmat_sweep.Manifest.load] routes
every header through an internal `_migrate_header(data, from_version)`
shim before constructing the in-memory manifest. Today the shim is a
pass-through for `v1 → v1`; the ladder exists so that when v2 ships,
the per-version migration step (renames, splits, default backfills)
lands in one place and v1 manifests keep loading unchanged. Major bumps
are one-shot migrations applied on read; minor additive fields do not
go through the shim.
