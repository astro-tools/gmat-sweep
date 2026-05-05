# Resume

A long sweep may produce a few failed runs (a flaky GMAT setting, an OS
hiccup, a `Ctrl-C` halfway through the queue). Rather than re-running
everything, point [`Sweep.from_manifest`][gmat_sweep.Sweep.from_manifest]
at the existing `manifest.jsonl` and call
[`Sweep.resume`][gmat_sweep.Sweep.resume] — only the failed and
never-recorded runs are re-submitted. Successful runs' Parquet files are
read from disk as-is.

## When to use it

- The original sweep was killed with `Ctrl-C` partway through. Some runs
  finished cleanly and their entries are on disk; the rest never started.
- A handful of runs failed for a reason you've now fixed (a setting in
  the script, a perturb bound, an environmental issue) and you want to
  rerun only those without re-doing the successful ones.
- The original Monte Carlo or Latin hypercube draw set should remain
  unchanged: resumed runs **must** sample bit-equal values to the
  originals. The resume flow re-derives per-run sub-seeds from the
  manifest's `sweep_seed`, so this is true by construction.

If the script's canonical hash has changed since the original sweep,
resume refuses by default — the old successful runs and the reruns
would have loaded different scripts and the aggregated DataFrame would
mix them. See [Script drift](#script-drift) below for the escape
hatch.

## How to use it

```python
from pathlib import Path
from gmat_sweep import Sweep
from gmat_sweep.backends.joblib import LocalJoblibPool

with LocalJoblibPool(workers=4) as pool:
    df = (
        Sweep.from_manifest(
            "./sweep/manifest.jsonl",
            "mission.script",
            backend=pool,
        )
        .resume()
        .to_dataframe()
    )
```

The returned DataFrame has the same shape as a fresh
[`sweep()`][gmat_sweep.sweep] call: `(run_id, time)`-MultiIndexed, with
one row per `(run, time-step)` pair, plus a `__status` column.

`from_manifest` requires:

- An existing `manifest.jsonl` whose parent directory still exists on
  disk — the successful runs' Parquet files are read from there.
- The original `.script`, whose canonical SHA-256 must match the
  manifest's `script_sha256` (see [Script drift](#script-drift)).
- A backend (a constructed [`Pool`][gmat_sweep.Pool]) — same
  contract as the regular [`Sweep`][gmat_sweep.Sweep] constructor.

## Last-wins entry semantics

The manifest is **append-only with `fsync` after every entry** (see
[Manifest schema](manifest-schema.md#append-only-invariant)). A resumed
run appends a new entry with the **same `run_id`** as the original
failed entry. The on-disk file then carries two lines for that
`run_id` — one `failed`, one `ok`.

[`Manifest.load`][gmat_sweep.Manifest.load] folds these last-wins:
when multiple entries share a `run_id`, the **last** occurrence's
content survives, kept in the position of the first occurrence. So:

- The in-memory `entries` list contains exactly one entry per `run_id`.
- [`find_failed`][gmat_sweep.Manifest.find_failed] only returns
  `run_id`s whose latest entry is `failed`.
- The on-disk file remains append-only — older entries are never
  rewritten or deleted, so a `Ctrl-C` during resume still leaves a
  parseable file.

A manifest from a sweep that never resumed has unique `run_id`s per
entry, so the dedup is a no-op there.

## Script drift

`from_manifest` recomputes
[`canonical_script_sha256`][gmat_sweep.canonical_script_sha256] over the
script you point it at and compares against the manifest's
`script_sha256`. A mismatch raises
[`SweepConfigError`][gmat_sweep.SweepConfigError] by default — silently
mixing old outputs with reruns that loaded a different script would
poison the aggregated DataFrame.

If you know the change is benign (e.g. a comment-only edit that the
canonical hash does *not* normalise) and you want to proceed anyway:

```python
Sweep.from_manifest(
    "./sweep/manifest.jsonl",
    "mission.script",
    backend=pool,
    allow_script_drift=True,
)
```

This produces a `RuntimeWarning` with both hashes and proceeds.

The canonical hash already normalises line endings and trailing
newlines (see
[Manifest schema § canonical script hash](manifest-schema.md#canonical-script-hash)),
so it does **not** trigger on whitespace-only diffs from a fresh
checkout.

## What runs and what doesn't

[`resume()`][gmat_sweep.Sweep.resume] submits the union of:

- `manifest.find_failed()` — entries whose latest status is `failed`.
- `manifest.find_missing(expected_run_ids)` — `run_id`s the rebuilt
  run iterable carries that have no entry on disk yet (the tail of a
  `Ctrl-C`'d sweep).

Successful runs are skipped; their Parquet outputs are reused from
their original `output_paths`. `skipped` runs (worker contract: the
worker explicitly chose not to execute) are also left alone — they
are not rerun.

## Limitations

- **Single-machine.** `script_path` and per-run `output_dir` are
  recorded as absolute paths in the manifest, so a manifest written on
  one machine cannot be resumed on another.
- **Python-only entry point today.** A `gmat-sweep resume` CLI
  subcommand is planned; the Python API above is the supported way in
  the meantime.
