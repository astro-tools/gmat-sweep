"""Provenance bundle: pack a finished sweep into a self-describing ``.zip``.

The single internal entry point is :func:`_archive_sweep`. :meth:`Sweep.archive`
calls it with state from the in-memory sweep; the ``gmat-sweep archive`` CLI
calls it with state loaded from a manifest on disk. Both paths produce the
same byte-equal bundle, which is the contract the round-trip test asserts.

Bundle layout
-------------
::

    bundle.zip
    |-- README.md            generated reproduce recipe + manifest summary
    |-- script/<name>        copy of the .script the manifest references
    |-- manifest.jsonl       rewritten so output_paths/log_path are bundle-relative
    |-- MANIFEST.hash        sha256sum-format file covering every other member
    `-- runs/run-<id>/...    per-run Parquet files (and worker.log if requested)

The bundled manifest's ``output_paths`` and ``log_path`` are rewritten to
``runs/run-<id>/<basename>`` form so :func:`gmat_sweep.aggregate.lazy_multiindex`
resolves them against the unzip directory without further ceremony.
``include_logs=False`` (the default) drops every per-run ``worker.log`` from
both the bundle and the manifest's ``log_path`` field.

Determinism
-----------
Members are written in a fixed order with a frozen ``date_time`` and ``0o644``
external attrs so two archives of the same manifest compare byte-equal. This
makes the CLI ↔ API equivalence test possible and keeps Zenodo re-uploads
idempotent.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest, ManifestEntry, canonical_script_sha256

__all__ = ["_archive_sweep"]


_README_NAME = "README.md"
_MANIFEST_NAME = "manifest.jsonl"
_HASH_NAME = "MANIFEST.hash"
_SCRIPT_DIR = "script"
_RUNS_DIR = "runs"
_WORKER_LOG_NAME = "worker.log"

# Fixed timestamp for every ZipInfo so byte-equal archives are reproducible.
# Any constant in zip's 1980-2107 range works; this one is the project epoch.
_FROZEN_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_UNIX_FILE_MODE = 0o644 << 16


def _archive_sweep(
    *,
    manifest: Manifest,
    output_dir: Path,
    script_path: Path,
    out: Path,
    include_logs: bool,
    sweep_version: str,
    allow_script_drift: bool = False,
) -> Path:
    """Pack a finished sweep into ``out`` as a ``.zip`` and return the resolved path.

    ``manifest`` is the source of truth for which runs to include and what
    their per-run Parquet paths are. ``output_dir`` is the sweep root those
    paths resolve against; ``script_path`` is the ``.script`` whose canonical
    hash must match ``manifest.script_sha256`` (unless ``allow_script_drift``
    is set, mirroring :meth:`Sweep.from_manifest`). ``sweep_version`` is
    written into the generated README so the deposit records the packager
    version separately from ``manifest.gmat_sweep_version``.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    current_sha = canonical_script_sha256(script_path)
    if current_sha != manifest.script_sha256 and not allow_script_drift:
        raise SweepConfigError(
            f"script hash mismatch for {script_path}: "
            f"manifest={manifest.script_sha256}, current={current_sha}"
        )

    script_member = f"{_SCRIPT_DIR}/{script_path.name}"
    rewritten = _rewrite_entries_to_relative(manifest.entries, output_dir, include_logs)
    bundled_manifest = _clone_manifest_with_entries(manifest, rewritten)

    members: list[tuple[str, bytes]] = []
    members.append((script_member, script_path.read_bytes()))
    members.append((_MANIFEST_NAME, _serialise_manifest(bundled_manifest)))
    members.extend(_collect_run_members(manifest.entries, output_dir, include_logs))
    # README references the bundled manifest's contents, so build it after the
    # rewrite. Hash file references every other member, so build it last.
    readme_bytes = _render_readme(
        bundled_manifest, script_member, sweep_version, include_logs=include_logs
    ).encode("utf-8")
    members.append((_README_NAME, readme_bytes))
    members.append((_HASH_NAME, _render_hash_file(members)))

    members.sort(key=lambda m: m[0])

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, data in members:
            info = zipfile.ZipInfo(filename=name, date_time=_FROZEN_TIMESTAMP)
            info.external_attr = _UNIX_FILE_MODE
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)

    return out


def _rewrite_entries_to_relative(
    entries: Iterable[ManifestEntry],
    output_dir: Path,
    include_logs: bool,
) -> list[ManifestEntry]:
    """Rewrite each entry's ``output_paths`` and ``log_path`` to bundle-relative form.

    Per-run Parquets land at ``runs/run-<id>/<basename>``; the worker log
    lands at the same prefix but is dropped (entry ``log_path`` set to
    ``None``) when ``include_logs`` is ``False``.
    """
    rewritten: list[ManifestEntry] = []
    for entry in entries:
        run_dir = f"{_RUNS_DIR}/run-{entry.run_id}"
        new_paths: dict[str, Path] = {}
        for key, raw in entry.output_paths.items():
            new_paths[key] = Path(f"{run_dir}/{Path(raw).name}")
        if include_logs and entry.log_path is not None:
            new_log: Path | None = Path(f"{run_dir}/{_WORKER_LOG_NAME}")
        else:
            new_log = None
        rewritten.append(replace(entry, output_paths=new_paths, log_path=new_log))
    return rewritten


def _clone_manifest_with_entries(manifest: Manifest, entries: list[ManifestEntry]) -> Manifest:
    """Return a Manifest carrying the rewritten entries and the original header."""
    return Manifest(
        script_sha256=manifest.script_sha256,
        gmat_sweep_version=manifest.gmat_sweep_version,
        gmat_run_version=manifest.gmat_run_version,
        gmat_install_version=manifest.gmat_install_version,
        python_version=manifest.python_version,
        os_platform=manifest.os_platform,
        sweep_seed=manifest.sweep_seed,
        parameter_spec=dict(manifest.parameter_spec),
        run_count=manifest.run_count,
        backend=manifest.backend,
        schema_version=manifest.schema_version,
        entries=entries,
    )


def _serialise_manifest(manifest: Manifest) -> bytes:
    """Serialise a manifest header + entries as one trailing-newline-terminated JSONL blob."""
    lines = [json.dumps(manifest._header_dict(), sort_keys=True)]
    lines.extend(json.dumps(e.to_dict(), sort_keys=True) for e in manifest.entries)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _collect_run_members(
    entries: Iterable[ManifestEntry],
    output_dir: Path,
    include_logs: bool,
) -> list[tuple[str, bytes]]:
    """Read every per-run Parquet (and optional worker log) referenced by ``entries``."""
    members: list[tuple[str, bytes]] = []
    for entry in entries:
        if entry.status != "ok":
            # Failed/skipped runs have no Parquets to bundle. Their stderr and
            # overrides survive in the manifest, which is enough for resume.
            if include_logs and entry.log_path is not None:
                log_path = entry.log_path
                if not log_path.is_absolute():
                    log_path = output_dir / log_path
                if log_path.is_file():
                    members.append(
                        (
                            f"{_RUNS_DIR}/run-{entry.run_id}/{_WORKER_LOG_NAME}",
                            log_path.read_bytes(),
                        )
                    )
            continue
        for _key, raw in entry.output_paths.items():
            src = raw if raw.is_absolute() else output_dir / raw
            if not src.is_file():
                raise SweepConfigError(
                    f"manifest entry for run_id={entry.run_id} references missing "
                    f"output file: {src}"
                )
            members.append((f"{_RUNS_DIR}/run-{entry.run_id}/{src.name}", src.read_bytes()))
        if include_logs and entry.log_path is not None:
            log_path = entry.log_path
            if not log_path.is_absolute():
                log_path = output_dir / log_path
            if log_path.is_file():
                members.append(
                    (
                        f"{_RUNS_DIR}/run-{entry.run_id}/{_WORKER_LOG_NAME}",
                        log_path.read_bytes(),
                    )
                )
    return members


def _render_hash_file(members: list[tuple[str, bytes]]) -> bytes:
    """Render a ``sha256sum`` -compatible file covering every member except itself.

    Format: ``<hex-digest>  <relative-path>\\n`` per line, sorted by path. The
    file itself is excluded so a downstream ``sha256sum -c MANIFEST.hash``
    from the unzip directory verifies cleanly.
    """
    lines = [
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in members
        if name != _HASH_NAME
    ]
    lines.sort()
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_readme(
    manifest: Manifest,
    script_member: str,
    sweep_version: str,
    *,
    include_logs: bool,
) -> str:
    """Generate the bundle's README — reproduce recipe, summary, version stamp."""
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    duration_s = 0.0
    for entry in manifest.entries:
        counts[entry.status] += 1
        duration_s += entry.duration_s
    breakdown = ", ".join(f"{counts[k]} {k}" for k in ("ok", "failed", "skipped") if counts[k])
    if not breakdown:
        breakdown = "0 runs"

    log_note = (
        "Per-run worker logs are bundled under each `runs/run-<id>/worker.log`."
        if include_logs
        else "Per-run worker logs were excluded; pass `--include-logs` "
        "(or `include_logs=True`) at archive time to bundle them."
    )

    lines = [
        f"# Sweep archive — {len(manifest.entries)} runs",
        "",
        "This bundle was produced by `gmat-sweep archive`. It contains every input "
        "and output needed to re-aggregate or resume the sweep on a new machine.",
        "",
        "## Summary",
        "",
        f"- **Runs:** {len(manifest.entries)} ({breakdown})",
        f"- **Total runtime:** {duration_s:.2f} s",
        f"- **Script SHA-256:** `{manifest.script_sha256}`",
        f"- **gmat-sweep (sweep):** {manifest.gmat_sweep_version}",
        f"- **gmat-sweep (archive):** {sweep_version}",
        f"- **gmat-run:** {manifest.gmat_run_version}",
        f"- **GMAT install:** {manifest.gmat_install_version}",
        f"- **Python:** {manifest.python_version}",
        f"- **OS at sweep time:** {manifest.os_platform}",
        f"- **Backend:** {manifest.backend}",
        "",
        "## Layout",
        "",
        "```",
        "bundle/",
        "|-- README.md           (this file)",
        f"|-- {script_member}",
        "|-- manifest.jsonl      (paths rewritten to be bundle-relative)",
        "|-- MANIFEST.hash       (sha256sum -c compatible)",
        "`-- runs/run-<id>/...   (per-run Parquet outputs)",
        "```",
        "",
        log_note,
        "",
        "## Reproducing the sweep",
        "",
        "Unzip the bundle, then from the unzip directory:",
        "",
        "```bash",
        "# One-line summary of the manifest:",
        "gmat-sweep show manifest.jsonl",
        "",
        "# Re-run only the failed and missing runs:",
        f"gmat-sweep resume manifest.jsonl --script {script_member}",
        "```",
        "",
        "Or from Python:",
        "",
        "```python",
        "from pathlib import Path",
        "from gmat_sweep import Sweep",
        "from gmat_sweep.backends import LocalJoblibPool",
        "",
        "with LocalJoblibPool() as pool:",
        "    sweep = Sweep.from_manifest(",
        '        Path("manifest.jsonl"),',
        f'        Path("{script_member}"),',
        "        backend=pool,",
        "    ).resume()",
        "    df = sweep.to_dataframe()",
        "```",
        "",
        "## Verifying integrity",
        "",
        "From the unzip directory:",
        "",
        "```bash",
        "sha256sum -c MANIFEST.hash",
        "```",
    ]
    return "\n".join(lines) + "\n"
