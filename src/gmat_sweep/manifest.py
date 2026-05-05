"""Reproducibility manifest: load/save, canonical script-hash, find_failed/find_missing helpers.

The on-disk format is JSON Lines with a single header object on line 1 and
one :class:`ManifestEntry` per subsequent line. The header is written once
by :meth:`Manifest.save` and is never rewritten — :meth:`Manifest.append_entry`
only appends, and ``run_count`` on disk may therefore lag the entries below
it. This is by design: a torn last line costs that one entry, the header
stays valid, and a mid-sweep ``Ctrl-C`` leaves a parseable file.

Resume merges entries last-wins per ``run_id``. The on-disk file is
append-only — a resumed run writes a new entry with the same ``run_id`` as
the original failed entry, so the file may carry two (or more) entries for
that ``run_id``. :meth:`Manifest.load` keeps only the *last* occurrence per
``run_id`` (preserving the position of the *first* occurrence), so the
in-memory ``entries`` list is deduplicated and the resumed status wins.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from gmat_sweep.errors import ManifestCorruptError
from gmat_sweep.spec import RunOutcome, RunStatus

__all__ = ["Manifest", "ManifestEntry", "canonical_script_sha256"]


def canonical_script_sha256(script_path: Path) -> str:
    """SHA-256 of the script file after line-ending and trailing-newline normalisation.

    Reads the file as bytes, decodes as UTF-8, replaces ``\\r\\n`` and
    lone ``\\r`` with ``\\n``, and ensures exactly one trailing ``\\n``.
    The hash is computed over the resulting UTF-8 bytes.
    """
    raw = script_path.read_bytes().decode("utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    canonical = text.rstrip("\n") + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ManifestEntry:
    """One run's record in the manifest — overrides, status, outputs, timing."""

    run_id: int
    overrides: dict[str, Any]
    status: RunStatus
    output_paths: dict[str, Path]
    started_at: datetime
    ended_at: datetime
    duration_s: float
    stderr: str | None
    log_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overrides": dict(self.overrides),
            "status": self.status,
            "output_paths": {k: str(v) for k, v in self.output_paths.items()},
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "duration_s": self.duration_s,
            "stderr": self.stderr,
            "log_path": None if self.log_path is None else str(self.log_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        return cls(
            run_id=int(data["run_id"]),
            overrides=dict(data["overrides"]),
            status=cast(RunStatus, data["status"]),
            output_paths={k: Path(v) for k, v in data["output_paths"].items()},
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]),
            duration_s=float(data["duration_s"]),
            stderr=None if data["stderr"] is None else str(data["stderr"]),
            log_path=None if data["log_path"] is None else Path(data["log_path"]),
        )

    @classmethod
    def from_outcome(
        cls,
        outcome: RunOutcome,
        *,
        overrides: dict[str, Any],
        log_path: Path | None = None,
    ) -> ManifestEntry:
        """Build a manifest entry from a worker :class:`RunOutcome`."""
        return cls(
            run_id=outcome.run_id,
            overrides=dict(overrides),
            status=outcome.status,
            output_paths=dict(outcome.output_paths),
            started_at=outcome.started_at,
            ended_at=outcome.ended_at,
            duration_s=outcome.duration_s,
            stderr=outcome.stderr,
            log_path=log_path,
        )


@dataclass(slots=True)
class Manifest:
    """Sweep manifest — header fingerprint plus a growing list of per-run entries."""

    script_sha256: str
    gmat_sweep_version: str
    gmat_run_version: str
    gmat_install_version: str
    python_version: str
    os_platform: str
    sweep_seed: int | None
    parameter_spec: dict[str, Any]
    run_count: int
    entries: list[ManifestEntry] = field(default_factory=list)
    _path: Path | None = field(default=None, init=False, repr=False, compare=False)

    def _header_dict(self) -> dict[str, Any]:
        return {
            "script_sha256": self.script_sha256,
            "gmat_sweep_version": self.gmat_sweep_version,
            "gmat_run_version": self.gmat_run_version,
            "gmat_install_version": self.gmat_install_version,
            "python_version": self.python_version,
            "os_platform": self.os_platform,
            "sweep_seed": self.sweep_seed,
            "parameter_spec": dict(self.parameter_spec),
            "run_count": self.run_count,
        }

    @classmethod
    def _header_from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            script_sha256=str(data["script_sha256"]),
            gmat_sweep_version=str(data["gmat_sweep_version"]),
            gmat_run_version=str(data["gmat_run_version"]),
            gmat_install_version=str(data["gmat_install_version"]),
            python_version=str(data["python_version"]),
            os_platform=str(data["os_platform"]),
            sweep_seed=None if data["sweep_seed"] is None else int(data["sweep_seed"]),
            parameter_spec=dict(data["parameter_spec"]),
            run_count=int(data["run_count"]),
            entries=[],
        )

    def save(self, path: Path) -> None:
        """Write header + all current entries as JSON Lines, fsyncing file and parent dir."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._header_dict(), sort_keys=True)]
        lines.extend(json.dumps(e.to_dict(), sort_keys=True) for e in self.entries)
        payload = "\n".join(lines) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(path.parent)
        self._path = path

    def append_entry(self, entry: ManifestEntry) -> None:
        """Append one entry to the bound file with fsync, and to the in-memory list."""
        if self._path is None:
            raise RuntimeError(
                "Manifest.append_entry requires a path — call save() or load() first."
            )
        line = json.dumps(entry.to_dict(), sort_keys=True) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.entries.append(entry)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load a manifest from disk, tolerating a single torn final line."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        if not raw:
            raise ManifestCorruptError("manifest file is empty", path)

        # Split on '\n' and drop the last element. For a clean
        # newline-terminated file this drops the trailing empty string; for a
        # torn write (missing trailing newline) it drops the partial line.
        complete_lines = raw.split("\n")[:-1]

        if not complete_lines:
            raise ManifestCorruptError("manifest header line missing", path)

        try:
            header_data = json.loads(complete_lines[0])
        except json.JSONDecodeError as exc:
            raise ManifestCorruptError(f"manifest header is not valid JSON: {exc}", path) from exc

        try:
            manifest = cls._header_from_dict(header_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestCorruptError(f"manifest header is missing fields: {exc}", path) from exc

        # Last-wins per run_id: a resumed run appends a new entry with the
        # same run_id as the original failed entry. We keep only the last
        # occurrence's content but preserve the position of the first
        # occurrence so unique-run_id manifests load in file order unchanged.
        by_run_id: dict[int, ManifestEntry] = {}
        for i, line in enumerate(complete_lines[1:], start=2):
            try:
                entry_data = json.loads(line)
                entry = ManifestEntry.from_dict(entry_data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ManifestCorruptError(
                    f"manifest entry on line {i} is malformed: {exc}", path
                ) from exc
            by_run_id[entry.run_id] = entry
        manifest.entries.extend(by_run_id.values())

        manifest._path = path
        return manifest

    def find_failed(self) -> list[int]:
        """Return run_ids of entries with status ``failed``, in arrival order."""
        return [e.run_id for e in self.entries if e.status == "failed"]

    def find_missing(self, expected_run_ids: Iterable[int]) -> list[int]:
        """Return run_ids in ``expected_run_ids`` with no entry recorded, in input order."""
        present = {e.run_id for e in self.entries}
        return [rid for rid in expected_run_ids if rid not in present]


def _fsync_dir(directory: Path) -> None:
    # Directory fsync ensures the new file's metadata (existence, size) is
    # durable across crash. Not supported on Windows; treat as a no-op there.
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (OSError, PermissionError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
