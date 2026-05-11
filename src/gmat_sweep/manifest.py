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

The on-disk shape is frozen as ``schema_version=1``; the canonical
reference is ``docs/manifest-schema.md``. Older manifests that omit
``schema_version`` are loaded as ``1`` for backwards compatibility. A
header carrying a ``schema_version`` greater than
:data:`MANIFEST_SCHEMA_VERSION` is rejected with
:class:`ManifestCorruptError` — the running ``gmat-sweep`` is older
than the manifest's writer and cannot parse it safely.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from gmat_sweep.errors import ManifestCorruptError
from gmat_sweep.spec import RunOutcome, RunStatus

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "ManifestEntry",
    "canonical_script_sha256",
]


MANIFEST_SCHEMA_VERSION: int = 1
"""On-disk manifest schema version this ``gmat-sweep`` writes and reads.

:meth:`Manifest.load` accepts any header whose ``schema_version`` is
``<= MANIFEST_SCHEMA_VERSION`` (a missing field is treated as ``1`` for
backwards compatibility with manifests written before the field was
introduced) and rejects anything greater. See
``docs/manifest-schema.md`` for the field-by-field contract and the
compatibility policy that governs future bumps.
"""


def canonical_script_sha256(script_path: Path) -> str:
    """SHA-256 of the script file after BOM, line-ending, and trailing-newline normalisation.

    Reads the file as bytes, decodes as UTF-8, strips a leading UTF-8
    byte-order mark (``\\ufeff``), replaces ``\\r\\n`` and lone ``\\r``
    with ``\\n``, and ensures exactly one trailing ``\\n``. The hash is
    computed over the resulting UTF-8 bytes — a script saved from a
    BOM-emitting Windows editor and the same script saved without a
    BOM produce identical hashes.
    """
    raw = script_path.read_bytes().decode("utf-8")
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
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

    def _repr_html_(self) -> str:
        import html as _html

        from gmat_sweep._repr_html import (
            build_kv_table,
            format_overrides_html,
            format_paths_html,
            summarise_stderr_html,
        )

        if self.log_path is None:
            log = "(none)"
        else:
            log = f"<code>{_html.escape(str(self.log_path))}</code>"
        rows: list[tuple[str, str]] = [
            ("run_id", str(self.run_id)),
            ("status", self.status),
            ("duration", f"{self.duration_s:.2f} s"),
            ("started_at", self.started_at.isoformat()),
            ("ended_at", self.ended_at.isoformat()),
            ("log_path", log),
            ("output_paths", format_paths_html(self.output_paths)),
            ("overrides", format_overrides_html(self.overrides)),
            ("stderr", summarise_stderr_html(self.stderr)),
        ]
        return build_kv_table(f"ManifestEntry run_id={self.run_id}", rows)


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
    backend: str = "unknown"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    entries: list[ManifestEntry] = field(default_factory=list)
    fsync_each: bool = field(default=True, compare=False)
    """When ``True`` (default), every :meth:`append_entry` fsyncs the file.

    Set to ``False`` and tune :attr:`fsync_batch` to amortise the fsync
    cost across a batch of entries — useful for sub-second runs at large
    counts where the per-entry fsync dominates the driver's time. The
    durability tradeoff is documented in ``docs/manifest-schema.md`` and
    on :meth:`append_entry`. Not serialised — this is a per-process
    knob, not part of the on-disk format.
    """
    fsync_batch: int = field(default=50, compare=False)
    """Fsync interval (in entries) when :attr:`fsync_each` is ``False``.

    With ``fsync_each=False``, :meth:`append_entry` fsyncs after every
    ``fsync_batch`` entries (and :meth:`close` fsyncs the tail). Has no
    effect when ``fsync_each=True``.
    """
    _path: Path | None = field(default=None, init=False, repr=False, compare=False)

    def _header_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "script_sha256": self.script_sha256,
            "gmat_sweep_version": self.gmat_sweep_version,
            "gmat_run_version": self.gmat_run_version,
            "gmat_install_version": self.gmat_install_version,
            "python_version": self.python_version,
            "os_platform": self.os_platform,
            "sweep_seed": self.sweep_seed,
            "parameter_spec": dict(self.parameter_spec),
            "run_count": self.run_count,
            "backend": self.backend,
        }

    @classmethod
    def _migrate_header(cls, data: dict[str, Any], from_version: int, path: Path) -> dict[str, Any]:
        """Migrate a header dict from ``from_version`` to :data:`MANIFEST_SCHEMA_VERSION`.

        Pass-through for ``from_version == MANIFEST_SCHEMA_VERSION``.
        The ladder exists so a future schema bump has a single place to
        register a per-version migration step — when v2 lands, the v1 →
        v2 transformation goes here and :meth:`load` keeps working
        unchanged on v1 manifests.

        ``path`` is forwarded only for error reporting if the ladder is
        ever asked for a migration that hasn't been written.
        """
        if from_version == MANIFEST_SCHEMA_VERSION:
            return data
        # Future v1 → v2 (and beyond) migrations chain here. Reaching this
        # branch with no path raises so an unimplemented schema bump fails
        # loudly rather than silently producing a malformed Manifest.
        raise ManifestCorruptError(
            f"no migration path from manifest schema_version={from_version} "
            f"to current version {MANIFEST_SCHEMA_VERSION}",
            path,
            line_number=1,
        )

    @classmethod
    def _header_from_dict(cls, data: dict[str, Any]) -> Manifest:
        # schema_version is absent on manifests written before the field was
        # introduced; default to 1 so they keep loading. Manifest.load() guards
        # against versions newer than this gmat-sweep supports before getting
        # here. ``backend`` is additive — manifests that omit it load with
        # ``backend == "unknown"``.
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
            backend=str(data.get("backend", "unknown")),
            schema_version=int(data.get("schema_version", 1)),
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
        """Append one entry to the bound file, and to the in-memory list.

        With :attr:`fsync_each` ``True`` (default), every appended entry
        is fsynced before this method returns — strict per-entry
        durability, matching the v0.3 behaviour. With ``fsync_each``
        ``False``, the file is fsynced only on the boundary set by
        :attr:`fsync_batch` (every Nth entry); the tail is not durable
        until :meth:`close` is called or the next batch boundary is
        crossed. A host crash between fsync boundaries can therefore
        lose up to ``fsync_batch - 1`` recently-appended entries — the
        Parquet outputs and the runs themselves are unaffected, and the
        resume flow re-runs only the missing slice.
        """
        if self._path is None:
            raise RuntimeError(
                "Manifest.append_entry requires a path — call save() or load() first."
            )
        line = json.dumps(entry.to_dict(), sort_keys=True) + "\n"
        # Compute durability boundary against the count *after* this append
        # so the very first entry doesn't trigger a fsync on the
        # ``len(entries) + 1 == 1`` edge case when ``fsync_batch`` divides 1.
        new_count = len(self.entries) + 1
        should_fsync = self.fsync_each or (new_count % max(1, self.fsync_batch) == 0)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if should_fsync:
                os.fsync(f.fileno())
        self.entries.append(entry)

    def close(self) -> None:
        """Fsync the manifest file and parent directory.

        Idempotent and a no-op when the manifest has no bound path
        (i.e. neither :meth:`save` nor :meth:`load` has been called).
        Sweeps that opt into :attr:`fsync_each` ``False`` should call
        this on successful completion so the trailing batch of entries
        becomes durable; a ``KeyboardInterrupt`` deliberately skips
        ``close()`` so the resume flow exercises the
        ``fsync_batch - 1``-entry recovery window.
        """
        if self._path is None:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            os.fsync(f.fileno())
        _fsync_dir(self._path.parent)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load a manifest from disk, tolerating a single torn final line.

        Materialises every entry into the returned :class:`Manifest`'s
        :attr:`entries` list, deduplicated last-wins per ``run_id``.
        For tail-only operations on large manifests prefer
        :meth:`iter_entries`, :meth:`find_failed`, or
        :meth:`find_missing` — they stream the file without holding
        every entry in memory.
        """
        path = Path(path)
        manifest, entries_iter = cls._load_header_and_entries_iter(path)
        # Last-wins per run_id: a resumed run appends a new entry with the
        # same run_id as the original failed entry. We keep only the last
        # occurrence's content but preserve the position of the first
        # occurrence so unique-run_id manifests load in file order unchanged.
        by_run_id: dict[int, ManifestEntry] = {}
        for entry in entries_iter:
            by_run_id[entry.run_id] = entry
        manifest.entries.extend(by_run_id.values())
        manifest._path = path
        return manifest

    @classmethod
    def iter_entries(cls, path: Path) -> Iterator[ManifestEntry]:
        """Stream parsed entries from disk, lazily, without folding duplicates.

        Yields one :class:`ManifestEntry` per non-header line in file
        order; tolerates a single torn final line the same way
        :meth:`load` does (a partial trailing line is silently dropped).
        Validates the header (schema version, required fields) before
        the first yield, raising :class:`ManifestCorruptError` with
        ``line_number=1`` on header failures and ``line_number=i`` on
        per-line failures.

        Use this for tail-only scans (per-``run_id`` last-status folds,
        membership checks) where holding every entry in memory would be
        wasteful — :meth:`find_failed` and :meth:`find_missing` are
        built on top. Use :meth:`load` when you need the
        deduplicated, fully-materialised entry list.
        """
        _, entries_iter = cls._load_header_and_entries_iter(Path(path))
        yield from entries_iter

    @classmethod
    def find_failed(cls, path: Path) -> list[int]:
        """Return ``run_id``s whose latest entry has ``status == "failed"``, in first-seen order.

        Streams ``path`` via :meth:`iter_entries` and folds per-``run_id``
        last-wins state into a small dict — does not materialise the
        full entry list. Result matches ``[e.run_id for e in
        Manifest.load(path).entries if e.status == "failed"]`` while
        running in O(entries) time and O(unique run_ids) memory.
        """
        last_status: dict[int, RunStatus] = {}
        first_seen_order: list[int] = []
        for entry in cls.iter_entries(Path(path)):
            if entry.run_id not in last_status:
                first_seen_order.append(entry.run_id)
            last_status[entry.run_id] = entry.status
        return [rid for rid in first_seen_order if last_status[rid] == "failed"]

    @classmethod
    def find_missing(cls, path: Path, expected_run_ids: Iterable[int]) -> list[int]:
        """Return ``run_id``s in ``expected_run_ids`` with no entry on disk, in input order."""
        present: set[int] = set()
        for entry in cls.iter_entries(Path(path)):
            present.add(entry.run_id)
        return [rid for rid in expected_run_ids if rid not in present]

    @classmethod
    def _load_header_and_entries_iter(cls, path: Path) -> tuple[Manifest, Iterator[ManifestEntry]]:
        """Parse and migrate the header, then return the manifest plus a lazy entry iterator.

        Shared core of :meth:`load` and :meth:`iter_entries` — the header
        parse and schema-version handling are identical; only the
        downstream consumption (dedup-into-memory vs. yield-as-you-go)
        differs. The entries iterator owns the open file handle and
        closes it on exhaustion.
        """
        path = Path(path)
        # Header is small; read it eagerly so schema validation can fail
        # before we hand the iterator to the caller. Entries are read
        # lazily by ``_iter_entries_from_open_file``.
        f = open(path, encoding="utf-8")  # noqa: SIM115 — closed by the entries iterator
        try:
            first_line = f.readline()
            if not first_line:
                raise ManifestCorruptError("manifest file is empty", path)
            if not first_line.endswith("\n"):
                # The only line in the file has no trailing \n — torn header.
                raise ManifestCorruptError("manifest header line missing", path, line_number=1)
            try:
                header_data = json.loads(first_line)
            except json.JSONDecodeError as exc:
                raise ManifestCorruptError(
                    f"manifest header is not valid JSON: {exc}", path, line_number=1
                ) from exc

            # v0.1 manifests omit schema_version; treat as 1. Anything strictly
            # greater than the running gmat-sweep's supported version is unparseable
            # by definition — newer schemas may have changed semantics on existing
            # fields, and we cannot tell from here which fields are still safe.
            try:
                schema_version = int(header_data.get("schema_version", 1))
            except (TypeError, ValueError) as exc:
                raise ManifestCorruptError(
                    f"manifest schema_version is not an integer: "
                    f"{header_data.get('schema_version')!r}",
                    path,
                    line_number=1,
                ) from exc
            if schema_version > MANIFEST_SCHEMA_VERSION:
                raise ManifestCorruptError(
                    f"manifest schema_version={schema_version} "
                    f"is newer than this gmat-sweep supports",
                    path,
                    line_number=1,
                )

            migrated = cls._migrate_header(header_data, schema_version, path)

            try:
                manifest = cls._header_from_dict(migrated)
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestCorruptError(
                    f"manifest header is missing fields: {exc}", path, line_number=1
                ) from exc
        except BaseException:
            f.close()
            raise

        return manifest, cls._iter_entries_from_open_file(f, path)

    @staticmethod
    def _iter_entries_from_open_file(f: Any, path: Path) -> Iterator[ManifestEntry]:
        """Yield entries from an open file positioned past the header.

        Buffers one line ahead so a torn final line (no trailing ``\\n``)
        can be detected and silently dropped — matches :meth:`load`'s
        ``raw.split("\\n")[:-1]`` torn-tail tolerance. Closes the file on
        exhaustion or exception.
        """
        try:
            prev_line: str | None = None
            prev_line_no = 0
            line_no = 1  # header was line 1
            for line in f:
                line_no += 1
                if prev_line is not None:
                    yield Manifest._parse_entry_line(prev_line, path, prev_line_no)
                prev_line = line
                prev_line_no = line_no
            if prev_line is not None and prev_line.endswith("\n"):
                yield Manifest._parse_entry_line(prev_line, path, prev_line_no)
        finally:
            f.close()

    @staticmethod
    def _parse_entry_line(line: str, path: Path, line_no: int) -> ManifestEntry:
        try:
            entry_data = json.loads(line)
            return ManifestEntry.from_dict(entry_data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ManifestCorruptError(
                f"manifest entry on line {line_no} is malformed: {exc}",
                path,
                line_number=line_no,
            ) from exc

    @property
    def extension_run_count(self) -> int:
        """Cumulative number of extension runs appended beyond the original sweep.

        Derived as ``max(0, max(run_id for entries) + 1 - parameter_spec["n"])``
        for Monte Carlo manifests; ``0`` for any other ``parameter_spec``
        kind or for an MC manifest that has not been extended. The on-disk
        header is not rewritten on extension (manifest headers are
        append-only), so this is the canonical way to ask "how many
        :func:`gmat_sweep.monte_carlo_extend` calls have landed on top of
        this sweep's original ``n``."
        """
        kind = self.parameter_spec.get("_kind")
        if kind != "monte_carlo":
            return 0
        original_n_raw = self.parameter_spec.get("n")
        if not isinstance(original_n_raw, int):
            return 0
        if not self.entries:
            return 0
        max_run_id = max(e.run_id for e in self.entries)
        return max(0, max_run_id + 1 - original_n_raw)

    @property
    def total_run_count(self) -> int:
        """The live run count — the on-disk header's ``run_count`` plus any extensions.

        The header's ``run_count`` field is frozen at first
        :meth:`save` and is not rewritten on :meth:`Sweep.extend`
        (append-only manifest header — see ``docs/manifest-schema.md``).
        Reading ``run_count`` alone therefore lags the actual run set
        after every extend; this property is what callers want when
        they ask "how many runs are now in this sweep, including
        extensions?".

        Implementation: ``max(header.run_count, max(e.run_id) + 1)`` so
        a mid-sweep ``Ctrl-C`` (entries < run_count) still reports the
        expected total, and an extended manifest (max_run_id ≥
        run_count) reports the post-extend total.
        """
        if not self.entries:
            return self.run_count
        return max(self.run_count, max(e.run_id for e in self.entries) + 1)


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
