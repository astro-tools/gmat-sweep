"""Tests for gmat_sweep.manifest — JSON Lines manifest, append/fsync, lookup helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gmat_sweep.errors import ManifestCorruptError
from gmat_sweep.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    ManifestEntry,
    canonical_script_sha256,
)
from gmat_sweep.spec import RunOutcome


def _utc(year: int, month: int, day: int, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(year, month, day, h, m, s, tzinfo=timezone.utc)


def _make_entry(run_id: int = 0, status: str = "ok", *, log: bool = True) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides={"Sat.SMA": 7000.0 + run_id, "Sat.ECC": 0.001},
        status=status,  # type: ignore[arg-type]
        output_paths={"ReportFile1": Path(f"/o/run-{run_id}/r1.parquet")} if status == "ok" else {},
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 12),
        duration_s=12.0,
        stderr=None if status == "ok" else "boom",
        log_path=Path(f"/o/run-{run_id}/log.txt") if log else None,
    )


def _make_manifest(n_entries: int = 0) -> Manifest:
    return Manifest(
        script_sha256="a" * 64,
        gmat_sweep_version="0.1.0",
        gmat_run_version="0.4.0",
        gmat_install_version="R2026a",
        python_version="3.12.3",
        os_platform="Linux-6.6.0",
        sweep_seed=1729,
        parameter_spec={"Sat.SMA": [7000.0, 7100.0], "Sat.ECC": [0.001, 0.002]},
        run_count=4,
        entries=[_make_entry(i) for i in range(n_entries)],
    )


# ---- ManifestEntry round-trip --------------------------------------------


def test_manifest_entry_round_trips_through_json() -> None:
    original = _make_entry(run_id=3)
    serialised = json.dumps(original.to_dict(), sort_keys=True)
    restored = ManifestEntry.from_dict(json.loads(serialised))
    assert restored == original
    assert json.dumps(restored.to_dict(), sort_keys=True) == serialised


def test_manifest_entry_failed_round_trips_with_stderr_and_no_outputs() -> None:
    original = _make_entry(run_id=2, status="failed")
    restored = ManifestEntry.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.status == "failed"
    assert restored.stderr == "boom"
    assert restored.output_paths == {}


def test_manifest_entry_log_path_none_round_trips() -> None:
    original = _make_entry(run_id=1, log=False)
    restored = ManifestEntry.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.log_path is None


def test_manifest_entry_from_outcome_carries_overrides_and_log_path() -> None:
    outcome = RunOutcome.ok(
        run_id=7,
        output_paths={"ReportFile1": Path("/o/run-7/r1.parquet")},
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 5),
    )
    entry = ManifestEntry.from_outcome(
        outcome,
        overrides={"Sat.SMA": 7050.0},
        log_path=Path("/o/run-7/log.txt"),
    )
    assert entry.run_id == 7
    assert entry.status == "ok"
    assert entry.duration_s == 5.0
    assert entry.overrides == {"Sat.SMA": 7050.0}
    assert entry.log_path == Path("/o/run-7/log.txt")
    assert entry.stderr is None


def test_manifest_entry_from_outcome_log_path_defaults_to_none() -> None:
    outcome = RunOutcome.failed(
        run_id=4,
        stderr="GMAT exploded",
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
    )
    entry = ManifestEntry.from_outcome(outcome, overrides={"Sat.SMA": 7000.0})
    assert entry.log_path is None
    assert entry.status == "failed"
    assert entry.stderr == "GMAT exploded"
    assert entry.output_paths == {}


# ---- Manifest save / load round-trip -------------------------------------


def test_save_then_load_round_trips_bit_equal(tmp_path: Path) -> None:
    original = _make_manifest(n_entries=3)
    path = tmp_path / "manifest.jsonl"
    original.save(path)

    loaded = Manifest.load(path)
    assert loaded == original
    assert [e for e in loaded.entries] == original.entries

    # Re-saving produces the same bytes (deterministic key ordering).
    second = tmp_path / "second.jsonl"
    loaded.save(second)
    assert path.read_bytes() == second.read_bytes()


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    m.save(path)
    assert path.exists()


def test_save_writes_jsonl_with_header_first(tmp_path: Path) -> None:
    m = _make_manifest(n_entries=2)
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    header = json.loads(lines[0])
    assert header["script_sha256"] == m.script_sha256
    assert "entries" not in header  # entries live on subsequent lines
    for line in lines[1:]:
        assert "run_id" in json.loads(line)


def test_save_records_path_for_subsequent_append(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    # append_entry should now succeed without re-binding.
    m.append_entry(_make_entry(run_id=99))
    reloaded = Manifest.load(path)
    assert reloaded.entries[-1].run_id == 99


# ---- append_entry --------------------------------------------------------


def test_append_entry_extends_file_and_in_memory_list(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    e0 = _make_entry(run_id=0)
    e1 = _make_entry(run_id=1)
    m.append_entry(e0)
    m.append_entry(e1)

    assert m.entries == [e0, e1]
    reloaded = Manifest.load(path)
    assert reloaded.entries == [e0, e1]


def test_append_entry_without_save_or_load_raises() -> None:
    m = _make_manifest()
    with pytest.raises(RuntimeError, match="save\\(\\) or load\\(\\)"):
        m.append_entry(_make_entry())


def test_append_entry_after_load_writes_to_loaded_path(tmp_path: Path) -> None:
    seed = _make_manifest(n_entries=1)
    path = tmp_path / "manifest.jsonl"
    seed.save(path)

    loaded = Manifest.load(path)
    loaded.append_entry(_make_entry(run_id=5))

    final = Manifest.load(path)
    assert [e.run_id for e in final.entries] == [0, 5]


def test_append_entry_calls_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as os_module

    fsynced: list[int] = []
    real_fsync = os_module.fsync

    def fake_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("gmat_sweep.manifest.os.fsync", fake_fsync)

    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    fsynced.clear()
    m.append_entry(_make_entry())
    assert len(fsynced) >= 1


def test_save_tolerates_dir_fsync_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Directory fsync is best-effort (Windows can't do it; some filesystems
    # refuse). save() must not propagate the failure.
    import os as os_module

    real_open = os_module.open

    def fake_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        # Fail only when opening a directory for fsync.
        if Path(os_module.fsdecode(path)).is_dir():
            raise PermissionError("simulated dir fsync open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("gmat_sweep.manifest.os.open", fake_open)

    m = _make_manifest()
    m.save(tmp_path / "manifest.jsonl")  # must not raise


def test_save_tolerates_dir_fsync_call_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os as os_module

    real_fsync = os_module.fsync
    real_fstat = os_module.fstat

    def fake_fsync(fd: int) -> None:
        # Fail fsync only on directory file descriptors; leave file fsyncs alone.
        try:
            mode = real_fstat(fd).st_mode
        except OSError:
            mode = 0
        if (mode & 0o170000) == 0o040000:  # S_IFDIR
            raise OSError("simulated dir fsync failure")
        real_fsync(fd)

    monkeypatch.setattr("gmat_sweep.manifest.os.fsync", fake_fsync)

    m = _make_manifest()
    m.save(tmp_path / "manifest.jsonl")  # must not raise


def test_save_calls_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as os_module

    fsynced: list[int] = []
    real_fsync = os_module.fsync

    def fake_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("gmat_sweep.manifest.os.fsync", fake_fsync)

    m = _make_manifest()
    m.save(tmp_path / "manifest.jsonl")
    # At least the file fsync; parent dir fsync is best-effort.
    assert len(fsynced) >= 1


# ---- Truncation tolerance ------------------------------------------------


def test_load_tolerates_torn_final_line(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    m.append_entry(_make_entry(run_id=0))
    m.append_entry(_make_entry(run_id=1))

    raw = path.read_bytes()
    # Drop the trailing newline plus a few bytes from the last entry to simulate
    # a Ctrl-C in the middle of a write.
    truncated = raw.rstrip(b"\n")
    truncated = truncated[: -len(b'"run_id": 1}') // 2]
    path.write_bytes(truncated)

    loaded = Manifest.load(path)
    # The first entry survived; the partial second one was dropped.
    assert [e.run_id for e in loaded.entries] == [0]


def test_load_tolerates_truncation_at_line_boundary_after_header(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    m.append_entry(_make_entry(run_id=0))
    m.append_entry(_make_entry(run_id=1))

    raw = path.read_text(encoding="utf-8")
    # Truncate to keep only the header line (with its trailing \n).
    header_end = raw.index("\n") + 1
    path.write_text(raw[:header_end], encoding="utf-8")

    loaded = Manifest.load(path)
    assert loaded.entries == []
    assert loaded.script_sha256 == m.script_sha256


def test_load_tolerates_truncation_mid_entry_no_trailing_newline(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    m.append_entry(_make_entry(run_id=0))

    raw = path.read_text(encoding="utf-8")
    # Cut the entry line in half, no trailing newline → torn write.
    header_end = raw.index("\n") + 1
    entry_line = raw[header_end:].rstrip("\n")
    path.write_text(raw[:header_end] + entry_line[: len(entry_line) // 2], encoding="utf-8")

    loaded = Manifest.load(path)
    assert loaded.entries == []


# ---- Corruption ----------------------------------------------------------


def test_load_empty_file_raises_manifest_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path


def test_load_torn_header_with_no_newline_raises_manifest_corrupt(tmp_path: Path) -> None:
    # File content has no '\n' at all — the header line itself was being written
    # when the process died. Treated as corruption (no complete header).
    path = tmp_path / "torn-header.jsonl"
    path.write_text('{"script_sha256": "abc"', encoding="utf-8")
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path
    assert "header" in str(excinfo.value)


def test_load_invalid_header_json_raises_manifest_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path
    assert "header" in str(excinfo.value)


def test_load_header_missing_fields_raises_manifest_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.jsonl"
    path.write_text(json.dumps({"script_sha256": "abc"}) + "\n", encoding="utf-8")
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path


def test_load_complete_but_malformed_entry_line_raises(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    with open(path, "a", encoding="utf-8") as f:
        f.write("{this is not valid}\n")  # complete line, malformed JSON
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert "line 2" in str(excinfo.value)


def test_load_complete_entry_with_missing_field_raises(tmp_path: Path) -> None:
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": 0}) + "\n")
    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path


# ---- find_failed / find_missing -----------------------------------------


def test_load_dedupes_duplicate_run_ids_last_wins(tmp_path: Path) -> None:
    """Resume appends a fresh entry with the same run_id as the failed one;
    Manifest.load merges them last-wins so the resumed status survives."""
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    failed = _make_entry(run_id=1, status="failed")
    retried = _make_entry(run_id=1, status="ok")
    m.append_entry(failed)
    m.append_entry(retried)

    reloaded = Manifest.load(path)
    by_run_id = {e.run_id: e for e in reloaded.entries}
    assert by_run_id[1].status == "ok"
    assert by_run_id[1].output_paths != {}
    # Only the resumed entry survives — there are no two run_id=1 entries.
    assert sum(1 for e in reloaded.entries if e.run_id == 1) == 1


def test_load_dedup_preserves_first_occurrence_position(tmp_path: Path) -> None:
    """For unique run_ids load() must preserve file order. For a duplicated
    run_id the entry stays at the position of its FIRST appearance — the
    resumed entry rides the original's slot, not the tail."""
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    m.append_entry(_make_entry(run_id=0, status="ok"))
    m.append_entry(_make_entry(run_id=1, status="failed"))
    m.append_entry(_make_entry(run_id=2, status="ok"))
    m.append_entry(_make_entry(run_id=1, status="ok"))  # resumed retry

    reloaded = Manifest.load(path)
    assert [e.run_id for e in reloaded.entries] == [0, 1, 2]
    by_run_id = {e.run_id: e for e in reloaded.entries}
    assert by_run_id[1].status == "ok"


def test_load_dedup_no_op_when_run_ids_unique(tmp_path: Path) -> None:
    """Manifests without resume duplicates must round-trip unchanged."""
    m = _make_manifest(n_entries=3)
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    reloaded = Manifest.load(path)
    assert reloaded.entries == m.entries


def test_find_failed_after_resume_excludes_recovered_run_ids(tmp_path: Path) -> None:
    """After dedup, find_failed only surfaces run_ids whose LAST entry is failed."""
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    m.append_entry(_make_entry(run_id=0, status="failed"))
    m.append_entry(_make_entry(run_id=1, status="failed"))
    m.append_entry(_make_entry(run_id=0, status="ok"))  # recovered on resume

    reloaded = Manifest.load(path)
    assert reloaded.find_failed() == [1]


def test_find_failed_returns_failed_run_ids_in_order() -> None:
    m = _make_manifest()
    m.entries = [
        _make_entry(0, status="ok"),
        _make_entry(1, status="failed"),
        _make_entry(2, status="ok"),
        _make_entry(3, status="failed"),
        _make_entry(4, status="skipped"),
    ]
    assert m.find_failed() == [1, 3]


def test_find_failed_empty_when_all_ok() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(0), _make_entry(1)]
    assert m.find_failed() == []


def test_find_failed_returns_all_when_all_failed() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(i, status="failed") for i in range(3)]
    assert m.find_failed() == [0, 1, 2]


def test_find_failed_skipped_runs_are_not_failed() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(0, status="skipped")]
    assert m.find_failed() == []


def test_find_missing_preserves_input_order() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(0), _make_entry(2), _make_entry(4)]
    assert m.find_missing([4, 3, 2, 1, 0]) == [3, 1]


def test_find_missing_empty_input_returns_empty() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(0)]
    assert m.find_missing([]) == []


def test_find_missing_all_present_returns_empty() -> None:
    m = _make_manifest()
    m.entries = [_make_entry(0), _make_entry(1)]
    assert m.find_missing([0, 1]) == []


def test_find_missing_none_present_returns_all() -> None:
    m = _make_manifest()
    assert m.find_missing([0, 1, 2]) == [0, 1, 2]


# ---- canonical_script_sha256 --------------------------------------------


def _write_script(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_canonical_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    body = "Create Spacecraft Sat;\nSat.SMA = 7000;\nPropagate prop(Sat);\n"
    lf = _write_script(tmp_path, "lf.script", body.encode("utf-8"))
    crlf = _write_script(tmp_path, "crlf.script", body.replace("\n", "\r\n").encode("utf-8"))
    cr = _write_script(tmp_path, "cr.script", body.replace("\n", "\r").encode("utf-8"))

    h_lf = canonical_script_sha256(lf)
    assert canonical_script_sha256(crlf) == h_lf
    assert canonical_script_sha256(cr) == h_lf


def test_canonical_hash_is_stable_across_trailing_newline_variants(tmp_path: Path) -> None:
    base = "Create Spacecraft Sat;\nSat.SMA = 7000;"
    no_nl = _write_script(tmp_path, "none.script", base.encode("utf-8"))
    one_nl = _write_script(tmp_path, "one.script", (base + "\n").encode("utf-8"))
    two_nl = _write_script(tmp_path, "two.script", (base + "\n\n").encode("utf-8"))

    h = canonical_script_sha256(no_nl)
    assert canonical_script_sha256(one_nl) == h
    assert canonical_script_sha256(two_nl) == h


def test_canonical_hash_changes_with_payload(tmp_path: Path) -> None:
    a = _write_script(tmp_path, "a.script", b"Sat.SMA = 7000;\n")
    b = _write_script(tmp_path, "b.script", b"Sat.SMA = 7100;\n")
    assert canonical_script_sha256(a) != canonical_script_sha256(b)


def test_canonical_hash_is_64_hex_chars(tmp_path: Path) -> None:
    p = _write_script(tmp_path, "x.script", b"x\n")
    h = canonical_script_sha256(p)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---- Header field surface check ------------------------------------------


def test_header_dict_carries_every_documented_field() -> None:
    m = _make_manifest()
    header: dict[str, Any] = m._header_dict()
    expected = {
        "schema_version",
        "script_sha256",
        "gmat_sweep_version",
        "gmat_run_version",
        "gmat_install_version",
        "python_version",
        "os_platform",
        "sweep_seed",
        "parameter_spec",
        "run_count",
        "backend",
    }
    assert set(header.keys()) == expected


# ---- backend field -------------------------------------------------------


def test_manifest_backend_round_trips(tmp_path: Path) -> None:
    """The pool-class name on the header survives a save+load round-trip."""
    m = _make_manifest(n_entries=1)
    m.backend = "DaskPool"
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    reloaded = Manifest.load(path)
    assert reloaded.backend == "DaskPool"


def test_load_manifest_without_backend_loads_as_unknown(tmp_path: Path) -> None:
    """Manifests written before the ``backend`` field landed (v0.1/v0.2)
    omit it; loading them must yield ``backend == "unknown"`` rather than
    raise."""
    m = _make_manifest(n_entries=1)
    header = m._header_dict()
    del header["backend"]

    path = tmp_path / "manifest.jsonl"
    body = json.dumps(header, sort_keys=True) + "\n"
    body += json.dumps(m.entries[0].to_dict(), sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8")

    reloaded = Manifest.load(path)
    assert reloaded.backend == "unknown"
    assert reloaded.script_sha256 == m.script_sha256


# ---- schema_version freeze ----------------------------------------------


def test_save_writes_schema_version_in_header(tmp_path: Path) -> None:
    """Headers written by this gmat-sweep carry the supported schema version."""
    m = _make_manifest()
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["schema_version"] == MANIFEST_SCHEMA_VERSION


def test_load_manifest_without_schema_version_loads_as_v1(tmp_path: Path) -> None:
    """A manifest header that omits schema_version loads as schema_version=1
    for backwards compatibility with manifests written before the field was
    introduced."""
    m = _make_manifest(n_entries=2)
    header = m._header_dict()
    del header["schema_version"]  # simulate a manifest from before the field landed

    path = tmp_path / "manifest.jsonl"
    lines = [json.dumps(header, sort_keys=True)]
    lines.extend(json.dumps(e.to_dict(), sort_keys=True) for e in m.entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reloaded = Manifest.load(path)
    assert reloaded.schema_version == 1
    assert reloaded.script_sha256 == m.script_sha256
    assert [e.run_id for e in reloaded.entries] == [0, 1]


def test_load_rejects_schema_version_newer_than_supported(tmp_path: Path) -> None:
    """A header carrying a schema_version greater than the running gmat-sweep
    supports raises ManifestCorruptError — the reader is older than the writer
    and may have lost or changed semantics on existing fields."""
    m = _make_manifest()
    header = m._header_dict()
    header["schema_version"] = MANIFEST_SCHEMA_VERSION + 1

    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path
    assert "schema_version" in str(excinfo.value)
    assert "newer" in str(excinfo.value)


def test_load_rejects_non_integer_schema_version(tmp_path: Path) -> None:
    m = _make_manifest()
    header = m._header_dict()
    header["schema_version"] = "not-an-int"

    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ManifestCorruptError) as excinfo:
        Manifest.load(path)
    assert excinfo.value.path == path
    assert "schema_version" in str(excinfo.value)


def test_save_and_load_round_trip_preserves_schema_version(tmp_path: Path) -> None:
    m = _make_manifest(n_entries=2)
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    reloaded = Manifest.load(path)
    assert reloaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert reloaded == m


def test_load_drops_unknown_extra_header_fields(tmp_path: Path) -> None:
    """Forward-compat: a header carrying unknown extra fields still loads.

    The schema-freeze policy promises that an older reader can keep loading a
    manifest written by a newer writer until the schema_version is bumped — so
    extra header fields the running gmat-sweep doesn't recognise must be
    silently ignored, not rejected. Without this, every additive field would
    become an immediate compatibility break.
    """
    m = _make_manifest(n_entries=1)
    header = m._header_dict()
    header["future_field_we_dont_know_about"] = {"hint": "v0.3 thing"}

    path = tmp_path / "manifest.jsonl"
    body = json.dumps(header, sort_keys=True) + "\n"
    body += json.dumps(m.entries[0].to_dict(), sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8")

    reloaded = Manifest.load(path)
    assert reloaded.script_sha256 == m.script_sha256
    assert reloaded.run_count == m.run_count
    assert [e.run_id for e in reloaded.entries] == [0]
