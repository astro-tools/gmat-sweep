"""Forward-compatibility contract for ``Sweep.from_manifest().resume()`` (v0.2).

The v0.2 resume flow doesn't exist yet — but the v0.1 manifest format must
already carry every field the resume call will consume, so the on-disk shape
written by v0.1 sweeps remains replayable once :class:`Sweep.from_manifest` lands.

This module is a structural assertion: it round-trips a manifest through
:meth:`Manifest.save` / :meth:`Manifest.load` and asserts every field below is
present, typed correctly, and survives the round-trip bit-equal. No
``Sweep.from_manifest`` import — that's the deliberately deferred half.

The expected fields are derived from the v0.2 resume contract:

- **Header** — script SHA (detect upstream change), sweep/runner/install
  versions (detect tooling change), python/os fingerprint (informational),
  sweep seed (recreate Monte Carlo runs), parameter spec (know what runs were
  planned), run count (verify completeness).
- **Per-entry** — run_id (key the resume runs by), overrides (recreate the
  worker call), status (pick which runs to retry), output_paths (skip already-
  succeeded runs), started_at / ended_at / duration_s (telemetry), stderr (carry
  failure context across the resume boundary), log_path (point to the worker
  log for triage).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gmat_sweep.manifest import Manifest, ManifestEntry

_REQUIRED_HEADER_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)

_REQUIRED_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "overrides",
        "status",
        "output_paths",
        "started_at",
        "ended_at",
        "duration_s",
        "stderr",
        "log_path",
    }
)


def _utc(year: int, month: int, day: int, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(year, month, day, h, m, s, tzinfo=timezone.utc)


def _build_manifest_with_every_status() -> Manifest:
    """Build a manifest with one ok / one failed / one skipped entry."""
    return Manifest(
        script_sha256="0" * 64,
        gmat_sweep_version="0.1.0",
        gmat_run_version="0.4.0",
        gmat_install_version="R2026a",
        python_version="3.12.3",
        os_platform="Linux-6.6.0",
        sweep_seed=1729,
        parameter_spec={"Sat.SMA": [7000.0, 7100.0]},
        run_count=3,
        entries=[
            ManifestEntry(
                run_id=0,
                overrides={"Sat.SMA": 7000.0},
                status="ok",
                output_paths={"R": Path("/sweep/run-0/R.parquet")},
                started_at=_utc(2026, 5, 4, 0, 0, 0),
                ended_at=_utc(2026, 5, 4, 0, 0, 5),
                duration_s=5.0,
                stderr=None,
                log_path=Path("/sweep/run-0/worker.log"),
            ),
            ManifestEntry(
                run_id=1,
                overrides={"Sat.SMA": 7100.0},
                status="failed",
                output_paths={},
                started_at=_utc(2026, 5, 4, 0, 0, 6),
                ended_at=_utc(2026, 5, 4, 0, 0, 7),
                duration_s=1.0,
                stderr="Traceback (most recent call last):\n  ...",
                log_path=Path("/sweep/run-1/worker.log"),
            ),
            ManifestEntry(
                run_id=2,
                overrides={"Sat.SMA": 7200.0},
                status="skipped",
                output_paths={},
                started_at=_utc(2026, 5, 4, 0, 0, 8),
                ended_at=_utc(2026, 5, 4, 0, 0, 8),
                duration_s=0.0,
                stderr=None,
                log_path=None,
            ),
        ],
    )


# ---- header surface ------------------------------------------------------


def test_header_carries_every_field_resume_will_need() -> None:
    """Resume needs every field listed in `_REQUIRED_HEADER_FIELDS` and no other."""
    m = _build_manifest_with_every_status()
    header = m._header_dict()
    assert set(header.keys()) == _REQUIRED_HEADER_FIELDS


def test_header_round_trips_through_jsonl_bit_equal(tmp_path: Path) -> None:
    m = _build_manifest_with_every_status()
    path = tmp_path / "manifest.jsonl"
    m.save(path)

    raw_header_line = path.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(raw_header_line)
    assert set(header.keys()) == _REQUIRED_HEADER_FIELDS

    reloaded = Manifest.load(path)
    for field in _REQUIRED_HEADER_FIELDS:
        assert getattr(reloaded, field) == getattr(m, field), field


def test_header_field_types_are_resume_safe() -> None:
    m = _build_manifest_with_every_status()
    assert isinstance(m.schema_version, int)
    assert isinstance(m.script_sha256, str)
    assert isinstance(m.gmat_sweep_version, str)
    assert isinstance(m.gmat_run_version, str)
    assert isinstance(m.gmat_install_version, str)
    assert isinstance(m.python_version, str)
    assert isinstance(m.os_platform, str)
    assert m.sweep_seed is None or isinstance(m.sweep_seed, int)
    assert isinstance(m.parameter_spec, dict)
    assert isinstance(m.run_count, int)


# ---- entry surface -------------------------------------------------------


def test_entry_carries_every_field_resume_will_need() -> None:
    m = _build_manifest_with_every_status()
    for entry in m.entries:
        assert set(entry.to_dict().keys()) == _REQUIRED_ENTRY_FIELDS


def test_entries_round_trip_bit_equal(tmp_path: Path) -> None:
    m = _build_manifest_with_every_status()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    reloaded = Manifest.load(path)
    assert reloaded.entries == m.entries


def test_each_run_status_round_trips_with_its_diagnostic_payload() -> None:
    """Resume must distinguish ok / failed / skipped and pick up their stderr / outputs."""
    m = _build_manifest_with_every_status()
    by_status = {e.status: e for e in m.entries}
    assert set(by_status) == {"ok", "failed", "skipped"}

    assert by_status["ok"].stderr is None
    assert by_status["ok"].output_paths != {}

    assert by_status["failed"].stderr is not None and by_status["failed"].stderr != ""
    assert by_status["failed"].output_paths == {}

    assert by_status["skipped"].stderr is None
    assert by_status["skipped"].output_paths == {}


def test_overrides_preserve_resume_keying(tmp_path: Path) -> None:
    """Resume needs to recreate the worker call from `overrides` alone."""
    m = _build_manifest_with_every_status()
    path = tmp_path / "manifest.jsonl"
    m.save(path)
    reloaded = Manifest.load(path)
    for original, restored in zip(m.entries, reloaded.entries, strict=True):
        assert restored.overrides == original.overrides
        assert isinstance(restored.run_id, int)


def test_failed_run_carries_resume_triage_payload() -> None:
    """Resume must surface the stderr a v0.1 worker captured for the failed run."""
    m = _build_manifest_with_every_status()
    failed = next(e for e in m.entries if e.status == "failed")
    assert failed.stderr is not None
    assert "Traceback" in failed.stderr
    assert failed.log_path is not None


# ---- forward compat -----------------------------------------------------


def test_unknown_extra_header_fields_are_dropped_on_load(tmp_path: Path) -> None:
    """A future resume tool MAY emit extra header fields; v0.1 load must ignore them.

    The `_header_from_dict` constructor reads only the documented fields, so a
    forward-emitted manifest with additional fields still loads through the v0.1
    code path. This protects forward compatibility: old code reading new manifests
    keeps working until the new fields are formally required.
    """
    m = _build_manifest_with_every_status()
    header = m._header_dict()
    header["future_field_we_dont_know_about"] = {"hint": "v0.3 thing"}

    path = tmp_path / "manifest.jsonl"
    body = json.dumps(header, sort_keys=True) + "\n"
    path.write_text(body, encoding="utf-8")

    reloaded = Manifest.load(path)
    assert reloaded.script_sha256 == m.script_sha256
    assert reloaded.run_count == m.run_count
