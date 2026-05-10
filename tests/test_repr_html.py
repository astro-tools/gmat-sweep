"""Tests for the ``_repr_html_`` methods on Sweep, RunOutcome, ManifestEntry."""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.manifest import ManifestEntry
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeResults


def _utc(year: int, month: int, day: int, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(year, month, day, h, m, s, tzinfo=timezone.utc)


def _assert_well_formed_html(markup: str) -> None:
    """Feed ``markup`` to :class:`html.parser.HTMLParser` and assert it doesn't raise.

    HTMLParser is permissive (it accepts unclosed tags), but it raises on
    structurally broken markup like an unterminated attribute. This is the
    practical "valid HTML" check the issue calls for — a stricter XML
    parse would reject perfectly fine HTML.
    """

    class _Parser(HTMLParser):
        def error(self, message: str) -> None:  # pragma: no cover - parser API
            raise AssertionError(f"HTMLParser error: {message}")

    parser = _Parser()
    parser.feed(markup)
    parser.close()


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT script\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _make_runs(script: Path, output_dir: Path, n: int) -> list[RunSpec]:
    return [
        RunSpec(
            script_path=script,
            overrides={"Sat.SMA": 7000.0 + i},
            output_dir=output_dir / f"run-{i}",
            run_id=i,
            seed=None,
            run_options={},
        )
        for i in range(n)
    ]


def _payload_run_hook() -> Any:
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- Sweep ----------------------------------------------------------------


def test_sweep_repr_html_pre_run(tmp_path: Path) -> None:
    script = _write_script(tmp_path)
    runs = _make_runs(script, tmp_path / "out", n=3)
    with LocalJoblibPool(max_workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=tmp_path / "out" / "manifest.jsonl",
            output_dir=tmp_path / "out",
            script_path=script,
            parameter_spec={"_kind": "grid", "Sat.SMA": [7000.0, 7100.0, 7200.0]},
            sweep_seed=42,
            progress=False,
        )

        html = sweep._repr_html_()

    _assert_well_formed_html(html)
    assert html.startswith("<table")
    assert html.endswith("</table>")
    assert "run_count" in html and ">3<" in html
    assert "grid" in html
    assert "LocalJoblibPool" in html
    assert "not yet executed" in html
    assert "sweep_seed" in html and "42" in html


def test_sweep_repr_html_post_run(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=2)
    with LocalJoblibPool(max_workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"_kind": "grid", "Sat.SMA": [7000.0, 7100.0]},
            progress=False,
        ).run()

        html = sweep._repr_html_()

    _assert_well_formed_html(html)
    assert "outcomes" in html
    assert "2 ok" in html
    assert "0 failed" in html
    assert "not yet executed" not in html


def test_sweep_repr_html_handles_unreadable_script(tmp_path: Path) -> None:
    # Pass a non-existent script path so canonical_script_sha256 raises.
    # The repr should degrade gracefully rather than blow up the notebook cell.
    script = tmp_path / "missing.script"
    runs: list[RunSpec] = []
    with LocalJoblibPool(max_workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=tmp_path / "out" / "manifest.jsonl",
            output_dir=tmp_path / "out",
            script_path=script,
            parameter_spec={"_kind": "grid"},
            progress=False,
        )
        html = sweep._repr_html_()

    _assert_well_formed_html(html)
    assert "script not readable" in html


# ---- RunOutcome -----------------------------------------------------------


def test_run_outcome_repr_html_ok() -> None:
    outcome = RunOutcome.ok(
        run_id=7,
        output_paths={"ReportFile1": Path("/o/run-7/r1.parquet")},
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 12),
    )
    html = outcome._repr_html_()
    _assert_well_formed_html(html)
    assert "run_id" in html and ">7<" in html
    assert "ok" in html
    assert "12.00 s" in html
    assert "ReportFile1" in html
    assert "r1.parquet" in html
    assert "(no stderr)" in html


def test_run_outcome_repr_html_failed_truncates_long_stderr() -> None:
    long_first_line = "E" * 200
    outcome = RunOutcome.failed(
        run_id=3,
        stderr=f"{long_first_line}\nsecond line not shown",
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
    )
    html = outcome._repr_html_()
    _assert_well_formed_html(html)
    assert "failed" in html
    assert "..." in html
    assert "second line not shown" not in html
    assert "(none)" in html  # output_paths is empty for failed runs


def test_run_outcome_repr_html_escapes_html_in_stderr() -> None:
    outcome = RunOutcome.failed(
        run_id=1,
        stderr="<script>alert('boom')</script>",
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
    )
    html = outcome._repr_html_()
    _assert_well_formed_html(html)
    # The literal '<script>' must not appear unescaped — anything that does
    # would let a stderr line break out of the table cell.
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ---- ManifestEntry --------------------------------------------------------


def _make_entry(
    *,
    run_id: int = 0,
    status: str = "ok",
    overrides: dict[str, Any] | None = None,
    log: bool = True,
    stderr: str | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides=overrides if overrides is not None else {"Sat.SMA": 7000.0 + run_id},
        status=status,  # type: ignore[arg-type]
        output_paths={"R1": Path(f"/o/run-{run_id}/r1.parquet")} if status == "ok" else {},
        started_at=_utc(2026, 5, 4, 0, 0, 0),
        ended_at=_utc(2026, 5, 4, 0, 0, 12),
        duration_s=12.0,
        stderr=stderr,
        log_path=Path(f"/o/run-{run_id}/log.txt") if log else None,
    )


def test_manifest_entry_repr_html_ok() -> None:
    entry = _make_entry(run_id=2)
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    assert "run_id" in html and ">2<" in html
    assert "ok" in html
    assert "12.00 s" in html
    assert "Sat.SMA" in html
    assert "log.txt" in html
    assert "r1.parquet" in html
    assert "(no stderr)" in html


def test_manifest_entry_repr_html_failed_with_no_log() -> None:
    entry = _make_entry(run_id=5, status="failed", log=False, stderr="boom")
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    assert "failed" in html
    assert "boom" in html
    # log_path is None → cell renders as "(none)"; output_paths is empty for
    # failed runs, also "(none)". We just need the status row to include the
    # word at least twice (log_path + output_paths cells).
    assert html.count("(none)") >= 2


def test_manifest_entry_repr_html_renders_many_overrides() -> None:
    overrides = {f"Sat.Field{i}": float(i) for i in range(50)}
    entry = _make_entry(run_id=1, overrides=overrides)
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    # No truncation cap — every key is present.
    for i in range(50):
        assert f"Sat.Field{i}" in html


def test_manifest_entry_repr_html_empty_overrides() -> None:
    entry = _make_entry(run_id=1, overrides={})
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    assert "(none)" in html


def test_manifest_entry_repr_html_escapes_keys_and_values() -> None:
    entry = _make_entry(
        run_id=1,
        overrides={"<dangerous>": "<x>"},
    )
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    assert "<dangerous>" not in html
    assert "&lt;dangerous&gt;" in html


@pytest.mark.parametrize("status", ["ok", "failed", "skipped"])
def test_manifest_entry_repr_html_includes_status(status: str) -> None:
    entry = _make_entry(status=status, stderr=None if status == "ok" else "x")
    html = entry._repr_html_()
    _assert_well_formed_html(html)
    assert status in html
