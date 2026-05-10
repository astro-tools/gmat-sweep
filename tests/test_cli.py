"""Tests for gmat_sweep.cli — argparse-driven console-script entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from gmat_sweep import cli
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest, ManifestEntry
from tests.conftest import FakeGmatRun, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _payload_run_hook() -> Any:
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- _parse_grid_spec: linspace form ------------------------------------


def test_parse_grid_spec_linspace_returns_evenly_spaced_floats() -> None:
    name, values = cli._parse_grid_spec("Sat.SMA=7000:8000:5")
    assert name == "Sat.SMA"
    assert values == [7000.0, 7250.0, 7500.0, 7750.0, 8000.0]


def test_parse_grid_spec_linspace_count_2_returns_endpoints() -> None:
    name, values = cli._parse_grid_spec("a=0:10:2")
    assert (name, values) == ("a", [0.0, 10.0])


def test_parse_grid_spec_linspace_rejects_non_numeric_bounds() -> None:
    with pytest.raises(SweepConfigError, match="numeric"):
        cli._parse_grid_spec("a=foo:10:5")


def test_parse_grid_spec_linspace_rejects_non_integer_count() -> None:
    with pytest.raises(SweepConfigError, match="integer"):
        cli._parse_grid_spec("a=0:10:5.5")


def test_parse_grid_spec_linspace_rejects_count_below_two() -> None:
    with pytest.raises(SweepConfigError, match=">= 2"):
        cli._parse_grid_spec("a=0:10:1")


def test_parse_grid_spec_linspace_rejects_wrong_arity() -> None:
    with pytest.raises(SweepConfigError, match="three colon-separated"):
        cli._parse_grid_spec("a=0:10")


# ---- _parse_grid_spec: explicit form ------------------------------------


def test_parse_grid_spec_explicit_int_values() -> None:
    name, values = cli._parse_grid_spec("Sat.DryMass=100,200,300")
    assert name == "Sat.DryMass"
    assert values == [100, 200, 300]
    assert all(isinstance(v, int) for v in values)


def test_parse_grid_spec_explicit_float_values() -> None:
    _, values = cli._parse_grid_spec("a=1.5,2.5,3.5")
    assert values == [1.5, 2.5, 3.5]


def test_parse_grid_spec_explicit_string_fallback() -> None:
    _, values = cli._parse_grid_spec("Mode=Cartesian,Keplerian")
    assert values == ["Cartesian", "Keplerian"]


def test_parse_grid_spec_explicit_single_value() -> None:
    _, values = cli._parse_grid_spec("a=42")
    assert values == [42]


def test_parse_grid_spec_explicit_rejects_empty_value() -> None:
    with pytest.raises(SweepConfigError, match="empty value"):
        cli._parse_grid_spec("a=1,,3")


# ---- _parse_grid_spec: malformed ----------------------------------------


def test_parse_grid_spec_rejects_missing_equals() -> None:
    with pytest.raises(SweepConfigError, match="must contain '='"):
        cli._parse_grid_spec("Sat.SMA")


def test_parse_grid_spec_rejects_empty_name() -> None:
    with pytest.raises(SweepConfigError, match="missing a name"):
        cli._parse_grid_spec("=1,2,3")


def test_parse_grid_spec_rejects_empty_rhs() -> None:
    with pytest.raises(SweepConfigError, match="no values"):
        cli._parse_grid_spec("a=")


# ---- run subcommand: end-to-end -----------------------------------------


def test_run_writes_manifest_and_prints_summary(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:3",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    assert (out / "manifest.jsonl").exists()
    captured = capsys.readouterr()
    assert "3 runs" in captured.out
    assert "3 ok" in captured.out
    assert str(out) in captured.out


def test_run_with_two_grid_flags_produces_cartesian_product(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--grid",
            "Sat.ECC=0.0,0.1,0.2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.run_count == 6
    assert manifest.parameter_spec["_kind"] == "grid"
    grid_axes = {k: v for k, v in manifest.parameter_spec.items() if k != "_kind"}
    assert sorted(grid_axes.keys()) == ["Sat.ECC", "Sat.SMA"]


def test_run_with_failing_runs_summary_shows_breakdown(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    def _setitem(_key: str, value: Any) -> None:
        if value == 8000.0:
            raise ValueError("rejected")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "2 runs" in captured.out
    assert "1 ok" in captured.out
    assert "1 failed" in captured.out
    assert "skipped" not in captured.out


def test_run_rejects_missing_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--out",
            str(tmp_path / "out"),
            str(tmp_path / "does-not-exist.script"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


def test_run_rejects_malformed_grid(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "run",
            "--grid",
            "no-equals-sign",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "must contain '='" in capsys.readouterr().err


def test_run_rejects_duplicate_grid_axis(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--grid",
            "a=3,4",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "more than once" in capsys.readouterr().err


# ---- show subcommand ----------------------------------------------------


def test_show_prints_summary(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    cli.main(
        [
            "run",
            "--grid",
            "a=1,2,3",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()  # discard run's summary

    rc = cli.main(["show", str(out / "manifest.jsonl")])
    assert rc == 0
    captured = capsys.readouterr()
    assert "3 runs" in captured.out
    assert "3 ok" in captured.out
    assert str(out) in captured.out


def test_show_on_missing_file_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-manifest.jsonl"
    rc = cli.main(["show", str(missing)])
    assert rc == cli.EXIT_MANIFEST
    assert "not found" in capsys.readouterr().err


def test_show_on_zero_entry_manifest_prints_zero_ok_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A freshly-saved manifest header with no entries appended yet is the
    edge case ``_format_summary`` falls back on. The output must remain
    parseable — callers that grep for ``"<N> runs"`` or pipe the line into
    a downstream tool expect a stable shape regardless of bucket counts.
    """
    manifest = Manifest(
        script_sha256="a" * 64,
        gmat_sweep_version="0.3.0",
        gmat_run_version="0.4.0",
        gmat_install_version="R2026a",
        python_version="3.12.3",
        os_platform="Linux-6.6.0",
        sweep_seed=None,
        parameter_spec={"_kind": "grid", "Sat.SMA": [7000.0]},
        run_count=1,
        entries=[],
    )
    path = tmp_path / "manifest.jsonl"
    manifest.save(path)

    rc = cli.main(["show", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 runs" in out
    assert "0 ok" in out
    assert str(tmp_path) in out


# ---- show --detail / --run / --filter -----------------------------------


def _show_entry(
    run_id: int,
    status: str,
    *,
    duration_s: float = 1.0,
    stderr: str | None = None,
    log_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides=overrides if overrides is not None else {"Sat.SMA": 7000.0 + run_id},
        status=status,  # type: ignore[arg-type]
        output_paths={},
        started_at=datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 4, 0, 0, 1, tzinfo=timezone.utc),
        duration_s=duration_s,
        stderr=stderr,
        log_path=log_path,
    )


def _write_show_manifest(tmp_path: Path, entries: list[ManifestEntry]) -> Path:
    manifest = Manifest(
        script_sha256="a" * 64,
        gmat_sweep_version="0.2.0",
        gmat_run_version="0.4.0",
        gmat_install_version="R2026a",
        python_version="3.12.3",
        os_platform="Linux-6.6.0",
        sweep_seed=None,
        parameter_spec={"_kind": "grid", "Sat.SMA": [7000.0, 7100.0]},
        run_count=len(entries),
        entries=entries,
    )
    path = tmp_path / "manifest.jsonl"
    manifest.save(path)
    return path


def test_truncate_stderr_summary_short_unchanged() -> None:
    assert cli._truncate_stderr_summary("ValueError: boom") == "ValueError: boom"


def test_truncate_stderr_summary_long_ellipsised_to_60_chars() -> None:
    long = "x" * 100
    out = cli._truncate_stderr_summary(long)
    assert len(out) == 60
    assert out.endswith("...")


def test_truncate_stderr_summary_takes_first_line_only() -> None:
    assert cli._truncate_stderr_summary("first line\nsecond line\n") == "first line"


def test_truncate_stderr_summary_none_and_empty_return_empty_string() -> None:
    assert cli._truncate_stderr_summary(None) == ""
    assert cli._truncate_stderr_summary("") == ""


def test_show_detail_orders_failed_before_ok_then_run_id_ascending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        _show_entry(0, "ok"),
        _show_entry(1, "failed", stderr="ValueError: out of range\n"),
        _show_entry(2, "skipped", stderr="prerequisite missing"),
        _show_entry(3, "ok"),
        _show_entry(4, "failed", stderr="other"),
    ]
    path = _write_show_manifest(tmp_path, entries)

    rc = cli.main(["show", "--detail", str(path)])
    assert rc == 0
    out_lines = capsys.readouterr().out.splitlines()
    # Header + 5 data rows + summary line.
    assert len(out_lines) == 7
    assert out_lines[0].startswith("run_id")
    statuses_in_order = [line.split()[1] for line in out_lines[1:6]]
    assert statuses_in_order == ["failed", "failed", "skipped", "ok", "ok"]
    run_ids_in_order = [int(line.split()[0]) for line in out_lines[1:6]]
    assert run_ids_in_order == [1, 4, 2, 0, 3]
    assert out_lines[-1].startswith("5 runs")


def test_show_detail_renders_dash_for_ok_stderr_and_missing_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_show_manifest(tmp_path, [_show_entry(0, "ok")])
    cli.main(["show", "--detail", str(path)])
    out_lines = capsys.readouterr().out.splitlines()
    # The data row should carry two em-dash cells: stderr_summary and log_path.
    # (The summary line below also contains a dash, so we assert on the row only.)
    data_row = out_lines[1]
    assert data_row.count("—") == 2


def test_show_detail_truncates_long_stderr_in_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    long_stderr = "ValueError: " + "x" * 200
    entries = [_show_entry(0, "failed", stderr=long_stderr)]
    path = _write_show_manifest(tmp_path, entries)

    cli.main(["show", "--detail", str(path)])
    out = capsys.readouterr().out
    assert "..." in out
    # No single line in the table should exceed header + truncated stderr column +
    # other columns; in particular, the full 200-char stderr must not appear.
    assert "x" * 200 not in out


def test_show_detail_filter_failed_keeps_only_failed_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        _show_entry(0, "ok"),
        _show_entry(1, "failed", stderr="boom"),
        _show_entry(2, "skipped"),
        _show_entry(3, "failed", stderr="bang"),
    ]
    path = _write_show_manifest(tmp_path, entries)

    rc = cli.main(["show", "--detail", "--filter", "failed", str(path)])
    assert rc == 0
    out_lines = capsys.readouterr().out.splitlines()
    # Header + 2 failed rows + summary (which still reflects all 4 runs).
    assert len(out_lines) == 4
    assert all(" failed " in line for line in out_lines[1:3])
    assert out_lines[-1].startswith("4 runs")


def test_show_filter_without_detail_exits_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_show_manifest(tmp_path, [_show_entry(0, "ok")])
    rc = cli.main(["show", "--filter", "ok", str(path)])
    assert rc == cli.EXIT_CONFIG
    assert "--filter requires --detail" in capsys.readouterr().err


def test_show_detail_and_run_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_show_manifest(tmp_path, [_show_entry(0, "ok")])
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["show", "--detail", "--run", "0", str(path)])
    assert exc_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_show_run_prints_full_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    multiline_stderr = "ValueError: out of range\nTraceback details here"
    log_path = tmp_path / "run-7" / "worker.log"
    entries = [
        _show_entry(0, "ok"),
        _show_entry(
            7,
            "failed",
            stderr=multiline_stderr,
            log_path=log_path,
            overrides={"Sat.SMA": 8000.0, "Sat.ECC": 0.1},
        ),
    ]
    path = _write_show_manifest(tmp_path, entries)

    rc = cli.main(["show", "--run", "7", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id: 7" in out
    assert "status" in out and "failed" in out
    # str(Path) uses native separators (\\ on Windows, / elsewhere); compare on str.
    assert str(log_path) in out
    # Full unsuppressed stderr — both lines must be present.
    assert "ValueError: out of range" in out
    assert "Traceback details here" in out
    # Override dict, sorted keys, repr values.
    assert "Sat.ECC" in out
    assert "Sat.SMA" in out
    assert "8000.0" in out


def test_show_run_for_ok_entry_reports_no_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_show_manifest(tmp_path, [_show_entry(0, "ok")])
    rc = cli.main(["show", "--run", "0", str(path)])
    assert rc == 0
    assert "(no stderr)" in capsys.readouterr().out


def test_show_run_unknown_run_id_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_show_manifest(tmp_path, [_show_entry(0, "ok"), _show_entry(1, "ok")])
    rc = cli.main(["show", "--run", "999", str(path)])
    assert rc == cli.EXIT_MANIFEST
    assert "run_id 999 not found in manifest" in capsys.readouterr().err


def test_show_no_flags_byte_for_byte_matches_format_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [_show_entry(0, "ok"), _show_entry(1, "failed", stderr="boom")]
    path = _write_show_manifest(tmp_path, entries)

    rc = cli.main(["show", str(path)])
    assert rc == 0
    expected = cli._format_summary(Manifest.load(path), path.parent)
    assert capsys.readouterr().out == expected + "\n"


def test_run_maps_backend_error_to_exit_code(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gmat_sweep.errors import BackendError

    script = _write_script(tmp_path)

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise BackendError("worker pool died")

    monkeypatch.setattr("gmat_sweep.cli.sweep", _boom)

    rc = cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_BACKEND
    assert "backend error" in capsys.readouterr().err


def test_run_maps_generic_gmat_sweep_error_to_exit_code(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gmat_sweep.errors import GmatSweepError

    script = _write_script(tmp_path)

    class _Custom(GmatSweepError):
        pass

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise _Custom("something else broke")

    monkeypatch.setattr("gmat_sweep.cli.sweep", _boom)

    rc = cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_GENERIC
    assert "something else broke" in capsys.readouterr().err


def test_show_on_corrupt_file_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "manifest.jsonl"
    bad.write_text("not json at all\n", encoding="utf-8")

    rc = cli.main(["show", str(bad)])
    assert rc == cli.EXIT_MANIFEST
    assert "manifest" in capsys.readouterr().err


# ---- top-level --help / no args -----------------------------------------


def test_main_with_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    assert "gmat-sweep" in capsys.readouterr().out


def test_main_with_no_args_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2


def test_run_help_lists_grid_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--grid" in captured.out
    assert "lo:hi:count" in captured.out


def test_main_help_lists_all_six_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for subcommand in ("run", "show", "monte-carlo", "latin-hypercube", "explicit", "resume"):
        assert subcommand in out


# ---- _parse_perturb_spec ------------------------------------------------


def test_parse_perturb_spec_normal_returns_tuple_dist() -> None:
    name, dist = cli._parse_perturb_spec("Sat.SMA=normal:7100:50")
    assert name == "Sat.SMA"
    assert dist == ("normal", 7100.0, 50.0)


def test_parse_perturb_spec_uniform_returns_tuple_dist() -> None:
    name, dist = cli._parse_perturb_spec("Sat.INC=uniform:0:90")
    assert (name, dist) == ("Sat.INC", ("uniform", 0.0, 90.0))


def test_parse_perturb_spec_lognormal_returns_tuple_dist() -> None:
    _, dist = cli._parse_perturb_spec("Sat.DryMass=lognormal:5:0.1")
    assert dist == ("lognormal", 5.0, 0.1)


def test_parse_perturb_spec_rejects_missing_equals() -> None:
    with pytest.raises(SweepConfigError, match="must contain '='"):
        cli._parse_perturb_spec("Sat.SMA")


def test_parse_perturb_spec_rejects_empty_name() -> None:
    with pytest.raises(SweepConfigError, match="missing a name"):
        cli._parse_perturb_spec("=normal:1:2")


def test_parse_perturb_spec_rejects_empty_rhs() -> None:
    with pytest.raises(SweepConfigError, match="no distribution"):
        cli._parse_perturb_spec("Sat.SMA=")


def test_parse_perturb_spec_rejects_wrong_arity() -> None:
    with pytest.raises(SweepConfigError, match="three colon-separated"):
        cli._parse_perturb_spec("Sat.SMA=normal:7100")


def test_parse_perturb_spec_rejects_unknown_tag() -> None:
    with pytest.raises(SweepConfigError, match="unknown perturb distribution tag 'zaphod'"):
        cli._parse_perturb_spec("Sat.SMA=zaphod:1:2")


def test_parse_perturb_spec_rejects_non_numeric_params() -> None:
    with pytest.raises(SweepConfigError, match="must be numeric"):
        cli._parse_perturb_spec("Sat.SMA=normal:foo:50")


# ---- _load_samples ------------------------------------------------------


def test_load_samples_csv(tmp_path: Path) -> None:
    csv = tmp_path / "samples.csv"
    csv.write_text("Sat.SMA,Sat.ECC\n7100,0.01\n7200,0.02\n", encoding="utf-8")
    df = cli._load_samples(csv)
    assert list(df.columns) == ["Sat.SMA", "Sat.ECC"]
    assert len(df) == 2


def test_load_samples_parquet(tmp_path: Path) -> None:
    df_in = pd.DataFrame({"Sat.SMA": [7100, 7200], "Sat.ECC": [0.01, 0.02]})
    parquet = tmp_path / "samples.parquet"
    df_in.to_parquet(parquet)
    df_out = cli._load_samples(parquet)
    assert list(df_out.columns) == ["Sat.SMA", "Sat.ECC"]
    assert len(df_out) == 2


def test_load_samples_rejects_unknown_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "samples.txt"
    bad.write_text("Sat.SMA\n7100\n", encoding="utf-8")
    with pytest.raises(SweepConfigError, match="not supported"):
        cli._load_samples(bad)


def test_load_samples_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SweepConfigError, match="not found"):
        cli._load_samples(tmp_path / "no-such-file.csv")


# ---- monte-carlo subcommand ---------------------------------------------


def test_monte_carlo_writes_manifest_and_prints_summary(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "3",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--seed",
            "42",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    assert (out / "manifest.jsonl").exists()
    captured = capsys.readouterr()
    assert "3 runs" in captured.out
    assert "3 ok" in captured.out


def test_monte_carlo_with_two_perturb_flags(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "2",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--perturb",
            "Sat.INC=uniform:0:90",
            "--seed",
            "0",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.run_count == 2
    assert sorted(manifest.parameter_spec["perturb"].keys()) == ["Sat.INC", "Sat.SMA"]


def test_monte_carlo_rejects_unknown_perturb_tag(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "1",
            "--perturb",
            "Sat.SMA=zaphod:1:2",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "zaphod" in err


def test_monte_carlo_rejects_duplicate_perturb_axis(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "1",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--perturb",
            "Sat.SMA=uniform:6000:8000",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "more than once" in capsys.readouterr().err


def test_monte_carlo_rejects_missing_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "1",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--out",
            str(tmp_path / "out"),
            str(tmp_path / "does-not-exist.script"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


# ---- latin-hypercube subcommand -----------------------------------------


def test_latin_hypercube_writes_manifest_and_prints_summary(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "latin-hypercube",
            "--n",
            "4",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--seed",
            "42",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.run_count == 4
    assert manifest.parameter_spec["_kind"] == "latin_hypercube"
    captured = capsys.readouterr()
    assert "4 runs" in captured.out


def test_latin_hypercube_rejects_missing_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        [
            "latin-hypercube",
            "--n",
            "2",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--out",
            str(tmp_path / "out"),
            str(tmp_path / "does-not-exist.script"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


# ---- explicit subcommand ------------------------------------------------


def test_explicit_with_csv_samples(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    samples = tmp_path / "samples.csv"
    samples.write_text("Sat.SMA\n7100\n7200\n7300\n", encoding="utf-8")
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(samples),
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.run_count == 3
    assert manifest.parameter_spec["_kind"] == "explicit"
    captured = capsys.readouterr()
    assert "3 runs" in captured.out


def test_explicit_with_parquet_samples(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    samples = tmp_path / "samples.parquet"
    pd.DataFrame({"Sat.SMA": [7100, 7200]}).to_parquet(samples)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(samples),
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.run_count == 2


def test_explicit_rejects_missing_samples_file(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(tmp_path / "no-such-samples.csv"),
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "not found" in capsys.readouterr().err


def test_explicit_rejects_unknown_samples_suffix(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("Sat.SMA\n7100\n", encoding="utf-8")
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(samples),
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "not supported" in capsys.readouterr().err


def test_explicit_rejects_missing_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    samples = tmp_path / "samples.csv"
    samples.write_text("Sat.SMA\n7100\n", encoding="utf-8")
    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(samples),
            "--out",
            str(tmp_path / "out"),
            str(tmp_path / "does-not-exist.script"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


# ---- resume subcommand --------------------------------------------------


def test_resume_replays_failed_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    # First run: one of the two runs fails.
    def _setitem(_key: str, value: Any) -> None:
        if value == 8000.0:
            raise ValueError("first-pass rejection")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())
    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    assert rc == 0
    capsys.readouterr()  # discard first summary

    # Resume with no setitem failure: the failed run should now succeed.
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    rc = cli.main(
        [
            "resume",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    summary_line = captured.out.splitlines()[0]
    breakdown = summary_line.split("(", 1)[1].split(")", 1)[0]
    assert "2 runs" in summary_line
    assert breakdown == "2 ok"


def test_resume_on_missing_manifest_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    rc = cli.main(
        [
            "resume",
            str(tmp_path / "no-such-manifest.jsonl"),
            "--script",
            str(script),
        ]
    )
    assert rc == cli.EXIT_MANIFEST
    assert "not found" in capsys.readouterr().err


def test_resume_on_missing_script_exits_config_code(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    rc = cli.main(
        [
            "resume",
            str(out / "manifest.jsonl"),
            "--script",
            str(tmp_path / "does-not-exist.script"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


def test_resume_rejects_script_drift(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    # Mutate the script so its canonical hash drifts.
    script.write_text(
        "% GMAT mission\nCreate Spacecraft Sat;\nCreate ForceModel FM;\n", encoding="utf-8"
    )

    rc = cli.main(
        [
            "resume",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script hash mismatch" in capsys.readouterr().err


def test_resume_help_lists_allow_script_drift(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["resume", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--allow-script-drift" in out
    assert "--script" in out


# ---- extend subcommand --------------------------------------------------


def test_extend_appends_new_runs_and_prints_combined_summary(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    rc = cli.main(
        [
            "monte-carlo",
            "--n",
            "4",
            "--perturb",
            "Sat.SMA=normal:7100:50",
            "--seed",
            "42",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = cli.main(
        [
            "extend",
            str(out / "manifest.jsonl"),
            "--n",
            "6",
            "--script",
            str(script),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    summary = capsys.readouterr().out.splitlines()[0]
    assert "10 runs" in summary  # 4 original + 6 extended
    assert "10 ok" in summary


def test_extend_on_missing_manifest_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    rc = cli.main(
        [
            "extend",
            str(tmp_path / "no-such-manifest.jsonl"),
            "--n",
            "2",
            "--script",
            str(script),
        ]
    )
    assert rc == cli.EXIT_MANIFEST
    assert "not found" in capsys.readouterr().err


def test_extend_on_grid_manifest_exits_config_code(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    """A grid manifest is structurally valid but extend refuses it — the
    SweepConfigError surfaces as exit code 2."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:7100:2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    rc = cli.main(
        [
            "extend",
            str(out / "manifest.jsonl"),
            "--n",
            "3",
            "--script",
            str(script),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "Monte Carlo" in capsys.readouterr().err


def test_extend_help_lists_allow_script_drift(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["extend", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--allow-script-drift" in out
    assert "--script" in out
    assert "--n" in out


# ---- archive subcommand -------------------------------------------------


def test_archive_writes_zip_and_matches_api(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI archive output must be byte-equal to the Sweep.archive output —
    proves the CLI is a thin wrapper and that bundles are deterministic."""
    import zipfile

    from gmat_sweep.sweep import Sweep

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:3",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    cli_bundle = tmp_path / "cli.zip"
    rc = cli.main(
        [
            "archive",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--out",
            str(cli_bundle),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "3 runs" in captured.out
    assert str(cli_bundle) in captured.out
    assert cli_bundle.is_file()

    with zipfile.ZipFile(cli_bundle) as zf:
        names = sorted(zf.namelist())
    assert "manifest.jsonl" in names
    assert "MANIFEST.hash" in names
    assert "script/mission.script" in names

    # API and CLI produce byte-equal bundles for the same manifest.
    with LocalJoblibPool(workers=1) as pool:
        api_sweep = Sweep.from_manifest(
            out / "manifest.jsonl", script, backend=pool, progress=False
        )
    api_bundle = api_sweep.archive(tmp_path / "api.zip")
    assert cli_bundle.read_bytes() == api_bundle.read_bytes()


def test_archive_include_logs_bundles_logs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    import zipfile

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000,7100",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    bundle = tmp_path / "with-logs.zip"
    rc = cli.main(
        [
            "archive",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--out",
            str(bundle),
            "--include-logs",
        ]
    )
    assert rc == 0
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert any(name.endswith("/worker.log") for name in names)


def test_archive_on_missing_manifest_exits_manifest_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    rc = cli.main(
        [
            "archive",
            str(tmp_path / "no-such-manifest.jsonl"),
            "--script",
            str(script),
            "--out",
            str(tmp_path / "bundle.zip"),
        ]
    )
    assert rc == cli.EXIT_MANIFEST
    assert "not found" in capsys.readouterr().err


def test_archive_on_missing_script_exits_config_code(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    rc = cli.main(
        [
            "archive",
            str(out / "manifest.jsonl"),
            "--script",
            str(tmp_path / "does-not-exist.script"),
            "--out",
            str(tmp_path / "bundle.zip"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script not found" in capsys.readouterr().err


def test_archive_rejects_script_drift(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    cli.main(
        [
            "run",
            "--grid",
            "a=1,2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    capsys.readouterr()

    script.write_text(
        "% GMAT mission\nCreate Spacecraft Sat;\nCreate ForceModel FM;\n", encoding="utf-8"
    )

    rc = cli.main(
        [
            "archive",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--out",
            str(tmp_path / "bundle.zip"),
        ]
    )
    assert rc == cli.EXIT_CONFIG
    assert "script hash mismatch" in capsys.readouterr().err


def test_archive_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["archive", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--include-logs" in out
    assert "--allow-script-drift" in out
    assert "--out" in out


# ---- _parse_backend_arg -------------------------------------------------


def test_parse_backend_arg_int_value() -> None:
    assert cli._parse_backend_arg("threads_per_worker=2") == ("threads_per_worker", 2)


def test_parse_backend_arg_float_value() -> None:
    assert cli._parse_backend_arg("memory_limit=1.5") == ("memory_limit", 1.5)


def test_parse_backend_arg_string_fallback() -> None:
    assert cli._parse_backend_arg("address=ray://host:10001") == (
        "address",
        "ray://host:10001",
    )


def test_parse_backend_arg_rejects_missing_equals() -> None:
    with pytest.raises(SweepConfigError, match="must be 'KEY=VALUE'"):
        cli._parse_backend_arg("threads_per_worker")


def test_parse_backend_arg_rejects_empty_key() -> None:
    with pytest.raises(SweepConfigError, match="missing a key"):
        cli._parse_backend_arg("=2")


def test_parse_backend_arg_rejects_empty_value() -> None:
    with pytest.raises(SweepConfigError, match="has no value"):
        cli._parse_backend_arg("threads_per_worker=")


# ---- _build_pool / --backend wiring -------------------------------------

# A recording fake stands in for DaskPool / RayPool: subclasses LocalJoblibPool
# so the rest of the CLI pipeline (submit / as_completed / manifest write)
# still runs in-process via joblib(n_jobs=1), but every constructor kwarg the
# CLI passed is captured for assertion. Tests opt into the recording behavior
# by monkey-patching cli.DaskPool / cli.RayPool to a fresh subclass per test
# (so .calls doesn't bleed across tests).


def _make_recording_pool_class() -> Any:
    """Return a fresh recording-fake Pool class. Each call yields a new class.

    The returned class records every constructor kwargs dict on its
    ``calls`` attribute and routes execution through
    :class:`LocalJoblibPool` with ``workers=1`` so the surrounding CLI
    pipeline (sweep → submit → as_completed → manifest write) runs
    synchronously in the test process.
    """

    class _RecordingPool(LocalJoblibPool):
        calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, **kwargs: Any) -> None:
            type(self).calls.append(dict(kwargs))
            super().__init__(workers=1)

    return _RecordingPool


def _make_missing_extra_pool_class(message: str) -> Any:
    """Return a fake that raises :class:`BackendError` on construction."""
    from gmat_sweep.errors import BackendError as _BackendError

    class _MissingExtraPool:
        def __init__(self, **_kwargs: Any) -> None:
            raise _BackendError(message)

    return _MissingExtraPool


def test_build_pool_local_default_returns_local_joblib_pool() -> None:
    args = argparse.Namespace(backend="local", workers=-1, backend_arg=[])
    with cli._build_pool(args) as pool:
        assert isinstance(pool, LocalJoblibPool)


def test_build_pool_local_rejects_backend_arg() -> None:
    args = argparse.Namespace(backend="local", workers=-1, backend_arg=["foo=1"])
    with pytest.raises(SweepConfigError, match="not supported with --backend local"):
        cli._build_pool(args)


def test_build_pool_dask_constructs_dask_pool_with_workers_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "DaskPool", fake)

    args = argparse.Namespace(
        backend="dask",
        workers=4,
        backend_arg=["threads_per_worker=2"],
    )
    with cli._build_pool(args):
        pass

    assert fake.calls == [{"n_workers": 4, "threads_per_worker": 2}]


def test_build_pool_dask_negative_workers_maps_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "DaskPool", fake)

    args = argparse.Namespace(backend="dask", workers=-1, backend_arg=[])
    with cli._build_pool(args):
        pass

    assert fake.calls == [{"n_workers": None}]


def test_build_pool_ray_constructs_ray_pool_with_num_cpus_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "RayPool", fake)

    args = argparse.Namespace(
        backend="ray",
        workers=8,
        backend_arg=["address=ray://host:10001"],
    )
    with cli._build_pool(args):
        pass

    assert fake.calls == [{"num_cpus": 8, "address": "ray://host:10001"}]


def test_build_pool_mpi_forwards_kwargs_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--backend mpi`` ignores ``--workers`` and forwards every ``--backend-arg``.

    Rank count is set externally — by ``mpirun -n K`` under the pre-allocated
    launch mode, or by ``--backend-arg max_workers=N`` under dynamic-spawn.
    The CLI does not synthesise a ``max_workers`` from ``--workers``.
    """
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "MPIPool", fake)

    args = argparse.Namespace(
        backend="mpi",
        workers=8,
        backend_arg=["max_workers=4"],
    )
    with cli._build_pool(args):
        pass

    assert fake.calls == [{"max_workers": 4}]


def test_build_pool_mpi_no_backend_arg_constructs_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--backend mpi`` with no ``--backend-arg`` constructs ``MPIPool()``."""
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "MPIPool", fake)

    args = argparse.Namespace(backend="mpi", workers=-1, backend_arg=[])
    with cli._build_pool(args):
        pass

    assert fake.calls == [{}]


def test_build_pool_rejects_duplicate_backend_arg() -> None:
    args = argparse.Namespace(
        backend="dask",
        workers=-1,
        backend_arg=["threads_per_worker=2", "threads_per_worker=4"],
    )
    with pytest.raises(SweepConfigError, match="given more than once"):
        cli._build_pool(args)


@pytest.mark.parametrize(
    "subcommand,extra_args",
    [
        ("run", ["--grid", "Sat.SMA=7000:8000:2"]),
        ("monte-carlo", ["--n", "2", "--perturb", "Sat.SMA=normal:7100:50", "--seed", "0"]),
        ("latin-hypercube", ["--n", "2", "--perturb", "Sat.SMA=normal:7100:50", "--seed", "0"]),
    ],
)
@pytest.mark.parametrize(
    "backend,fake_attr,worker_kw",
    [
        ("dask", "DaskPool", "n_workers"),
        ("ray", "RayPool", "num_cpus"),
    ],
)
def test_sweep_running_subcommands_route_through_selected_backend(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    extra_args: list[str],
    backend: str,
    fake_attr: str,
    worker_kw: str,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, fake_attr, fake)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    script = _write_script(tmp_path)
    out = tmp_path / "out"

    rc = cli.main(
        [
            subcommand,
            *extra_args,
            "--workers",
            "1",
            "--backend",
            backend,
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    assert fake.calls == [{worker_kw: 1}]


def test_explicit_subcommand_routes_through_selected_backend(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "DaskPool", fake)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    samples = tmp_path / "samples.csv"
    samples.write_text("Sat.SMA\n7000\n7100\n", encoding="utf-8")

    rc = cli.main(
        [
            "explicit",
            "--samples",
            str(samples),
            "--workers",
            "1",
            "--backend",
            "dask",
            "--backend-arg",
            "threads_per_worker=2",
            "--out",
            str(out),
            str(script),
        ]
    )

    assert rc == 0
    assert fake.calls == [{"n_workers": 1, "threads_per_worker": 2}]


def test_run_with_backend_dask_missing_extra_exits_backend_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "DaskPool",
        _make_missing_extra_pool_class(
            "DaskPool requires the [dask] extra: pip install gmat-sweep[dask]"
        ),
    )

    script = _write_script(tmp_path)

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--backend",
            "dask",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_BACKEND
    err = capsys.readouterr().err
    assert "backend error" in err
    assert "[dask]" in err


def test_run_with_backend_ray_missing_extra_exits_backend_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "RayPool",
        _make_missing_extra_pool_class(
            "RayPool requires the [ray] extra: pip install gmat-sweep[ray]"
        ),
    )

    script = _write_script(tmp_path)

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--backend",
            "ray",
            "--out",
            str(tmp_path / "out"),
            str(script),
        ]
    )
    assert rc == cli.EXIT_BACKEND
    err = capsys.readouterr().err
    assert "backend error" in err
    assert "[ray]" in err


def test_show_does_not_accept_backend_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["show", "--backend", "dask", "/tmp/manifest.jsonl"])
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_run_with_invalid_backend_choice_rejected_by_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "run",
                "--grid",
                "a=1,2",
                "--backend",
                "spark",
                "--out",
                "/tmp/out",
                "/tmp/m.script",
            ]
        )
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_resume_subcommand_accepts_backend_flag(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "DaskPool", fake)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    script = _write_script(tmp_path)
    out = tmp_path / "out"

    # First, run a sweep to produce a manifest the resume can pick up.
    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    assert rc == 0
    fake.calls.clear()  # discard any local-path noise (none expected, but be defensive)

    rc = cli.main(
        [
            "resume",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--workers",
            "1",
            "--backend",
            "dask",
        ]
    )
    assert rc == 0
    assert fake.calls == [{"n_workers": 1}]


def test_resume_subcommand_forwards_backend_arg_kwargs(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resume`` routes through ``_build_pool`` like every other sweep-running
    subcommand — ``--backend-arg KEY=VALUE`` must reach the pool constructor."""
    fake = _make_recording_pool_class()
    monkeypatch.setattr(cli, "DaskPool", fake)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    script = _write_script(tmp_path)
    out = tmp_path / "out"

    rc = cli.main(
        [
            "run",
            "--grid",
            "Sat.SMA=7000:8000:2",
            "--workers",
            "1",
            "--out",
            str(out),
            str(script),
        ]
    )
    assert rc == 0
    fake.calls.clear()

    rc = cli.main(
        [
            "resume",
            str(out / "manifest.jsonl"),
            "--script",
            str(script),
            "--workers",
            "1",
            "--backend",
            "dask",
            "--backend-arg",
            "threads_per_worker=2",
        ]
    )
    assert rc == 0
    assert fake.calls == [{"n_workers": 1, "threads_per_worker": 2}]
