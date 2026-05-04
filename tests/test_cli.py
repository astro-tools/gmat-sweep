"""Tests for gmat_sweep.cli — argparse-driven console-script entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep import cli
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest
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
    assert sorted(manifest.parameter_spec.keys()) == ["Sat.ECC", "Sat.SMA"]


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
