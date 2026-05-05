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
