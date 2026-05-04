"""Tests for gmat_sweep.api.sweep — end-to-end wrapper over Sweep + LocalJoblibPool."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.api import sweep
from gmat_sweep.manifest import Manifest
from tests.conftest import FakeGmatRun, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _payload_run_hook(value: float = 1.0) -> Any:
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [value]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- explicit out --------------------------------------------------------


def test_sweep_returns_multiindexed_dataframe(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        workers=1,
        out=out,
    )

    assert df.index.names == ["run_id", "time"]
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1]
    assert (out / "manifest.jsonl").exists()


def test_sweep_writes_manifest_with_grid_in_header(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        workers=1,
        out=out,
        seed=42,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.parameter_spec == {"Sat.SMA": [7000.0, 7100.0]}
    assert manifest.sweep_seed == 42
    assert manifest.run_count == 2


def test_sweep_string_path_is_accepted(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(
        str(script),
        grid={"Sat.SMA": [7000.0]},
        workers=1,
        out=out,
    )

    assert len(df) == 1


def test_sweep_creates_out_directory_when_missing(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "does" / "not" / "exist"
    assert not out.exists()

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(script, grid={"Sat.SMA": [7000.0]}, workers=1, out=out)

    assert out.is_dir()
    assert (out / "manifest.jsonl").exists()


# ---- no out: temp dir lifetime --------------------------------------------


def test_sweep_with_no_out_returns_usable_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """lazy_multiindex materialises Parquet in-memory before returning, so the
    DataFrame is usable even after the temp dir would have been GC'd."""
    script = _write_script(tmp_path)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(value=3.14))

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0]},
        workers=1,
        out=None,
    )

    assert df["x"].iloc[0] == 3.14


def test_sweep_with_no_out_cleans_up_temp_dir_after_df_dropped(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the returned DataFrame triggers the weakref finalizer that
    cleans up the sweep-scoped temp directory."""
    import tempfile as _tempfile

    captured_paths: list[Path] = []
    real_tempdir = _tempfile.TemporaryDirectory

    def _spy_tempdir(*args: Any, **kwargs: Any) -> Any:
        td = real_tempdir(*args, **kwargs)
        captured_paths.append(Path(td.name))
        return td

    monkeypatch.setattr("gmat_sweep.api.tempfile.TemporaryDirectory", _spy_tempdir)

    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(script, grid={"Sat.SMA": [7000.0]}, workers=1, out=None)

    assert len(captured_paths) == 1
    sweep_dir = captured_paths[0]
    assert sweep_dir.is_dir()

    del df
    gc.collect()

    assert not sweep_dir.exists()


# ---- failure modes don't raise -------------------------------------------


def test_sweep_with_failing_run_includes_failed_row(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    def _setitem(_key: str, value: Any) -> None:
        if value == 7100.0:
            raise ValueError("rejected")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        workers=1,
        out=out,
    )

    statuses = set(df["__status"].unique())
    assert statuses == {"ok", "failed"}


# ---- empty grid ----------------------------------------------------------


def test_sweep_with_empty_grid_runs_once(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """full_factorial({}) yields a single empty override dict; sweep() honours it."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(script, grid={}, workers=1, out=out)

    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == [0]


# ---- progress -----------------------------------------------------------


def test_sweep_progress_false_quiet_on_stderr(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """progress=False propagates from sweep() through to Sweep so the tqdm
    progress bar does not paint to stderr — needed when notebooks are
    committed with executed outputs (otherwise each tqdm refresh lands as a
    captured stderr snapshot in the .ipynb)."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(script, grid={"Sat.SMA": [7000.0, 7100.0]}, workers=1, out=out, progress=False)

    captured = capsys.readouterr()
    assert "gmat-sweep" not in captured.err
    assert "%" not in captured.err


# ---- module-import-time logger config -----------------------------------


def test_gmat_sweep_logger_default_level_is_warning() -> None:
    """gmat_sweep configures its top-level logger at module import; without it
    every per-run INFO record would land in the parent process's handlers."""
    import logging

    assert logging.getLogger("gmat_sweep").level == logging.WARNING
