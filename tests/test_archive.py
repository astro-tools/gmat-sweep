"""Sweep.archive end-to-end: layout, hash file, log toggle, and resume round-trip."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.api import sweep
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeMission, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _install_sma_echoing_loader(fake_gmat_run: FakeGmatRun) -> None:
    """Load each run's Sat.SMA into the report ``x`` column so frames carry per-run signal."""

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(script_path=Path(path))

        def _run(**_kwargs: Any) -> FakeResults:
            sma = next(v for k, v in mission.overrides_log if k == "Sat.SMA")
            payload = pd.DataFrame(
                {"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [float(sma)]}
            )
            return FakeResults(reports={"R": payload})

        mission.run_hook = _run
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]


def _run_grid(*, script: Path, out: Path, n: int = 16, fake_gmat_run: FakeGmatRun) -> Sweep:
    _install_sma_echoing_loader(fake_gmat_run)
    grid_values = [7000.0 + i for i in range(n)]
    with LocalJoblibPool(workers=1) as pool:
        sweep_obj = Sweep(
            runs=__import__(
                "gmat_sweep.grids", fromlist=["expand_grid_to_run_specs"]
            ).expand_grid_to_run_specs({"Sat.SMA": grid_values}, script, out),
            backend=pool,
            manifest_path=out / "manifest.jsonl",
            output_dir=out,
            script_path=script,
            parameter_spec={"_kind": "grid", "Sat.SMA": grid_values},
            progress=False,
        ).run()
    return sweep_obj


def _list_zip(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return sorted(zf.namelist())


# ---- Layout & manifest rewrite -------------------------------------------


def test_archive_emits_expected_layout(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=4, fake_gmat_run=fake_gmat_run)

    bundle = sweep_obj.archive(tmp_path / "bundle.zip")
    assert bundle == tmp_path / "bundle.zip"

    names = _list_zip(bundle)
    assert "README.md" in names
    assert "manifest.jsonl" in names
    assert "MANIFEST.hash" in names
    assert "script/mission.script" in names
    for run_id in range(4):
        assert f"runs/run-{run_id}/report__R.parquet" in names
    # No worker logs by default.
    assert not any(name.endswith("/worker.log") for name in names)


def test_bundled_manifest_has_relative_paths_and_no_logs_by_default(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=3, fake_gmat_run=fake_gmat_run)

    bundle = sweep_obj.archive(tmp_path / "bundle.zip")
    with zipfile.ZipFile(bundle) as zf:
        raw = zf.read("manifest.jsonl").decode("utf-8")

    lines = [json.loads(line) for line in raw.splitlines()]
    header, *entries = lines
    assert header["script_sha256"] == sweep_obj.to_manifest().script_sha256

    for entry in entries:
        for path in entry["output_paths"].values():
            assert not Path(path).is_absolute()
            assert path.startswith(f"runs/run-{entry['run_id']}/")
        assert entry["log_path"] is None


def test_archive_include_logs_bundles_logs_and_keeps_log_path(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=2, fake_gmat_run=fake_gmat_run)

    bundle = sweep_obj.archive(tmp_path / "with-logs.zip", include_logs=True)
    names = _list_zip(bundle)
    assert "runs/run-0/worker.log" in names
    assert "runs/run-1/worker.log" in names

    with zipfile.ZipFile(bundle) as zf:
        raw = zf.read("manifest.jsonl").decode("utf-8")
    entries = [json.loads(line) for line in raw.splitlines()[1:]]
    assert entries[0]["log_path"] == f"runs/run-{entries[0]['run_id']}/worker.log"


# ---- MANIFEST.hash format ------------------------------------------------


def test_manifest_hash_file_matches_sha256sum_format(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=2, fake_gmat_run=fake_gmat_run)

    bundle = sweep_obj.archive(tmp_path / "bundle.zip")
    with zipfile.ZipFile(bundle) as zf:
        hash_lines = zf.read("MANIFEST.hash").decode("utf-8").splitlines()
        recorded = {}
        for line in hash_lines:
            digest, _, name = line.partition("  ")
            recorded[name] = digest
            assert len(digest) == 64

        # Every member except MANIFEST.hash is covered, and digests match.
        for name in zf.namelist():
            if name == "MANIFEST.hash":
                assert name not in recorded
                continue
            assert recorded[name] == hashlib.sha256(zf.read(name)).hexdigest()


# ---- Determinism: API and CLI produce byte-equal bundles ------------------


def test_two_archives_of_same_sweep_are_byte_equal(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=4, fake_gmat_run=fake_gmat_run)

    a = sweep_obj.archive(tmp_path / "a.zip")
    b = sweep_obj.archive(tmp_path / "b.zip")
    assert a.read_bytes() == b.read_bytes()


# ---- DoD: 16-run grid round-trips bit-equal across archive + resume -------


def test_16_run_grid_archive_and_resume_yields_bit_equal_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The headline acceptance test from the issue: a 16-run grid sweep,
    archived and then unzipped, resumes to a DataFrame bit-equal to the
    original aggregated frame."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=16, fake_gmat_run=fake_gmat_run)
    original = sweep_obj.to_dataframe()

    bundle = sweep_obj.archive(tmp_path / "bundle.zip")

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(extracted)

    with LocalJoblibPool(workers=1) as pool:
        resumed = (
            Sweep.from_manifest(
                extracted / "manifest.jsonl",
                extracted / "script" / "mission.script",
                backend=pool,
                progress=False,
            )
            .resume()
            .to_dataframe()
        )

    pd.testing.assert_frame_equal(resumed, original)


# ---- Failure / skip path: no Parquet, manifest entry preserved ------------


def test_failed_runs_are_packed_without_parquet_with_stderr_intact(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    failing = {7002.0}

    def _setitem(_key: str, value: Any) -> None:
        if value in failing:
            raise ValueError("rejected by GMAT")

    def _load(path: Any, **_: Any) -> FakeMission:
        def _run(**_kwargs: Any) -> FakeResults:
            return FakeResults(
                reports={
                    "R": pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})
                }
            )

        mission = FakeMission(script_path=Path(path), setitem_hook=_setitem, run_hook=_run)
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7001.0, 7002.0, 7003.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    failed = Manifest.find_failed(out / "manifest.jsonl")
    assert len(failed) == 1
    failed_run_id = failed[0]

    with LocalJoblibPool(workers=1) as pool:
        loaded_sweep = Sweep.from_manifest(
            out / "manifest.jsonl", script, backend=pool, progress=False
        )
    bundle = loaded_sweep.archive(tmp_path / "bundle.zip")

    names = _list_zip(bundle)
    assert not any(f"runs/run-{failed_run_id}/report__" in name for name in names)

    with zipfile.ZipFile(bundle) as zf:
        entries = [
            json.loads(line) for line in zf.read("manifest.jsonl").decode("utf-8").splitlines()[1:]
        ]
    failed_entry = next(e for e in entries if e["run_id"] == failed_run_id)
    assert failed_entry["status"] == "failed"
    assert "rejected by GMAT" in failed_entry["stderr"]


# ---- Drift handling -------------------------------------------------------


def test_archive_refuses_when_script_hash_drifts(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=2, fake_gmat_run=fake_gmat_run)

    script.write_text("% mutated\nCreate Spacecraft Sat;\n", encoding="utf-8")

    with pytest.raises(SweepConfigError, match="script hash mismatch"):
        sweep_obj.archive(tmp_path / "bundle.zip")


def test_archive_before_run_raises_runtime_error(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)
    from gmat_sweep.grids import expand_grid_to_run_specs

    out = tmp_path / "out"
    out.mkdir()
    runs = expand_grid_to_run_specs({"Sat.SMA": [7000.0, 7001.0]}, script, out)

    with LocalJoblibPool(workers=1) as pool:
        sweep_obj = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=out / "manifest.jsonl",
            output_dir=out,
            script_path=script,
            parameter_spec={"_kind": "grid", "Sat.SMA": [7000.0, 7001.0]},
            progress=False,
        )
        with pytest.raises(RuntimeError, match=r"Sweep\.run"):
            sweep_obj.archive(tmp_path / "bundle.zip")


# ---- Generated README -----------------------------------------------------


def test_generated_readme_includes_summary_and_resume_recipe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sweep_obj = _run_grid(script=script, out=out, n=3, fake_gmat_run=fake_gmat_run)

    bundle = sweep_obj.archive(tmp_path / "bundle.zip")
    with zipfile.ZipFile(bundle) as zf:
        readme = zf.read("README.md").decode("utf-8")

    assert "3 ok" in readme
    assert "Script SHA-256" in readme
    assert "gmat-sweep resume manifest.jsonl --script script/mission.script" in readme
    assert "sha256sum -c MANIFEST.hash" in readme
