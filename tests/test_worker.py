"""Tests for gmat_sweep.worker.run_one — happy path, every failure mode, never raises."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.spec import RunSpec
from gmat_sweep.worker import run_one
from tests.conftest import FakeGmatRun, FakeGmatRunError, FakeResults


def _make_spec(
    *,
    output_dir: Path,
    run_id: int = 0,
    overrides: dict[str, Any] | None = None,
    run_options: dict[str, Any] | None = None,
    seed: int | None = None,
    script_name: str = "mission.script",
) -> RunSpec:
    return RunSpec(
        script_path=Path(f"/missions/{script_name}"),
        overrides=overrides if overrides is not None else {},
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
        run_options=run_options if run_options is not None else {},
    )


# ---- happy path ----------------------------------------------------------


def test_run_one_ok_writes_parquet_and_returns_ok(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    df_a = pd.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
    df_b = pd.DataFrame({"t": [0.0], "y": [42.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R1": df_a, "R2": df_b}, log="GMAT engine log line\n")

    fake_gmat_run.install_loader(run_hook=_run)

    out_dir = tmp_path / "run-0"
    spec = _make_spec(
        output_dir=out_dir,
        overrides={"Sat.SMA": 7000.0, "Sat.ECC": 0.01},
        run_options={"overwrite": True},
    )
    outcome = run_one(spec)

    assert outcome.status == "ok"
    assert outcome.run_id == 0
    assert outcome.stderr is None
    assert outcome.duration_s >= 0
    assert outcome.started_at <= outcome.ended_at
    assert set(outcome.output_paths) == {"R1", "R2"}

    # Parquet round-trip.
    pd.testing.assert_frame_equal(pd.read_parquet(outcome.output_paths["R1"]), df_a)
    pd.testing.assert_frame_equal(pd.read_parquet(outcome.output_paths["R2"]), df_b)

    # Worker log present and contains override + engine-log lines.
    log_text = (out_dir / "worker.log").read_text(encoding="utf-8")
    assert "Sat.SMA" in log_text
    assert "GMAT engine log line" in log_text
    assert "status=ok" in log_text

    # mission.run got the working_dir and the forwarded run_options.
    assert fake_gmat_run.last_mission is not None
    assert fake_gmat_run.last_mission.run_kwargs_log == [
        {"working_dir": out_dir, "overwrite": True}
    ]
    # Overrides were applied in spec order.
    assert fake_gmat_run.last_mission.overrides_log == [
        ("Sat.SMA", 7000.0),
        ("Sat.ECC", 0.01),
    ]


def test_run_one_ok_with_no_reports_returns_empty_output_paths(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    spec = _make_spec(output_dir=tmp_path / "run-0")
    outcome = run_one(spec)

    assert outcome.status == "ok"
    assert outcome.output_paths == {}
    assert outcome.stderr is None
    assert (tmp_path / "run-0" / "worker.log").exists()


def test_run_one_creates_output_dir_when_missing(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    nested = tmp_path / "a" / "b" / "run-0"
    assert not nested.exists()
    outcome = run_one(_make_spec(output_dir=nested))

    assert outcome.status == "ok"
    assert nested.is_dir()


def test_run_one_logs_seed_when_set(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    spec = _make_spec(output_dir=tmp_path / "run-0", seed=1729)
    run_one(spec)

    log_text = (tmp_path / "run-0" / "worker.log").read_text(encoding="utf-8")
    assert "seed=1729" in log_text


# ---- failure modes -------------------------------------------------------


def test_run_one_failed_when_override_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    def _setitem(key: str, _value: Any) -> None:
        if key == "Sat.ECC":
            raise ValueError(f"GMAT rejected write to {key!r}: ECC > 1")

    fake_gmat_run.install_loader(setitem_hook=_setitem)

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir, overrides={"Sat.SMA": 7000.0, "Sat.ECC": 1.5}))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "Sat.ECC" in outcome.stderr
    assert "Traceback" in outcome.stderr
    assert outcome.output_paths == {}
    log_text = (out_dir / "worker.log").read_text(encoding="utf-8")
    assert "status=failed" in log_text


def test_run_one_failed_when_load_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    fake_gmat_run.install_failing_loader(FileNotFoundError("/missions/nope.script does not exist"))

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir, script_name="nope.script"))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "nope.script" in outcome.stderr


def test_run_one_failed_when_run_raises_includes_engine_log(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    engine_log = "GMAT: detailed engine error trace"

    def _run(**_: Any) -> FakeResults:
        raise FakeGmatRunError("RunScript returned status -1", log=engine_log)

    fake_gmat_run.install_loader(run_hook=_run)

    outcome = run_one(_make_spec(output_dir=tmp_path / "run-0"))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "RunScript returned status -1" in outcome.stderr
    assert engine_log in outcome.stderr


def test_run_one_failed_when_gmat_run_import_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Block the import without using the fake_gmat_run fixture: install a
    # meta-path finder that raises ImportError for "gmat_run".
    monkeypatch.delitem(sys.modules, "gmat_run", raising=False)

    class _BlockingFinder:
        def find_spec(self, name: str, _path: Any = None, _target: Any = None) -> None:
            if name == "gmat_run":
                raise ImportError("simulated gmat_run install failure")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "gmat_run" in outcome.stderr


def test_run_one_failed_when_parquet_write_fails(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    df = pd.DataFrame({"t": [0.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R1": df})

    fake_gmat_run.install_loader(run_hook=_run)

    out_dir = tmp_path / "run-0"
    out_dir.mkdir()
    # Block the parquet write by pre-creating R1.parquet as a *directory* —
    # df.to_parquet then fails with IsADirectoryError / PermissionError.
    (out_dir / "R1.parquet").mkdir()

    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "Traceback" in outcome.stderr


# ---- contract: never raises ----------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["load_raises", "setitem_raises", "run_raises", "import_blocked"],
)
def test_run_one_never_raises_for_known_failure_modes(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if scenario == "import_blocked":
        monkeypatch.delitem(sys.modules, "gmat_run", raising=False)

        class _BlockingFinder:
            def find_spec(self, name: str, _path: Any = None, _target: Any = None) -> None:
                if name == "gmat_run":
                    raise ImportError("blocked")
                return None

        monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])
    else:
        fake = request.getfixturevalue("fake_gmat_run")
        if scenario == "load_raises":
            fake.install_failing_loader(RuntimeError("boom"))
        elif scenario == "setitem_raises":

            def _setitem(*_a: Any, **_k: Any) -> None:
                raise RuntimeError("nope")

            fake.install_loader(setitem_hook=_setitem)
        elif scenario == "run_raises":

            def _run(**_: Any) -> FakeResults:
                raise RuntimeError("explosion")

            fake.install_loader(run_hook=_run)

    spec = _make_spec(output_dir=tmp_path / "run-0", overrides={"x": 1})
    # The contract: this call returns; it does not raise.
    outcome = run_one(spec)
    assert outcome.status == "failed"
    assert outcome.stderr


# ---- defensive: a stale handler on the same logger does not bleed across runs ----


def test_run_one_does_not_bleed_handlers_between_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    import logging

    spec_a = _make_spec(output_dir=tmp_path / "run-0", run_id=0)
    spec_b = _make_spec(output_dir=tmp_path / "run-1", run_id=1)
    run_one(spec_a)
    run_one(spec_b)
    run_one(spec_a)

    # Same logger name across the two run_id=0 calls; we should not have
    # accumulated multiple file handlers attached to it.
    logger = logging.getLogger("gmat_sweep.worker.run_0")
    assert logger.handlers == []
