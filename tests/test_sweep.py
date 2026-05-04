"""Tests for gmat_sweep.sweep — orchestrator wiring, manifest fsync, ctrl-c safety."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.manifest import Manifest, canonical_script_sha256
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeResults


def _write_script(tmp_path: Path, name: str = "mission.script") -> Path:
    path = tmp_path / name
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


def _payload_run_hook(rows: int = 1) -> Any:
    payload = pd.DataFrame(
        {
            "time": pd.to_datetime([f"2026-05-04T00:00:0{i}" for i in range(rows)]),
            "x": [float(i) for i in range(rows)],
        }
    )

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- run() basics ---------------------------------------------------------


def test_sweep_run_returns_self(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0]},
            progress=False,
        )
        assert sweep.run() is sweep


def test_sweep_run_writes_one_manifest_entry_per_run(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=4)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0, 7001.0, 7002.0, 7003.0]},
            progress=False,
        ).run()

    manifest = Manifest.load(output_dir / "manifest.jsonl")
    assert manifest.run_count == 4
    assert len(manifest.entries) == 4
    assert {e.run_id for e in manifest.entries} == {0, 1, 2, 3}
    by_run_id = {e.run_id: e for e in manifest.entries}
    for run_id in range(4):
        entry = by_run_id[run_id]
        assert entry.status == "ok"
        assert entry.overrides == {"Sat.SMA": 7000.0 + run_id}
        assert entry.log_path == output_dir / f"run-{run_id}" / "worker.log"


def test_sweep_manifest_header_carries_script_hash_seed_and_spec(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path, name="m.script")
    expected_sha = canonical_script_sha256(script)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0]},
            sweep_seed=1729,
            progress=False,
        ).run()

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert reloaded.script_sha256 == expected_sha
    assert reloaded.sweep_seed == 1729
    assert reloaded.parameter_spec == {"Sat.SMA": [7000.0]}
    assert reloaded.run_count == 1
    # gmat_sweep_version is the package version pulled at import time —
    # asserting it equals the live module value is the cheapest way to confirm
    # it isn't a stale string literal in sweep.py.
    import gmat_sweep

    assert reloaded.gmat_sweep_version == gmat_sweep.__version__


def test_sweep_manifest_parent_directory_is_created(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "deeply" / "nested" / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    assert (output_dir / "manifest.jsonl").exists()


# ---- aggregation ---------------------------------------------------------


def test_sweep_to_dataframe_returns_multiindexed_frame(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=3)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=2))

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0, 7001.0, 7002.0]},
            progress=False,
        ).run()

    df = sweep.to_dataframe()
    assert df.index.names == ["run_id", "time"]
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2]
    assert (df["__status"] == "ok").all()


def test_sweep_to_dataframe_marks_failed_run(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=3)

    def _setitem(_key: str, value: Any) -> None:
        if value == 7001.0:
            raise ValueError("rejected by GMAT")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook(rows=1))

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    df = sweep.to_dataframe()
    assert set(df["__status"].unique()) == {"ok", "failed"}
    failed_rows = df.loc[df["__status"] == "failed"]
    assert len(failed_rows) == 1
    assert failed_rows.index.get_level_values("run_id").tolist() == [1]


def test_sweep_to_manifest_requires_run_first(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        )
        with pytest.raises(RuntimeError, match="run"):
            sweep.to_manifest()


# ---- progress -------------------------------------------------------------


def test_sweep_progress_disabled_quiet_on_stderr(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=2)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    captured = capsys.readouterr()
    # tqdm draws a progress bar by writing carriage returns and percentages
    # to stderr; with progress=False neither should appear.
    assert "gmat-sweep" not in captured.err
    assert "%" not in captured.err


# ---- ctrl-c safety --------------------------------------------------------


class _InterruptingPool(Pool):
    """Pool that yields N outcomes successfully, then raises KeyboardInterrupt.

    Used to drive the partial-manifest assertion without standing up a real
    subprocess pool — the actual loky path is exercised separately in
    test_backends_joblib.
    """

    def __init__(self, *, yield_count: int) -> None:
        self._submitted: list[RunSpec] = []
        self._yield_count = yield_count

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        self._submitted.append(spec)
        return Future()

    def as_completed(self, _futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        now = datetime.now(timezone.utc)
        for spec in self._submitted[: self._yield_count]:
            yield RunOutcome.ok(
                run_id=spec.run_id,
                output_paths={},
                started_at=now,
                ended_at=now,
            )
        raise KeyboardInterrupt

    def close(self) -> None:
        pass


def test_sweep_keyboard_interrupt_leaves_parsable_partial_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=4)

    pool = _InterruptingPool(yield_count=2)
    sweep = Sweep(
        runs=runs,
        backend=pool,
        manifest_path=output_dir / "manifest.jsonl",
        output_dir=output_dir,
        script_path=script,
        parameter_spec={},
        progress=False,
    )
    with pytest.raises(KeyboardInterrupt):
        sweep.run()

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert len(reloaded.entries) == 2
    assert {e.run_id for e in reloaded.entries} == {0, 1}
