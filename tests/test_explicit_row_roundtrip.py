"""End-to-end explicit-row sweep round-trip — v0.2 validation suite (issue #40).

Pins the two halves of the explicit-row contract (#34): the worker observes
every input row's override via ``mission[key] = value`` *before* ``mission.run()``
is called for that run, and the manifest's ``parameter_spec`` reconstructs a
DataFrame whose contents equal the input.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from gmat_sweep.api import sweep
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.manifest import Manifest
from tests.conftest import FakeGmatRun, FakeMission, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def test_each_row_value_observed_via_mission_setitem_before_run(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Every ``Sat.SMA`` value declared in the input DataFrame must be applied
    to the live mission via ``mission["Sat.SMA"] = value`` before ``mission.run()``
    is called for that run.

    We verify both the *what* (every input value reaches a mission) and the
    *order* (within one mission, the setitem precedes the run call) — the
    second is what makes the override actually take effect on the GMAT engine
    rather than being applied to a no-op copy.
    """
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    sma_values = [7000.0, 7100.0, 7200.0, 7300.0]

    # Per-mission observations: each load() captures the (set-before-run) state
    # of its own mission, keyed by working_dir's run-N suffix.
    per_run_observations: dict[int, dict[str, Any]] = {}

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(script_path=Path(path))

        def _run(*, working_dir: Path, **_kwargs: Any) -> FakeResults:
            run_id = int(working_dir.name.removeprefix("run-"))
            # Snapshot the mission's overrides_log at the moment run() is called —
            # i.e. every setitem that happened on this mission so far. The fact
            # that overrides_log is non-empty here is the "before run()" assertion.
            per_run_observations[run_id] = {k: v for k, v in mission.overrides_log}
            return FakeResults(
                reports={
                    "R": pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})
                }
            )

        mission.run_hook = _run
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    samples = pd.DataFrame({"Sat.SMA": sma_values})
    sweep(script, samples=samples, backend=LocalJoblibPool(workers=1), out=out, progress=False)

    # Every run_id observed exactly its own row's Sat.SMA, set before run().
    assert set(per_run_observations.keys()) == {0, 1, 2, 3}
    for run_id, expected_sma in enumerate(sma_values):
        assert per_run_observations[run_id] == {"Sat.SMA": expected_sma}


def test_manifest_parameter_spec_reconstructs_input_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """``Manifest.load(...).parameter_spec`` carries the explicit-row sweep's
    ``columns`` and ``rows`` such that
    ``pd.DataFrame(spec["rows"], columns=spec["columns"])`` reconstructs a
    DataFrame whose contents equal the input."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(
        run_hook=lambda **_: FakeResults(
            reports={
                "R": pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})
            }
        )
    )

    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0],
            "Sat.ECC": [0.001, 0.002, 0.003, 0.004],
        }
    )
    sweep(script, samples=samples, backend=LocalJoblibPool(workers=1), out=out, progress=False)

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    assert spec["_kind"] == "explicit"

    reconstructed = pd.DataFrame(spec["rows"], columns=spec["columns"])
    pd.testing.assert_frame_equal(reconstructed, samples)
