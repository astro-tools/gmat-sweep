"""Parameter type round-trip from sweep spec → worker → GMAT → readback.

Covers the five field shapes #11 calls out:

- ``float`` — :class:`Spacecraft` SMA in km, expected within FP tolerance.
- ``int`` — :class:`Spacecraft` NAIFId, exact integer round-trip.
- ``datetime`` epoch — :class:`Spacecraft` Epoch, GMAT's wire format is a
  string (``'15 Jan 2026 12:34:56.789'``); the v0.1 contract is that whatever
  string the user puts on the spec is what the worker hands to GMAT.
- vector burn DV components — :class:`ImpulsiveBurn` Element1 / 2 / 3, set
  per-element (no whole-vector setter on the gmat-run surface today).
- ``str`` enum — :class:`Spacecraft` DisplayStateType, exact string round-trip.

Each case loads the same fixture mission as the worker would, applies the
override the same way the worker does (``mission[key] = value``), and reads it
back via ``mission[key]`` *before* ``mission.run()`` — which is the sentence
the issue's acceptance line specifies. We don't actually propagate; the worker
boundary we care about is the override application, not the propagation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

gmat_run = pytest.importorskip("gmat_run")


@pytest.fixture
def loaded_mission(leo_basic_script: Path) -> object:
    """Fresh :class:`gmat_run.Mission` per test — overrides mutate the live GMAT instance."""
    return gmat_run.Mission.load(leo_basic_script)


# ---- float --------------------------------------------------------------


def test_float_sma_round_trip(loaded_mission: object) -> None:
    loaded_mission["Sat.SMA"] = 7250.5  # type: ignore[index]
    assert loaded_mission["Sat.SMA"] == pytest.approx(7250.5, abs=1e-9)  # type: ignore[index]


def test_float_negative_round_trip(loaded_mission: object) -> None:
    # Negative DV element — gmat-sweep must not strip sign on the worker side.
    loaded_mission["DV.Element1"] = -0.123  # type: ignore[index]
    assert loaded_mission["DV.Element1"] == pytest.approx(-0.123, abs=1e-12)  # type: ignore[index]


# ---- int ----------------------------------------------------------------


def test_int_naif_id_round_trip(loaded_mission: object) -> None:
    loaded_mission["Sat.NAIFId"] = -42  # type: ignore[index]
    assert loaded_mission["Sat.NAIFId"] == -42  # type: ignore[index]


# ---- datetime / epoch ---------------------------------------------------


def test_epoch_string_round_trip(loaded_mission: object) -> None:
    epoch = "15 Jan 2026 12:34:56.789"
    loaded_mission["Sat.Epoch"] = epoch  # type: ignore[index]
    assert loaded_mission["Sat.Epoch"] == epoch  # type: ignore[index]


# ---- vector burn DV components ------------------------------------------


def test_burn_vector_components_round_trip(loaded_mission: object) -> None:
    components = {"DV.Element1": 0.5, "DV.Element2": -0.25, "DV.Element3": 1e-3}
    for key, value in components.items():
        loaded_mission[key] = value  # type: ignore[index]
    for key, value in components.items():
        assert loaded_mission[key] == pytest.approx(value, abs=1e-12), key  # type: ignore[index]


# ---- str enum -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["Cartesian", "Keplerian", "ModifiedKeplerian", "SphericalAZFPA"],
)
def test_display_state_type_str_enum_round_trip(loaded_mission: object, value: str) -> None:
    loaded_mission["Sat.DisplayStateType"] = value  # type: ignore[index]
    assert loaded_mission["Sat.DisplayStateType"] == value  # type: ignore[index]


# ---- worker-boundary parity --------------------------------------------


def test_worker_applies_overrides_before_run(leo_basic_script: Path, tmp_path: Path) -> None:
    """Run the actual worker against the fixture and assert the override
    landed on the live mission *before* propagation by reading the resulting
    ReportFile (whose values reflect the post-override state).
    """
    import pandas as pd

    from gmat_sweep.spec import RunSpec
    from gmat_sweep.worker import run_one

    spec = RunSpec(
        script_path=leo_basic_script,
        overrides={"Sat.SMA": 7234.5},
        output_dir=tmp_path / "run-0",
        run_id=0,
        seed=None,
        run_options={},
    )
    outcome = run_one(spec)
    assert outcome.status == "ok", outcome.stderr

    df = pd.read_parquet(outcome.output_paths["RF"])
    # The first ReportFile row is the epoch state — Sat.Earth.SMA there
    # must equal the override exactly. If the worker dropped the override
    # (e.g. applied it after run() instead of before), the value would be
    # the script default (7000.0) instead.
    assert df["Sat.Earth.SMA"].iloc[0] == pytest.approx(7234.5, abs=1e-6)
