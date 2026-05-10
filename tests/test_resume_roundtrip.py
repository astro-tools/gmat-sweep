"""End-to-end resume round-trip — v0.2 validation suite (issue #40).

Supersedes the v0.1 forward-only ``test_manifest_replay_contract`` (which
asserted that v0.1 manifests carried every field a future ``resume`` would
need). With ``Sweep.from_manifest()`` and ``Sweep.resume()`` shipped (#36)
the contract is now exercised end-to-end: kill a sweep partway through,
``from_manifest(...).resume()`` after fixing the failure source, assert the
result is what an unbroken sweep would have been.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.api import monte_carlo, sweep
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeMission, FakeResults

_DEFAULT_SCRIPT_BODY = "% GMAT mission\nCreate Spacecraft Sat;\n"


def _write_script(tmp_path: Path, content: str = _DEFAULT_SCRIPT_BODY) -> Path:
    path = tmp_path / "mission.script"
    path.write_text(content, encoding="utf-8")
    return path


def _payload_run_hook() -> Any:
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


def _install_sma_echoing_loader(fake_gmat_run: FakeGmatRun) -> None:
    """Same SMA-echoing loader as test_monte_carlo_determinism — encodes
    each run's Sat.SMA override in the report ``x`` column so post-resume
    DataFrames carry the per-run draw signal."""

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


# ---- 16-run grid with 3 failures, resumed --------------------------------


def test_16_run_grid_with_3_failures_resumes_to_all_ok(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The headline acceptance: 16-run grid sweep with 3 deliberately failing
    runs; after fixing the failure source and calling
    ``Sweep.from_manifest(path).resume()`` the resumed DataFrame has 16
    ``__status="ok"`` rows."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    failing = {7003.0, 7007.0, 7011.0}
    grid_values = [7000.0 + i for i in range(16)]

    def _setitem(_key: str, value: Any) -> None:
        if value in failing:
            raise ValueError("rejected by GMAT")

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(
            script_path=Path(path),
            setitem_hook=_setitem,
            run_hook=_payload_run_hook(),
        )
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    sweep(
        script,
        grid={"Sat.SMA": grid_values},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    assert sorted(Manifest.find_failed(out / "manifest.jsonl")) == [3, 7, 11]

    # Patch the failure source out and resume.
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    with LocalJoblibPool(workers=1) as pool:
        df = (
            Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False)
            .resume()
            .to_dataframe()
        )

    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == list(range(16))
    assert (df["__status"] == "ok").all()


# ---- Monte Carlo resume: bit-equal draws for resumed runs ----------------


def test_monte_carlo_resume_preserves_bit_equal_draws_for_failed_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """When an MC sweep is resumed, the resumed runs must carry the same
    Sat.SMA overrides the originally-failed runs carried — the run-level seed
    is fixed by ``derive_run_seeds(parent_seed, n)[run_id]`` and the resume
    re-applies the same spec, so the draw is bit-equal across the original
    failed attempt and the resumed successful attempt."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    # Pre-compute the deterministic Sat.SMA draw for each run_id and pick a
    # value to fail on, so we know the failure matches at least one specific
    # run_id rather than relying on chance.
    from gmat_sweep.grids import expand_monte_carlo_to_run_specs

    perturb = {"Sat.SMA": ("normal", 7100.0, 50.0)}
    planned_specs = expand_monte_carlo_to_run_specs(
        perturb=perturb, n=8, seed=1729, script_path=script, output_dir=out
    )
    target_failure_value = planned_specs[3].overrides["Sat.SMA"]

    def _setitem(_key: str, value: Any) -> None:
        if value == target_failure_value:
            raise ValueError("rejected by GMAT")

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(
            script_path=Path(path),
            setitem_hook=_setitem,
            run_hook=_payload_run_hook(),
        )
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    monte_carlo(
        script,
        n=8,
        perturb=perturb,
        seed=1729,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    assert 3 in Manifest.find_failed(out / "manifest.jsonl")
    pre_resume = Manifest.load(out / "manifest.jsonl")
    pre_overrides = {e.run_id: e.overrides for e in pre_resume.entries}

    # Resume with no failure source.
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    with LocalJoblibPool(workers=1) as pool:
        Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False).resume()

    post_resume = Manifest.load(out / "manifest.jsonl")
    post_overrides = {e.run_id: e.overrides for e in post_resume.entries}

    assert post_overrides == pre_overrides
    assert all(e.status == "ok" for e in post_resume.entries)


# ---- script drift detection ----------------------------------------------


def test_script_drift_raises_sweep_config_error(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    # Mutate the script: same path, different bytes ⇒ different canonical hash.
    script.write_text("% mission v2\nCreate Spacecraft Sat;\n", encoding="utf-8")

    with (
        LocalJoblibPool(workers=1) as pool,
        pytest.raises(SweepConfigError, match="script hash mismatch"),
    ):
        Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False)


def test_allow_script_drift_warns_and_proceeds(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    script.write_text("% mission v2\nCreate Spacecraft Sat;\n", encoding="utf-8")

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())
    with (
        LocalJoblibPool(workers=1) as pool,
        pytest.warns(RuntimeWarning, match="script hash mismatch"),
    ):
        rebuilt = Sweep.from_manifest(
            out / "manifest.jsonl",
            script,
            backend=pool,
            allow_script_drift=True,
            progress=False,
        )
    assert len(rebuilt._runs) == 2


# ---- bit-equal end-to-end DataFrame after resume -------------------------


def test_resumed_dataframe_matches_unbroken_sweep_via_sma_echo(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A 4-run grid sweep run-then-resumed (with one mid-flight failure)
    produces the same per-run ``x=Sat.SMA`` values as an equivalent sweep
    that succeeded on the first pass."""
    script = _write_script(tmp_path)

    # Reference run with no failures.
    _install_sma_echoing_loader(fake_gmat_run)
    df_reference = sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=LocalJoblibPool(workers=1),
        out=tmp_path / "ref",
        progress=False,
    )

    # Failure-then-resume run.
    out = tmp_path / "fail-resume"

    def _setitem_fail_on_7100(_key: str, value: Any) -> None:
        if value == 7100.0:
            raise ValueError("rejected by GMAT")

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(script_path=Path(path), setitem_hook=_setitem_fail_on_7100)

        def _run(**_kwargs: Any) -> FakeResults:
            sma = next(v for k, v in mission.overrides_log if k == "Sat.SMA")
            return FakeResults(
                reports={
                    "R": pd.DataFrame(
                        {"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [float(sma)]}
                    )
                }
            )

        mission.run_hook = _run
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    _install_sma_echoing_loader(fake_gmat_run)
    with LocalJoblibPool(workers=1) as pool:
        df_resumed = (
            Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False)
            .resume()
            .to_dataframe()
        )

    pd.testing.assert_series_equal(df_resumed["x"].sort_index(), df_reference["x"].sort_index())
    assert (df_resumed["__status"] == "ok").all()
