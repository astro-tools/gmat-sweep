"""Structural tests for monte_carlo_extend / latin_hypercube_extend (issue #92).

End-to-end bit-equivalence assertions live in
``test_monte_carlo_determinism.py`` next to the rest of the determinism
contract; this file holds the validation, refusal, and Manifest-property
tests that don't need a running sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.api import latin_hypercube, latin_hypercube_extend, monte_carlo, monte_carlo_extend
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.grids import expand_monte_carlo_extension_to_run_specs
from gmat_sweep.manifest import Manifest
from tests.conftest import FakeGmatRun, FakeMission, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _install_payload_loader(fake_gmat_run: FakeGmatRun) -> None:
    """Loader whose runs always return a one-row constant report."""
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    fake_gmat_run.install_loader(run_hook=_run)


# ---- expand_monte_carlo_extension_to_run_specs ---------------------------


def test_extension_expander_emits_only_the_requested_tail(tmp_path: Path) -> None:
    specs = expand_monte_carlo_extension_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        old_n=100,
        n=20,
        seed=42,
        script_path=tmp_path / "mission.script",
        output_dir=tmp_path / "out",
    )
    assert [s.run_id for s in specs] == list(range(100, 120))
    # Output dirs follow the same run-<run_id> convention as the original.
    assert specs[0].output_dir.name == "run-100"
    assert specs[-1].output_dir.name == "run-119"


def test_extension_expander_overlap_is_bit_equal_to_original(tmp_path: Path) -> None:
    """Extending [old_n, old_n + n) draws the same per-parameter values as
    a fresh expand_monte_carlo_to_run_specs(total) call at the same indices.

    This is the property the high-level contract rests on, exposed at the
    expander layer so a regression in :func:`numpy.random.SeedSequence.spawn`
    or in our derivation surfaces here rather than at end-to-end DataFrame
    comparison."""
    from gmat_sweep.grids import expand_monte_carlo_to_run_specs

    perturb = {
        "Sat.SMA": ("normal", 7100.0, 50.0),
        "Sat.INC": ("uniform", 0.0, 90.0),
    }
    fresh = expand_monte_carlo_to_run_specs(
        perturb=perturb,
        n=300,
        seed=42,
        script_path=tmp_path / "mission.script",
        output_dir=tmp_path / "out",
    )
    extension = expand_monte_carlo_extension_to_run_specs(
        perturb=perturb,
        old_n=100,
        n=200,
        seed=42,
        script_path=tmp_path / "mission.script",
        output_dir=tmp_path / "out",
    )

    assert [s.overrides for s in extension] == [s.overrides for s in fresh[100:300]]
    assert [s.seed for s in extension] == [s.seed for s in fresh[100:300]]


def test_extension_expander_validates_inputs(tmp_path: Path) -> None:
    perturb = {"Sat.SMA": ("normal", 7100.0, 50.0)}
    script_path = tmp_path / "mission.script"
    output_dir = tmp_path / "out"
    with pytest.raises(SweepConfigError, match="non-empty perturb"):
        expand_monte_carlo_extension_to_run_specs(
            perturb={},
            old_n=10,
            n=5,
            seed=42,
            script_path=script_path,
            output_dir=output_dir,
        )
    with pytest.raises(SweepConfigError, match="old_n >= 0"):
        expand_monte_carlo_extension_to_run_specs(
            perturb=perturb,
            old_n=-1,
            n=5,
            seed=42,
            script_path=script_path,
            output_dir=output_dir,
        )
    with pytest.raises(SweepConfigError, match="n >= 1"):
        expand_monte_carlo_extension_to_run_specs(
            perturb=perturb,
            old_n=10,
            n=0,
            seed=42,
            script_path=script_path,
            output_dir=output_dir,
        )


# ---- Manifest.extension_run_count ----------------------------------------


def test_extension_run_count_is_zero_for_fresh_monte_carlo(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    monte_carlo(
        script,
        n=10,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.extension_run_count == 0


def test_extension_run_count_is_positive_after_extend(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    monte_carlo(
        script,
        n=10,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    monte_carlo_extend(
        out / "manifest.jsonl",
        script,
        n=15,
        backend=LocalJoblibPool(workers=1),
        progress=False,
    )
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.extension_run_count == 15
    # parameter_spec.n stays frozen at the original value — header is append-only.
    assert manifest.parameter_spec["n"] == 10


def test_extension_run_count_is_zero_for_grid_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The derivation only applies to MC sweeps; a grid manifest (which has
    no concept of cumulative extension) reports zero."""
    from gmat_sweep.api import sweep

    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.extension_run_count == 0


# ---- monte_carlo_extend refusal paths ------------------------------------


def test_extend_refuses_grid_manifest(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    from gmat_sweep.api import sweep

    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    with pytest.raises(SweepConfigError, match="only applies to Monte Carlo"):
        monte_carlo_extend(
            out / "manifest.jsonl",
            script,
            n=5,
            backend=LocalJoblibPool(workers=1),
            progress=False,
        )


def test_extend_refuses_when_base_has_failed_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A torn base sweep — even one failed run — must block extension and
    direct the user at .resume() first. The contiguity invariant on
    [0, old_n) keeps the manifest interpretable downstream."""
    from types import SimpleNamespace

    script = _write_script(tmp_path)
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})
    fail_run_ids = {2, 5}
    seen: list[int] = []

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(script_path=Path(path))
        my_id = len(seen)
        seen.append(my_id)

        def _run(**_kwargs: Any) -> FakeResults:
            if my_id in fail_run_ids:
                raise RuntimeError("synthetic failure")
            return FakeResults(reports={"R": payload})

        mission.run_hook = _run
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]
    out = tmp_path / "out"
    monte_carlo(
        script,
        n=10,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )

    with pytest.raises(SweepConfigError, match="incomplete base sweep"):
        monte_carlo_extend(
            out / "manifest.jsonl",
            script,
            n=5,
            backend=LocalJoblibPool(workers=1),
            progress=False,
        )


def test_extend_validates_n_positive(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    monte_carlo(
        script,
        n=4,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    with pytest.raises(SweepConfigError, match="n >= 1"):
        monte_carlo_extend(
            out / "manifest.jsonl",
            script,
            n=0,
            backend=LocalJoblibPool(workers=1),
            progress=False,
        )


def test_extend_refuses_on_script_drift(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """A drifted script triggers Sweep.from_manifest's hash check before any
    runs dispatch — same surface as resume."""
    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    monte_carlo(
        script,
        n=4,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    script.write_text("% drifted contents\n", encoding="utf-8")
    with pytest.raises(SweepConfigError, match="script hash mismatch"):
        monte_carlo_extend(
            out / "manifest.jsonl",
            script,
            n=2,
            backend=LocalJoblibPool(workers=1),
            progress=False,
        )


# ---- latin_hypercube_extend ----------------------------------------------


def test_latin_hypercube_extend_refuses_with_clear_message(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """latin_hypercube_extend raises immediately with a message that names
    'stratification' so callers grepping for the reason find it."""
    script = _write_script(tmp_path)
    _install_payload_loader(fake_gmat_run)
    out = tmp_path / "out"
    latin_hypercube(
        script,
        n=8,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(workers=1),
        out=out,
        progress=False,
    )
    with pytest.raises(SweepConfigError, match="stratification"):
        latin_hypercube_extend(
            out / "manifest.jsonl",
            script,
            n=4,
            backend=LocalJoblibPool(workers=1),
            progress=False,
        )
