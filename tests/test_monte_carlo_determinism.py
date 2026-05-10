"""End-to-end Monte Carlo determinism — v0.2 validation suite (issue #40).

Pins the determinism contract underpinning Monte Carlo replay (#33) and the
resume flow (#36): two ``monte_carlo()`` calls at the same
``(mission, n, perturb, seed)`` produce DataFrames that compare bit-equal —
not just RunSpecs and not just manifest overrides, but the assembled
``(run_id, time)``-MultiIndexed result the user sees. The per-parameter
sub-seed contract (#33) is also pinned at the DataFrame level: adding a
second perturbed parameter to an existing sweep does not change the first
parameter's draws at any ``run_id``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from gmat_sweep.api import monte_carlo, monte_carlo_extend
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.grids import expand_monte_carlo_to_run_specs
from tests.conftest import FakeGmatRun, FakeMission, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _install_sma_echoing_loader(fake_gmat_run: FakeGmatRun) -> None:
    """Install a loader where each run's report payload echoes its Sat.SMA override.

    Without this, every run returns the same constant FakeResults payload and
    the assembled DataFrame loses the seed-derived signal — two different
    seeds would still produce bit-equal DataFrames. Echoing the per-run
    Sat.SMA into the report's ``x`` column makes the DataFrame reflect the
    draw, so the determinism contract is asserted on user-visible state.
    """

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


# ---- bit-equal DataFrames at the same seed -------------------------------


def test_two_calls_same_seed_produce_bit_equal_dataframes(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)

    df_a = monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-a",
        progress=False,
    )
    df_b = monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-b",
        progress=False,
    )

    pd.testing.assert_frame_equal(df_a, df_b)


def test_different_seeds_produce_distinct_dataframes(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)

    df_42 = monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-42",
        progress=False,
    )
    df_43 = monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=43,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-43",
        progress=False,
    )

    assert not df_42["x"].equals(df_43["x"])


# ---- per-parameter sub-seed independence ---------------------------------


def test_adding_a_second_perturbed_parameter_preserves_first_dataframe_draws(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Adding ``Sat.INC`` to ``perturb`` must not perturb any ``Sat.SMA`` draw.

    The Sat.SMA echo in the ``x`` column lets us compare DataFrames directly:
    every run's ``x`` must be identical between the one-parameter and
    two-parameter sweep at the same seed. The sub-seed derivation keys on the
    parameter *name*, not its position, which is what makes this work.
    """
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)

    df_one = monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-one",
        progress=False,
    )
    df_two = monte_carlo(
        script,
        n=20,
        perturb={
            "Sat.SMA": ("normal", 7100.0, 50.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "out-two",
        progress=False,
    )

    pd.testing.assert_series_equal(df_one["x"], df_two["x"])


# ---- monte_carlo_extend bit-equivalence (issue #92) ----------------------


def test_extend_preserves_original_run_ids_bit_for_bit(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A 100-run MC followed by ``monte_carlo_extend(n=200)`` produces a
    300-run aggregated DataFrame whose first 100 ``run_id``\\ s match the
    original 100 bit-for-bit. Pinned at the user-visible DataFrame level
    via the SMA echo so the assertion catches any silent draw drift."""
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)
    out = tmp_path / "out"

    df_pre = monte_carlo(
        script,
        n=100,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
        progress=False,
    )

    df_post = monte_carlo_extend(
        out / "manifest.jsonl",
        script,
        n=200,
        backend=LocalJoblibPool(max_workers=1),
        progress=False,
    )

    assert df_post.index.get_level_values("run_id").nunique() == 300
    pd.testing.assert_frame_equal(
        df_post.loc[df_post.index.get_level_values("run_id") < 100],
        df_pre,
    )


def test_extend_matches_fresh_full_size_sweep_at_overlap(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A fresh 300-run ``monte_carlo(n=300)`` produces a DataFrame whose
    first 100 ``run_id``\\ s match the extended sweep's first 100
    bit-for-bit. The contract: extend(n=200) on top of a 100-run base is
    indistinguishable from monte_carlo(n=300) at every run_id < 300."""
    script = _write_script(tmp_path)
    _install_sma_echoing_loader(fake_gmat_run)

    base_out = tmp_path / "base"
    monte_carlo(
        script,
        n=100,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=base_out,
        progress=False,
    )
    df_extended = monte_carlo_extend(
        base_out / "manifest.jsonl",
        script,
        n=200,
        backend=LocalJoblibPool(max_workers=1),
        progress=False,
    )

    df_fresh = monte_carlo(
        script,
        n=300,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=tmp_path / "fresh",
        progress=False,
    )

    pd.testing.assert_frame_equal(df_extended, df_fresh)


# ---- cross-process determinism -------------------------------------------


def test_expand_monte_carlo_overrides_are_bit_equal_across_processes(
    tmp_path: Path,
) -> None:
    """The per-run override draws produced by ``expand_monte_carlo_to_run_specs``
    are identical in a fresh subprocess.

    ``test_distributions`` already pins this for ``derive_run_seeds`` and
    ``derive_param_seed`` individually; this lifts the assertion one layer up
    to the spec generator that combines them with ``sample()``. A regression
    that introduced any process-affected RNG state (e.g. an unseeded global
    ``np.random``) would silently break Monte Carlo replay between the
    original sweep and a resumed sweep run on a fresh interpreter.
    """
    script_path = tmp_path / "mission.script"
    script_path.write_text("% mission\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    in_process_specs = expand_monte_carlo_to_run_specs(
        perturb={
            "Sat.SMA": ("normal", 7100.0, 50.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        n=8,
        seed=42,
        script_path=script_path,
        output_dir=output_dir,
    )
    in_process = [s.overrides for s in in_process_specs]

    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from gmat_sweep.grids import expand_monte_carlo_to_run_specs\n"
        f"specs = expand_monte_carlo_to_run_specs(\n"
        f'    perturb={{"Sat.SMA": ("normal", 7100.0, 50.0),'
        f' "Sat.INC": ("uniform", 0.0, 90.0)}},\n'
        f"    n=8,\n"
        f"    seed=42,\n"
        f"    script_path=Path({str(script_path)!r}),\n"
        f"    output_dir=Path({str(output_dir)!r}),\n"
        f")\n"
        "print(json.dumps([s.overrides for s in specs]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    cross_process = json.loads(result.stdout.strip())

    assert cross_process == in_process
