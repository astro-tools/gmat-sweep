"""Tests for gmat_sweep.grids — full-factorial and explicit-row run-spec expansion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.grids import (
    expand_grid_to_run_specs,
    expand_latin_hypercube_to_run_specs,
    expand_monte_carlo_to_run_specs,
    expand_samples_to_run_specs,
    full_factorial,
    latin_hypercube_samples,
)
from gmat_sweep.spec import RunSpec

# ---- full_factorial -------------------------------------------------------


def test_acceptance_example_six_specs_in_documented_order() -> None:
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    expected = [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 1, "b": 30},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
        {"a": 2, "b": 30},
    ]
    assert list(full_factorial(grid)) == expected


def test_keys_emit_in_lexicographic_order_regardless_of_input_order() -> None:
    # Insertion order is reversed; output should still be a-then-b.
    grid = {"b": [10, 20], "a": [1, 2]}
    out = list(full_factorial(grid))
    for d in out:
        assert list(d.keys()) == ["a", "b"]
    # And the lex-first key ("a") varies slowest.
    assert [d["a"] for d in out] == [1, 1, 2, 2]
    assert [d["b"] for d in out] == [10, 20, 10, 20]


def test_single_key_grid() -> None:
    assert list(full_factorial({"x": [1, 2, 3]})) == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_three_keys_lexicographic_and_product_order() -> None:
    grid: dict[str, list[Any]] = {"c": ["x"], "a": [1, 2], "b": [10, 20]}
    out = list(full_factorial(grid))
    assert out == [
        {"a": 1, "b": 10, "c": "x"},
        {"a": 1, "b": 20, "c": "x"},
        {"a": 2, "b": 10, "c": "x"},
        {"a": 2, "b": 20, "c": "x"},
    ]


def test_empty_mapping_yields_one_empty_dict() -> None:
    assert list(full_factorial({})) == [{}]


def test_generator_input_is_materialised_and_not_exhausted() -> None:
    def values() -> Any:
        yield 1
        yield 2

    grid = {"a": values(), "b": [10, 20]}
    # Iterating twice on the *result* is fine — generators on the input were
    # materialised at entry, not held by reference.
    first = list(full_factorial(grid))
    assert first == [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
    ]


def test_non_string_key_raises_sweep_config_error() -> None:
    with pytest.raises(SweepConfigError, match="grid keys must be strings"):
        list(full_factorial({1: [1, 2]}))  # type: ignore[dict-item]


def test_empty_iterable_value_raises_sweep_config_error() -> None:
    with pytest.raises(SweepConfigError, match="grid value for 'a' is empty"):
        list(full_factorial({"a": []}))


def test_empty_generator_value_raises_sweep_config_error() -> None:
    def empty() -> Any:
        return
        yield  # pragma: no cover - unreachable, marks the function as a generator

    with pytest.raises(SweepConfigError, match="grid value for 'a' is empty"):
        list(full_factorial({"a": empty()}))


def test_validation_runs_before_any_combination_is_yielded() -> None:
    # Even though "a" is well-formed and would normally produce 2 dicts, the
    # bad "b" entry should abort the whole call before anything is emitted.
    it = full_factorial({"a": [1, 2], "b": []})
    with pytest.raises(SweepConfigError):
        next(it)


def test_output_is_byte_for_byte_deterministic_across_calls() -> None:
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    a = json.dumps(list(full_factorial(grid)), sort_keys=True)
    b = json.dumps(list(full_factorial(grid)), sort_keys=True)
    assert a == b


# ---- expand_grid_to_run_specs --------------------------------------------


def test_expand_produces_sequential_run_ids_and_full_factorial_order() -> None:
    specs = expand_grid_to_run_specs(
        grid={"a": [1, 2], "b": [10, 20, 30]},
        script_path="/missions/flyby.script",
        output_dir="/sweep-out",
    )
    assert tuple(s.run_id for s in specs) == (0, 1, 2, 3, 4, 5)
    assert [s.overrides for s in specs] == [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 1, "b": 30},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
        {"a": 2, "b": 30},
    ]


def test_expand_packs_script_path_output_dir_seed_and_run_options() -> None:
    specs = expand_grid_to_run_specs(
        grid={"x": [7, 8]},
        script_path=Path("/missions/m.script"),
        output_dir=Path("/sweep-out"),
    )
    assert len(specs) == 2
    for spec, expected_id in zip(specs, (0, 1), strict=True):
        assert isinstance(spec, RunSpec)
        assert spec.script_path == Path("/missions/m.script")
        assert spec.output_dir == Path(f"/sweep-out/run-{expected_id}")
        assert spec.seed is None
        assert spec.run_options == {}


def test_expand_accepts_string_script_path_and_output_dir() -> None:
    specs = expand_grid_to_run_specs(
        grid={"x": [1]},
        script_path="/missions/m.script",
        output_dir="/out",
    )
    assert specs[0].script_path == Path("/missions/m.script")
    assert specs[0].output_dir == Path("/out/run-0")


def test_expand_empty_grid_yields_one_spec() -> None:
    specs = expand_grid_to_run_specs(
        grid={},
        script_path="/m.script",
        output_dir="/o",
    )
    assert len(specs) == 1
    assert specs[0].overrides == {}
    assert specs[0].run_id == 0
    assert specs[0].output_dir == Path("/o/run-0")


def test_expand_propagates_validation_errors() -> None:
    with pytest.raises(SweepConfigError):
        expand_grid_to_run_specs(grid={"a": []}, script_path="/m.script", output_dir="/o")
    with pytest.raises(SweepConfigError):
        expand_grid_to_run_specs(
            grid={1: [1]},  # type: ignore[dict-item]
            script_path="/m.script",
            output_dir="/o",
        )


def test_expand_output_round_trips_through_runspec_to_dict() -> None:
    specs = expand_grid_to_run_specs(
        grid={"a": [1, 2]},
        script_path="/m.script",
        output_dir="/o",
    )
    serialised = json.dumps([s.to_dict() for s in specs], sort_keys=True)
    restored = [RunSpec.from_dict(d) for d in json.loads(serialised)]
    assert restored == specs


# ---- expand_samples_to_run_specs -----------------------------------------


def test_expand_samples_acceptance_four_row_dataframe() -> None:
    """The issue's headline acceptance: a 4-row DataFrame yields 4 specs with
    sequential run_ids and the per-row override applied."""
    samples = pd.DataFrame({"Sat.SMA": [7000, 7100, 7200, 7300]})
    specs = expand_samples_to_run_specs(
        samples,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.run_id for s in specs] == [0, 1, 2, 3]
    assert [s.overrides for s in specs] == [
        {"Sat.SMA": 7000},
        {"Sat.SMA": 7100},
        {"Sat.SMA": 7200},
        {"Sat.SMA": 7300},
    ]


def test_expand_samples_packs_script_path_output_dir_seed_and_run_options() -> None:
    samples = pd.DataFrame({"x": [1, 2]})
    specs = expand_samples_to_run_specs(
        samples,
        script_path=Path("/missions/m.script"),
        output_dir=Path("/sweep-out"),
    )
    assert len(specs) == 2
    for spec, expected_id in zip(specs, (0, 1), strict=True):
        assert isinstance(spec, RunSpec)
        assert spec.script_path == Path("/missions/m.script")
        assert spec.output_dir == Path(f"/sweep-out/run-{expected_id}")
        assert spec.seed is None
        assert spec.run_options == {}


def test_expand_samples_accepts_string_paths() -> None:
    samples = pd.DataFrame({"x": [1]})
    specs = expand_samples_to_run_specs(
        samples,
        script_path="/missions/m.script",
        output_dir="/out",
    )
    assert specs[0].script_path == Path("/missions/m.script")
    assert specs[0].output_dir == Path("/out/run-0")


def test_expand_samples_multiple_columns_preserves_row_overrides() -> None:
    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, 7100.0],
            "Sat.ECC": [0.001, 0.002],
        }
    )
    specs = expand_samples_to_run_specs(samples, "/m.script", "/o")
    assert specs[0].overrides == {"Sat.SMA": 7000.0, "Sat.ECC": 0.001}
    assert specs[1].overrides == {"Sat.SMA": 7100.0, "Sat.ECC": 0.002}


def test_expand_samples_empty_dataframe_yields_zero_specs() -> None:
    """Empty DataFrame is degenerate but valid: 0 rows → 0 specs. The
    all-NaN check is skipped because ``.isna().all()`` is vacuously true
    on a zero-length column."""
    samples = pd.DataFrame({"x": pd.Series([], dtype=float)})
    specs = expand_samples_to_run_specs(samples, "/m.script", "/o")
    assert specs == []


def test_expand_samples_rejects_non_dataframe() -> None:
    with pytest.raises(SweepConfigError, match=r"must be a pandas\.DataFrame"):
        expand_samples_to_run_specs(
            {"x": [1, 2]},  # type: ignore[arg-type]
            "/m.script",
            "/o",
        )


def test_expand_samples_rejects_non_string_columns() -> None:
    samples = pd.DataFrame([[1, 2]], columns=[0, 1])
    with pytest.raises(SweepConfigError, match="column names must be strings"):
        expand_samples_to_run_specs(samples, "/m.script", "/o")


def test_expand_samples_rejects_duplicate_columns() -> None:
    samples = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(SweepConfigError, match="duplicate column names"):
        expand_samples_to_run_specs(samples, "/m.script", "/o")


def test_expand_samples_rejects_non_default_index() -> None:
    samples = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.RangeIndex(10, 14))
    with pytest.raises(SweepConfigError, match="default RangeIndex"):
        expand_samples_to_run_specs(samples, "/m.script", "/o")


def test_expand_samples_rejects_string_index() -> None:
    samples = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    with pytest.raises(SweepConfigError, match="default RangeIndex"):
        expand_samples_to_run_specs(samples, "/m.script", "/o")


def test_expand_samples_rejects_all_nan_column() -> None:
    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, 7100.0],
            "Sat.Dead": [float("nan"), float("nan")],
        }
    )
    with pytest.raises(SweepConfigError, match="all-NaN columns"):
        expand_samples_to_run_specs(samples, "/m.script", "/o")


def test_expand_samples_per_cell_nan_is_forwarded() -> None:
    """Per-cell NaN passes through unchanged — gmat-run is the line that
    decides whether NaN is a valid value for a given dotted path. The
    expander stays out of the way."""
    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, float("nan"), 7200.0],
        }
    )
    specs = expand_samples_to_run_specs(samples, "/m.script", "/o")
    assert specs[0].overrides == {"Sat.SMA": 7000.0}
    assert math.isnan(specs[1].overrides["Sat.SMA"])
    assert specs[2].overrides == {"Sat.SMA": 7200.0}


def test_expand_samples_runspec_round_trips_through_to_dict() -> None:
    samples = pd.DataFrame({"a": [1, 2], "b": [10.0, 20.0]})
    specs = expand_samples_to_run_specs(samples, "/m.script", "/o")
    serialised = json.dumps([s.to_dict() for s in specs], sort_keys=True)
    restored = [RunSpec.from_dict(d) for d in json.loads(serialised)]
    assert restored == specs


# ---- expand_monte_carlo_to_run_specs --------------------------------------


def test_expand_monte_carlo_returns_n_specs_with_per_run_seeds() -> None:
    specs = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=10,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.run_id for s in specs] == list(range(10))
    assert all(isinstance(s.seed, int) for s in specs)
    # Per-run seeds are distinct (the `derive_run_seeds` contract).
    assert len({s.seed for s in specs}) == 10
    assert all(set(s.overrides.keys()) == {"Sat.SMA"} for s in specs)


def test_expand_monte_carlo_deterministic_per_seed() -> None:
    a = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=20,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    b = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=20,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.overrides for s in a] == [s.overrides for s in b]
    assert [s.seed for s in a] == [s.seed for s in b]


def test_expand_monte_carlo_different_seed_different_draws() -> None:
    a = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=20,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    b = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=20,
        seed=43,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.overrides for s in a] != [s.overrides for s in b]


def test_expand_monte_carlo_per_param_seed_is_name_stable() -> None:
    """Adding a perturbed parameter must not change the draws of any other
    parameter at any run_id, regardless of where the new parameter falls
    in lexicographic order. This is the headline order-independence
    contract from issue #33."""
    one = expand_monte_carlo_to_run_specs(
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        n=20,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    # New param "Aaa.X" sorts BEFORE Sat.SMA — positional spawning would
    # have shifted Sat.SMA's draws. The name-derived sub-seed leaves them
    # intact.
    two = expand_monte_carlo_to_run_specs(
        perturb={
            "Aaa.X": ("uniform", 0.0, 1.0),
            "Sat.SMA": ("normal", 7100.0, 50.0),
        },
        n=20,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    sma_one = [s.overrides["Sat.SMA"] for s in one]
    sma_two = [s.overrides["Sat.SMA"] for s in two]
    assert sma_one == sma_two


def test_expand_monte_carlo_accepts_pre_frozen_rv() -> None:
    from scipy import stats

    specs = expand_monte_carlo_to_run_specs(
        perturb={"x": stats.beta(2, 5)},
        n=5,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    assert len(specs) == 5
    # beta(2, 5) is bounded on [0, 1].
    for s in specs:
        v = s.overrides["x"]
        assert 0.0 <= v <= 1.0


def test_expand_monte_carlo_rejects_empty_perturb() -> None:
    with pytest.raises(SweepConfigError, match="non-empty perturb"):
        expand_monte_carlo_to_run_specs(
            perturb={}, n=5, seed=42, script_path="/m.script", output_dir="/o"
        )


def test_expand_monte_carlo_rejects_n_less_than_one() -> None:
    with pytest.raises(SweepConfigError, match="requires n >= 1"):
        expand_monte_carlo_to_run_specs(
            perturb={"x": ("normal", 0, 1)},
            n=0,
            seed=42,
            script_path="/m.script",
            output_dir="/o",
        )


def test_expand_monte_carlo_propagates_distribution_validation() -> None:
    with pytest.raises(SweepConfigError, match="sigma must be > 0"):
        expand_monte_carlo_to_run_specs(
            perturb={"x": ("normal", 0.0, -1.0)},
            n=5,
            seed=42,
            script_path="/m.script",
            output_dir="/o",
        )


# ---- latin_hypercube_samples ----------------------------------------------


def test_latin_hypercube_samples_returns_n_rows_and_lex_sorted_columns() -> None:
    samples = latin_hypercube_samples(
        perturb={
            "Sat.SMA": ("uniform", 7000.0, 7400.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        n=64,
        seed=42,
    )
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 64
    assert list(samples.columns) == ["Sat.INC", "Sat.SMA"]


def test_latin_hypercube_samples_deterministic_per_seed() -> None:
    a = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 1.0)}, n=64, seed=42)
    b = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 1.0)}, n=64, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_latin_hypercube_samples_different_seed_different_draws() -> None:
    a = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 1.0)}, n=64, seed=42)
    b = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 1.0)}, n=64, seed=43)
    assert not a.equals(b)


def test_latin_hypercube_samples_stratification_one_per_n_tile() -> None:
    """The stratification guarantee: after sorting, sample i (out of n) lies
    in the unit-cube stratum [i/n, (i+1)/n). For a uniform(0, 1)
    distribution the cdf is the identity, so the sample values themselves
    must obey the stratum bounds."""
    n = 1000
    samples = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 1.0)}, n=n, seed=42)
    sorted_vals = samples["x"].sort_values().to_numpy()
    for i, v in enumerate(sorted_vals):
        assert i / n <= v < (i + 1) / n, f"sample {i}={v} outside stratum [{i / n}, {(i + 1) / n})"


def test_latin_hypercube_samples_ppf_transform_respects_user_distribution() -> None:
    """Mapping uniform[0, 5] should give samples in [0, 5] exactly — the
    LH-on-the-unit-cube strata composed with `uniform(loc=0, scale=5).ppf`
    keep the sample range pinned to the requested distribution support."""
    samples = latin_hypercube_samples(perturb={"x": ("uniform", 0.0, 5.0)}, n=200, seed=42)
    assert samples["x"].min() >= 0.0
    assert samples["x"].max() <= 5.0


def test_latin_hypercube_samples_rejects_empty_perturb() -> None:
    with pytest.raises(SweepConfigError, match="non-empty perturb"):
        latin_hypercube_samples(perturb={}, n=10, seed=42)


def test_latin_hypercube_samples_rejects_n_less_than_one() -> None:
    with pytest.raises(SweepConfigError, match="requires n >= 1"):
        latin_hypercube_samples(perturb={"x": ("uniform", 0, 1)}, n=0, seed=42)


def test_latin_hypercube_samples_propagates_distribution_validation() -> None:
    with pytest.raises(SweepConfigError, match="requires hi > lo"):
        latin_hypercube_samples(perturb={"x": ("uniform", 5.0, 1.0)}, n=10, seed=42)


# ---- expand_latin_hypercube_to_run_specs ---------------------------------


def test_expand_latin_hypercube_returns_n_specs_in_lex_column_order() -> None:
    specs = expand_latin_hypercube_to_run_specs(
        perturb={
            "Sat.SMA": ("uniform", 7000.0, 7400.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        n=8,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.run_id for s in specs] == list(range(8))
    # Lex-sorted columns: Sat.INC before Sat.SMA.
    for s in specs:
        assert list(s.overrides.keys()) == ["Sat.INC", "Sat.SMA"]


def test_expand_latin_hypercube_deterministic_per_seed() -> None:
    a = expand_latin_hypercube_to_run_specs(
        perturb={"x": ("uniform", 0.0, 1.0)},
        n=16,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    b = expand_latin_hypercube_to_run_specs(
        perturb={"x": ("uniform", 0.0, 1.0)},
        n=16,
        seed=42,
        script_path="/m.script",
        output_dir="/o",
    )
    assert [s.overrides for s in a] == [s.overrides for s in b]
