"""Tests for gmat_sweep.grids — full-factorial cartesian-product expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.grids import expand_grid_to_run_specs, full_factorial
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
        mission="/missions/flyby.script",
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
        mission=Path("/missions/m.script"),
        output_dir=Path("/sweep-out"),
    )
    assert len(specs) == 2
    for spec, expected_id in zip(specs, (0, 1), strict=True):
        assert isinstance(spec, RunSpec)
        assert spec.script_path == Path("/missions/m.script")
        assert spec.output_dir == Path(f"/sweep-out/run-{expected_id}")
        assert spec.seed is None
        assert spec.run_options == {}


def test_expand_accepts_string_mission_and_output_dir() -> None:
    specs = expand_grid_to_run_specs(
        grid={"x": [1]},
        mission="/missions/m.script",
        output_dir="/out",
    )
    assert specs[0].script_path == Path("/missions/m.script")
    assert specs[0].output_dir == Path("/out/run-0")


def test_expand_empty_grid_yields_one_spec() -> None:
    specs = expand_grid_to_run_specs(
        grid={},
        mission="/m.script",
        output_dir="/o",
    )
    assert len(specs) == 1
    assert specs[0].overrides == {}
    assert specs[0].run_id == 0
    assert specs[0].output_dir == Path("/o/run-0")


def test_expand_propagates_validation_errors() -> None:
    with pytest.raises(SweepConfigError):
        expand_grid_to_run_specs(grid={"a": []}, mission="/m.script", output_dir="/o")
    with pytest.raises(SweepConfigError):
        expand_grid_to_run_specs(
            grid={1: [1]},  # type: ignore[dict-item]
            mission="/m.script",
            output_dir="/o",
        )


def test_expand_output_round_trips_through_runspec_to_dict() -> None:
    specs = expand_grid_to_run_specs(
        grid={"a": [1, 2]},
        mission="/m.script",
        output_dir="/o",
    )
    serialised = json.dumps([s.to_dict() for s in specs], sort_keys=True)
    restored = [RunSpec.from_dict(d) for d in json.loads(serialised)]
    assert restored == specs
