"""Tests for gmat_sweep.distributions — distribution coercion and seeded sampling."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from gmat_sweep.distributions import derive_run_seeds, sample, to_rv_frozen
from gmat_sweep.errors import SweepConfigError

# ---- to_rv_frozen: shorthand mapping --------------------------------------


def test_normal_shorthand_maps_to_scipy_norm() -> None:
    rv = to_rv_frozen(("normal", 3.0, 2.0))
    assert isinstance(rv, rv_frozen)
    # scipy.stats.norm uses (loc, scale) in args; freeze stores them as kwds.
    assert rv.kwds == {"loc": 3.0, "scale": 2.0}
    assert rv.dist.name == "norm"


def test_uniform_shorthand_maps_to_scipy_uniform_with_loc_and_scale() -> None:
    rv = to_rv_frozen(("uniform", 5.0, 11.0))
    assert isinstance(rv, rv_frozen)
    assert rv.kwds == {"loc": 5.0, "scale": 6.0}
    assert rv.dist.name == "uniform"


def test_lognormal_shorthand_maps_to_scipy_lognorm_with_s_and_scale() -> None:
    rv = to_rv_frozen(("lognormal", 1.0, 0.5))
    assert isinstance(rv, rv_frozen)
    assert rv.kwds == {"s": 0.5, "scale": pytest.approx(math.exp(1.0))}
    assert rv.dist.name == "lognorm"


def test_pre_frozen_rv_passes_through_unchanged() -> None:
    pre = stats.beta(2, 5)
    out = to_rv_frozen(pre)
    assert out is pre


def test_shorthand_accepts_int_parameters_and_coerces_to_float() -> None:
    rv = to_rv_frozen(("normal", 0, 1))
    assert rv.kwds == {"loc": 0.0, "scale": 1.0}


# ---- to_rv_frozen: round-trip moments within 5% ---------------------------


def _moments_within_5pct(rv: rv_frozen, expected_mean: float, expected_std: float) -> None:
    samples = rv.rvs(size=10_000, random_state=np.random.default_rng(20260504))
    sample_mean = float(np.mean(samples))
    sample_std = float(np.std(samples))
    # 5% of the magnitude of the expected value, falling back to 5% of std when
    # the expected mean is zero.
    mean_tol = 0.05 * (abs(expected_mean) if expected_mean != 0 else expected_std)
    std_tol = 0.05 * abs(expected_std)
    assert abs(sample_mean - expected_mean) <= mean_tol, (
        f"mean {sample_mean} not within 5% of {expected_mean}"
    )
    assert abs(sample_std - expected_std) <= std_tol, (
        f"std {sample_std} not within 5% of {expected_std}"
    )


def test_normal_round_trip_moments_within_5pct() -> None:
    _moments_within_5pct(to_rv_frozen(("normal", 3.0, 2.0)), expected_mean=3.0, expected_std=2.0)


def test_uniform_round_trip_moments_within_5pct() -> None:
    lo, hi = 5.0, 11.0
    expected_mean = (lo + hi) / 2
    expected_std = (hi - lo) / math.sqrt(12)
    _moments_within_5pct(to_rv_frozen(("uniform", lo, hi)), expected_mean, expected_std)


def test_lognormal_round_trip_moments_within_5pct() -> None:
    mu, sigma = 1.0, 0.5
    expected_mean = math.exp(mu + sigma**2 / 2)
    expected_std = math.sqrt((math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2))
    _moments_within_5pct(to_rv_frozen(("lognormal", mu, sigma)), expected_mean, expected_std)


# ---- to_rv_frozen: validation errors --------------------------------------


def test_unknown_shorthand_tag_raises() -> None:
    with pytest.raises(SweepConfigError, match="unknown distribution shorthand tag"):
        to_rv_frozen(("triangular", 0, 1))


def test_non_tuple_non_rv_raises_for_string() -> None:
    with pytest.raises(SweepConfigError, match="must be a shorthand tuple or scipy rv_frozen"):
        to_rv_frozen("normal")


def test_non_tuple_non_rv_raises_for_int() -> None:
    with pytest.raises(SweepConfigError, match="must be a shorthand tuple or scipy rv_frozen"):
        to_rv_frozen(42)


def test_empty_tuple_raises() -> None:
    with pytest.raises(SweepConfigError, match="distribution spec tuple is empty"):
        to_rv_frozen(())


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_wrong_tuple_length_too_short_raises(tag: str) -> None:
    with pytest.raises(
        SweepConfigError, match=f"{tag!r} distribution spec must be a length-3 tuple"
    ):
        to_rv_frozen((tag, 1.0))


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_wrong_tuple_length_too_long_raises(tag: str) -> None:
    with pytest.raises(
        SweepConfigError, match=f"{tag!r} distribution spec must be a length-3 tuple"
    ):
        to_rv_frozen((tag, 1.0, 2.0, 3.0))


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_non_numeric_parameter_raises(tag: str) -> None:
    with pytest.raises(SweepConfigError, match="parameters must be numeric"):
        to_rv_frozen((tag, "oops", 1.0))


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_bool_parameter_rejected_as_non_numeric(tag: str) -> None:
    # bool is an int subclass; reject explicitly so callers don't accidentally
    # turn `("normal", True, 1.0)` into a degenerate normal at 1.0.
    with pytest.raises(SweepConfigError, match="parameters must be numeric"):
        to_rv_frozen((tag, True, 1.0))


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_non_finite_inf_parameter_raises(tag: str) -> None:
    with pytest.raises(SweepConfigError, match="parameters must be finite"):
        to_rv_frozen((tag, math.inf, 1.0))


@pytest.mark.parametrize("tag", ["normal", "uniform", "lognormal"])
def test_non_finite_nan_parameter_raises(tag: str) -> None:
    with pytest.raises(SweepConfigError, match="parameters must be finite"):
        to_rv_frozen((tag, 0.0, math.nan))


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_normal_non_positive_sigma_raises(sigma: float) -> None:
    with pytest.raises(SweepConfigError, match="'normal' distribution sigma must be > 0"):
        to_rv_frozen(("normal", 0.0, sigma))


@pytest.mark.parametrize("sigma", [0.0, -0.5])
def test_lognormal_non_positive_sigma_raises(sigma: float) -> None:
    with pytest.raises(SweepConfigError, match="'lognormal' distribution sigma must be > 0"):
        to_rv_frozen(("lognormal", 0.0, sigma))


@pytest.mark.parametrize(("lo", "hi"), [(1.0, 1.0), (5.0, 1.0)])
def test_uniform_degenerate_range_raises(lo: float, hi: float) -> None:
    with pytest.raises(SweepConfigError, match="requires hi > lo"):
        to_rv_frozen(("uniform", lo, hi))


# ---- derive_run_seeds -----------------------------------------------------


def test_derive_run_seeds_returns_n_distinct_ints_for_fixed_parent() -> None:
    seeds = derive_run_seeds(42, 1000)
    assert len(seeds) == 1000
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 1000


def test_derive_run_seeds_in_process_reproducible() -> None:
    a = derive_run_seeds(42, 1000)
    b = derive_run_seeds(42, 1000)
    assert a == b


def test_derive_run_seeds_cross_process_reproducible() -> None:
    code = (
        "from gmat_sweep.distributions import derive_run_seeds; "
        "import json; "
        "print(json.dumps(derive_run_seeds(42, 1000)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    cross_proc = json.loads(result.stdout.strip())
    assert cross_proc == derive_run_seeds(42, 1000)


def test_derive_run_seeds_different_parents_differ() -> None:
    assert derive_run_seeds(42, 8) != derive_run_seeds(43, 8)


def test_derive_run_seeds_zero_returns_empty_list_for_int_parent() -> None:
    assert derive_run_seeds(42, 0) == []


def test_derive_run_seeds_zero_returns_empty_list_for_none_parent() -> None:
    assert derive_run_seeds(None, 0) == []


def test_derive_run_seeds_negative_n_raises() -> None:
    with pytest.raises(SweepConfigError, match="requires n >= 0"):
        derive_run_seeds(42, -1)


def test_derive_run_seeds_none_parent_returns_correct_shape() -> None:
    # OS-entropy parent: only assert shape and types. Two calls almost surely
    # differ but we don't assert that — would be flaky.
    seeds = derive_run_seeds(None, 5)
    assert len(seeds) == 5
    assert all(isinstance(s, int) for s in seeds)


# ---- sample ---------------------------------------------------------------


def test_sample_reproducible_per_spec_seed_pair() -> None:
    a = sample(("normal", 0.0, 1.0), seed=42)
    b = sample(("normal", 0.0, 1.0), seed=42)
    assert a == b


def test_sample_differs_across_seeds() -> None:
    a = sample(("normal", 0.0, 1.0), seed=42)
    b = sample(("normal", 0.0, 1.0), seed=43)
    assert a != b


@pytest.mark.parametrize(
    "spec",
    [
        ("normal", 0.0, 1.0),
        ("uniform", -1.0, 1.0),
        ("lognormal", 0.0, 0.25),
    ],
)
def test_sample_returns_python_float_for_each_shorthand(spec: tuple[str, float, float]) -> None:
    out = sample(spec, seed=7)
    assert isinstance(out, float)


def test_sample_works_with_pre_frozen_rv() -> None:
    pre = stats.beta(2, 5)
    a = sample(pre, seed=99)
    b = sample(pre, seed=99)
    assert a == b
    assert isinstance(a, float)
