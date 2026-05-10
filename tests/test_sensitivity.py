"""Tests for ``gmat_sweep.sensitivity`` — Sobol sample design + analyse pipeline.

The Ishigami contract test is the headline check: at the SALib-recommended
``n=1024`` Saltelli base size, ``sobol_analyze`` must reproduce the published
S1 / ST values for the Ishigami function within Monte Carlo noise. Everything
else pins user-facing knobs (sample shape, determinism, the str / callable
``metric`` forms) and the failure modes that should raise
:class:`SweepConfigError` rather than degrade silently.

The whole module is gated on the ``[sensitivity]`` extra via
``pytest.importorskip("SALib")`` — the default install path never imports
SALib, and CI installs the extra explicitly so these tests run on the
coverage-gate cell.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("SALib")

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.sensitivity import sobol_analyze, sobol_sample


def _multiindex(n: int) -> pd.MultiIndex:
    return cast(
        pd.MultiIndex,
        pd.MultiIndex.from_arrays(
            [np.arange(n), np.zeros(n, dtype=int)],
            names=("run_id", "time"),
        ),
    )


def _ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> Any:
    """f(x1, x2, x3) = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)."""
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_sobol_analyze_recovers_ishigami_indices_within_mc_noise() -> None:
    """At the SALib-recommended ``n=1024`` base sample size the Saltelli
    design lands ~8200 evaluations on the Ishigami function. The published
    Sobol indices for ``a=7, b=0.1`` are
    ``S1 ≈ [0.314, 0.442, 0.000]`` and ``ST ≈ [0.558, 0.442, 0.244]`` —
    a 0.05 absolute tolerance comfortably brackets typical MC noise at
    this sample size."""
    perturb = {
        "x1": ("uniform", -math.pi, math.pi),
        "x2": ("uniform", -math.pi, math.pi),
        "x3": ("uniform", -math.pi, math.pi),
    }
    n = 1024
    samples = sobol_sample(perturb, n=n, seed=42, calc_second_order=True)

    y = _ishigami(samples.to_numpy())
    df = pd.DataFrame({"y": y}, index=_multiindex(len(samples)))

    result = sobol_analyze(df, perturb, metric="y", seed=42, calc_second_order=True)

    s1 = result[result["kind"] == "S1"].set_index("param_a")["value"]
    st = result[result["kind"] == "ST"].set_index("param_a")["value"]

    assert s1["x1"] == pytest.approx(0.314, abs=0.05)
    assert s1["x2"] == pytest.approx(0.442, abs=0.05)
    assert s1["x3"] == pytest.approx(0.000, abs=0.05)
    assert st["x1"] == pytest.approx(0.558, abs=0.05)
    assert st["x2"] == pytest.approx(0.442, abs=0.05)
    assert st["x3"] == pytest.approx(0.244, abs=0.05)


def test_sobol_sample_bit_equal_across_two_calls() -> None:
    """``sobol_sample`` is deterministic in ``(perturb, n, seed,
    calc_second_order)`` — required so a sweep can be re-launched against
    a recorded design without re-saving the DataFrame."""
    perturb = {
        "Sat.SMA": ("normal", 7100.0, 50.0),
        "Sat.INC": ("uniform", 0.0, 90.0),
    }
    a = sobol_sample(perturb, n=64, seed=42)
    b = sobol_sample(perturb, n=64, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_sobol_sample_shape_with_calc_second_order_true() -> None:
    """Saltelli design row count is ``n * (2*D + 2)`` when
    ``calc_second_order=True`` — pinning this guards against a surprise
    SALib behaviour change in the next release."""
    perturb = {
        "a": ("uniform", 0.0, 1.0),
        "b": ("uniform", 0.0, 1.0),
        "c": ("uniform", 0.0, 1.0),
    }
    n = 8
    df = sobol_sample(perturb, n=n, seed=0, calc_second_order=True)
    assert len(df) == n * (2 * 3 + 2)
    assert list(df.columns) == ["a", "b", "c"]


def test_sobol_sample_shape_with_calc_second_order_false() -> None:
    """``calc_second_order=False`` collapses the design to ``n * (D + 2)``
    rows — half-cost when the user does not need pairwise indices."""
    perturb = {
        "a": ("uniform", 0.0, 1.0),
        "b": ("uniform", 0.0, 1.0),
        "c": ("uniform", 0.0, 1.0),
    }
    n = 8
    df = sobol_sample(perturb, n=n, seed=0, calc_second_order=False)
    assert len(df) == n * (3 + 2)


def test_sobol_sample_columns_sorted_lexicographically() -> None:
    """Columns come out lexicographically regardless of ``perturb`` insertion
    order — the same stability guarantee :func:`latin_hypercube_samples`
    ships with."""
    perturb = {
        "Sat.SMA": ("uniform", 0.0, 1.0),
        "Aaa.X": ("uniform", 0.0, 1.0),
        "Mmm.Y": ("uniform", 0.0, 1.0),
    }
    df = sobol_sample(perturb, n=4, seed=0)
    assert list(df.columns) == ["Aaa.X", "Mmm.Y", "Sat.SMA"]


def test_sobol_sample_lifts_unit_cube_via_distribution_ppf() -> None:
    """The unit-cube SALib design must be lifted into the configured
    marginal — a normal-distributed parameter should land outside the
    [0, 1] range typical of the raw Saltelli samples."""
    perturb = {"x": ("normal", 100.0, 10.0)}
    df = sobol_sample(perturb, n=64, seed=0)
    # The mean is 100 with sigma 10; samples should mostly fall in [70, 130].
    assert df["x"].between(70.0, 130.0).all()
    # And clearly not on [0, 1].
    assert not df["x"].between(0.0, 1.0).any()


def test_sobol_analyze_with_callable_metric_recovers_linear_variance_split() -> None:
    """For ``Y = x1 + 2 * x2`` the variance-ratio decomposition gives
    ``S1[x1] = 1/5`` and ``S1[x2] = 4/5``. Use the callable ``metric`` form
    to perform the per-run reduction explicitly; the 0.05 tolerance again
    brackets MC noise at moderate sample size."""
    perturb = {
        "x1": ("uniform", 0.0, 1.0),
        "x2": ("uniform", 0.0, 1.0),
    }
    samples = sobol_sample(perturb, n=512, seed=0)
    y = samples["x1"].to_numpy() + 2.0 * samples["x2"].to_numpy()
    df = pd.DataFrame({"y": y}, index=_multiindex(len(samples)))

    result = sobol_analyze(
        df,
        perturb,
        metric=lambda d: d.groupby(level="run_id")["y"].last(),
        seed=0,
    )

    s1 = result[result["kind"] == "S1"].set_index("param_a")["value"]
    assert s1["x1"] == pytest.approx(0.20, abs=0.05)
    assert s1["x2"] == pytest.approx(0.80, abs=0.05)
    # S2 row should also be populated (calc_second_order defaults to True).
    s2 = result[result["kind"] == "S2"]
    assert len(s2) == 1
    assert s2.iloc[0]["param_a"] == "x1"
    assert s2.iloc[0]["param_b"] == "x2"


def test_sobol_analyze_calc_second_order_false_omits_s2_rows() -> None:
    """``calc_second_order=False`` must drop the ``S2`` rows from the
    output frame entirely — matching the trimmed sample design."""
    perturb = {
        "x1": ("uniform", 0.0, 1.0),
        "x2": ("uniform", 0.0, 1.0),
    }
    samples = sobol_sample(perturb, n=64, seed=0, calc_second_order=False)
    y = samples.to_numpy().sum(axis=1)
    df = pd.DataFrame({"y": y}, index=_multiindex(len(samples)))

    result = sobol_analyze(df, perturb, metric="y", seed=0, calc_second_order=False)

    assert set(result["kind"].unique()) == {"S1", "ST"}


def test_sobol_sample_raises_on_empty_perturb() -> None:
    with pytest.raises(SweepConfigError, match="non-empty"):
        sobol_sample({}, n=4)


def test_sobol_sample_raises_on_n_lt_1() -> None:
    with pytest.raises(SweepConfigError, match="n >= 1"):
        sobol_sample({"x": ("uniform", 0.0, 1.0)}, n=0)


def test_sobol_analyze_raises_on_empty_perturb() -> None:
    df = pd.DataFrame({"y": [1.0]}, index=_multiindex(1))
    with pytest.raises(SweepConfigError, match="non-empty"):
        sobol_analyze(df, {}, metric="y")


def test_sobol_analyze_raises_on_failed_runs() -> None:
    """A sweep result with any ``__status != 'ok'`` row carries NaN-padded
    values that would corrupt SALib's analysis. Forces the user to filter
    explicitly rather than silently propagating noise."""
    perturb = {"x": ("uniform", 0.0, 1.0)}
    samples = sobol_sample(perturb, n=4, seed=0)
    n_rows = len(samples)
    statuses = ["ok"] * n_rows
    statuses[2] = "failed"
    df = pd.DataFrame(
        {
            "y": np.arange(n_rows, dtype=float),
            "__status": statuses,
        },
        index=_multiindex(n_rows),
    )
    with pytest.raises(SweepConfigError, match="non-ok"):
        sobol_analyze(df, perturb, metric="y")


def test_sobol_analyze_raises_on_unknown_metric_column() -> None:
    perturb = {"x": ("uniform", 0.0, 1.0)}
    samples = sobol_sample(perturb, n=4, seed=0)
    df = pd.DataFrame(
        {"y": np.arange(len(samples), dtype=float)},
        index=_multiindex(len(samples)),
    )
    with pytest.raises(SweepConfigError, match="not found"):
        sobol_analyze(df, perturb, metric="missing")


def test_sobol_analyze_raises_when_str_metric_df_lacks_run_id_index() -> None:
    """A flat-RangeIndex df can't satisfy the str metric's per-run ``last``
    reduction — surface that with a config error rather than letting
    SALib raise a confusing length mismatch."""
    perturb = {"x": ("uniform", 0.0, 1.0)}
    samples = sobol_sample(perturb, n=4, seed=0)
    df = pd.DataFrame({"y": np.arange(len(samples), dtype=float)})
    with pytest.raises(SweepConfigError, match="run_id"):
        sobol_analyze(df, perturb, metric="y")


def test_sobol_analyze_raises_when_callable_returns_non_series() -> None:
    perturb = {"x": ("uniform", 0.0, 1.0)}
    samples = sobol_sample(perturb, n=4, seed=0)
    df = pd.DataFrame(
        {"y": np.arange(len(samples), dtype=float)},
        index=_multiindex(len(samples)),
    )
    with pytest.raises(SweepConfigError, match=r"pandas\.Series"):
        sobol_analyze(
            df,
            perturb,
            metric=lambda _d: np.zeros(len(samples)),  # type: ignore[arg-type,return-value]
        )


def test_sobol_analyze_raises_when_callable_returns_nan() -> None:
    perturb = {"x": ("uniform", 0.0, 1.0)}
    samples = sobol_sample(perturb, n=4, seed=0)
    df = pd.DataFrame(
        {"y": np.arange(len(samples), dtype=float)},
        index=_multiindex(len(samples)),
    )

    def bad(_d: pd.DataFrame) -> pd.Series:
        return pd.Series([np.nan] * len(samples))

    with pytest.raises(SweepConfigError, match="NaN"):
        sobol_analyze(df, perturb, metric=bad)


def test_sobol_sample_and_analyze_round_trip_via_top_level_imports() -> None:
    """``sobol_sample`` and ``sobol_analyze`` must be importable from the
    top-level ``gmat_sweep`` namespace — pins the public surface."""
    import gmat_sweep

    assert gmat_sweep.sobol_sample is sobol_sample
    assert gmat_sweep.sobol_analyze is sobol_analyze
