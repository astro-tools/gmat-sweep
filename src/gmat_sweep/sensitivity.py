"""Sobol sensitivity indices via SALib (extra ``[sensitivity]``).

Wraps :mod:`SALib.sample.sobol` and :mod:`SALib.analyze.sobol` so a
gmat-sweep user can produce a Saltelli/Sobol sample design as an
explicit-row DataFrame, hand it to :func:`gmat_sweep.sweep` via
``samples=``, and run :func:`sobol_analyze` on the result without
assembling SALib's ``problem`` dict by hand.

SALib is imported lazily inside each function — a default
``pip install gmat-sweep`` never touches SALib. Install the optional
extra to enable the module::

    pip install gmat-sweep[sensitivity]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from gmat_sweep.distributions import DistSpec, to_rv_frozen
from gmat_sweep.errors import SweepConfigError

__all__ = [
    "sobol_analyze",
    "sobol_sample",
]


def sobol_sample(
    perturb: Mapping[str, DistSpec],
    n: int,
    *,
    seed: int | None = None,
    calc_second_order: bool = True,
) -> pd.DataFrame:
    """Build a Saltelli/Sobol sample design as an explicit-row DataFrame.

    The returned DataFrame has parameter columns in lexicographic order and
    a default :class:`pandas.RangeIndex` — suitable input to
    :func:`gmat_sweep.sweep` via ``samples=``. Row count is ``n * (2*D + 2)``
    when ``calc_second_order=True`` (the default) and ``n * (D + 2)``
    otherwise, where ``D = len(perturb)``.

    The unit-cube design from :mod:`SALib.sample.sobol` is lifted into each
    parameter's marginal via ``to_rv_frozen(perturb[k]).ppf(...)``, so any
    distribution shape :data:`gmat_sweep.distributions.DistSpec` accepts is
    supported — not just the uniform/normal/lognormal cases SALib's own
    ``dists`` knob covers.

    Two calls at the same ``(perturb, n, seed, calc_second_order)`` produce
    bit-equal DataFrames. ``seed=None`` falls back to OS entropy and is
    **not** reproducible.

    Parameters
    ----------
    perturb:
        Mapping from dotted-path field name to a distribution spec. Same
        surface as :func:`gmat_sweep.monte_carlo`.
    n:
        Saltelli base sample size. Must be ``>= 1``. Total runs are
        ``n * (2*D + 2)`` (or ``n * (D + 2)``); SALib's authors recommend
        powers of two.
    seed:
        Optional integer seed forwarded to SALib's sampler.
    calc_second_order:
        Whether to expand the design for second-order indices. Match this
        flag in :func:`sobol_analyze`.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, ``n < 1``, or any parameter spec fails
        validation in :func:`gmat_sweep.distributions.to_rv_frozen`.
    """
    if not perturb:
        raise SweepConfigError("sobol_sample requires a non-empty perturb mapping")
    if n < 1:
        raise SweepConfigError(f"sobol_sample requires n >= 1, got {n}")

    from SALib.sample import sobol as _sobol_sample

    sorted_keys = sorted(perturb)
    rvs = [to_rv_frozen(perturb[k]) for k in sorted_keys]

    problem = _build_problem(sorted_keys)
    unit = _sobol_sample.sample(problem, n, calc_second_order=calc_second_order, seed=seed)

    columns: dict[str, Any] = {
        key: rvs[idx].ppf(unit[:, idx]) for idx, key in enumerate(sorted_keys)
    }
    return pd.DataFrame(columns)


def sobol_analyze(
    df: pd.DataFrame,
    perturb: Mapping[str, DistSpec],
    metric: str | Callable[[pd.DataFrame], pd.Series],
    *,
    calc_second_order: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compute Sobol indices on a sweep result via :mod:`SALib.analyze.sobol`.

    Reduces ``df`` to a per-run scalar Y vector via ``metric``, then runs
    SALib's Sobol analysis and returns a tidy long DataFrame with columns
    ``["kind", "param_a", "param_b", "value", "conf"]``:

    * ``kind`` is ``"S1"`` (first-order), ``"ST"`` (total-order), or
      ``"S2"`` (second-order, only present when ``calc_second_order=True``).
    * ``param_a`` is the parameter name. ``param_b`` is the second parameter
      for ``S2`` rows and :data:`pandas.NA` for ``S1`` / ``ST`` rows.
    * ``value`` is the Sobol index. ``conf`` is SALib's 95 % bootstrap
      confidence half-width.

    Parameters
    ----------
    df:
        Sweep result DataFrame. Must come from a sweep launched with the
        DataFrame :func:`sobol_sample` produced — the row count and ordering
        are part of the Saltelli design. Failed or skipped runs cannot be
        ingested; filter them out beforehand (``df = df[df["__status"] == "ok"]``).
    perturb:
        Same mapping handed to :func:`sobol_sample`. Used to recover the
        sorted parameter list.
    metric:
        Reduces the per-(run_id, time) input to one scalar per run.
        ``str`` form takes the value of that column at each run's final
        time-step (``df.groupby(level="run_id")[metric].last()``) — the
        common end-of-mission state shape. Callable form receives ``df``
        and must return a :class:`pandas.Series` of length ``n * (2*D + 2)``
        (or ``n * (D + 2)`` when ``calc_second_order=False``).
    calc_second_order:
        Match the value passed to :func:`sobol_sample`.
    seed:
        Optional integer seed forwarded to SALib's bootstrap resampler.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, the input has any non-``ok`` status row,
        the metric column is missing, the callable returns a non-Series,
        or the reduction yields any NaN value.
    """
    if not perturb:
        raise SweepConfigError("sobol_analyze requires a non-empty perturb mapping")

    from SALib.analyze import sobol as _sobol_analyze

    sorted_keys = sorted(perturb)

    y = _reduce_to_y(df, metric)

    problem = _build_problem(sorted_keys)
    result = _sobol_analyze.analyze(
        problem,
        np.asarray(y, dtype=float),
        calc_second_order=calc_second_order,
        seed=seed,
    )

    return _pack_indices(result, sorted_keys, calc_second_order=calc_second_order)


def _build_problem(sorted_keys: list[str]) -> dict[str, Any]:
    # Bounds are unit-cube; the per-parameter ppf lift in sobol_sample handles
    # the actual distributions, generalising beyond SALib's own `dists` knob.
    return {
        "num_vars": len(sorted_keys),
        "names": list(sorted_keys),
        "bounds": [[0.0, 1.0]] * len(sorted_keys),
    }


def _reduce_to_y(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], pd.Series],
) -> pd.Series:
    if "__status" in df.columns:
        not_ok = df["__status"].astype(str).ne("ok")
        if bool(not_ok.any()):
            raise SweepConfigError(
                "sobol_analyze rejects sweep results with non-ok status rows; "
                "filter to passing runs first, e.g. df = df[df['__status'] == 'ok']"
            )

    if isinstance(metric, str):
        if metric not in df.columns:
            raise SweepConfigError(
                f"sobol_analyze metric column {metric!r} not found in df.columns"
            )
        if "run_id" not in (df.index.names or ()):
            raise SweepConfigError(
                "sobol_analyze with a str metric requires df indexed by run_id "
                "(the default sweep() output)"
            )
        y = df.groupby(level="run_id")[metric].last()
    else:
        y_raw = metric(df)
        if not isinstance(y_raw, pd.Series):
            raise SweepConfigError(
                f"sobol_analyze metric callable must return a pandas.Series, "
                f"got {type(y_raw).__name__}"
            )
        y = y_raw

    if bool(y.isna().any()):
        raise SweepConfigError(
            "sobol_analyze metric reduction produced NaN values; "
            "SALib's analyse step cannot ingest NaN"
        )
    return y


def _pack_indices(
    result: Mapping[str, np.ndarray],
    sorted_keys: list[str],
    *,
    calc_second_order: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind in ("S1", "ST"):
        values = result[kind]
        confs = result[f"{kind}_conf"]
        for i, name in enumerate(sorted_keys):
            rows.append(
                {
                    "kind": kind,
                    "param_a": name,
                    "param_b": pd.NA,
                    "value": float(values[i]),
                    "conf": float(confs[i]),
                }
            )
    if calc_second_order:
        s2 = result["S2"]
        s2_conf = result["S2_conf"]
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                rows.append(
                    {
                        "kind": "S2",
                        "param_a": sorted_keys[i],
                        "param_b": sorted_keys[j],
                        "value": float(s2[i, j]),
                        "conf": float(s2_conf[i, j]),
                    }
                )
    return pd.DataFrame(rows, columns=["kind", "param_a", "param_b", "value", "conf"])
