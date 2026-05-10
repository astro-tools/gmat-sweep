"""Plot helpers for sweep DataFrames — corner/pair plots, 2-axis heatmaps, and bands.

The public entry points cover the figures users hand-roll most often on
top of a sweep result:

- :func:`sweep_corner` — pair plot of the perturbed dotted-paths coloured
  by a per-run scalar metric. Useful for Monte Carlo and Latin hypercube
  dispersions.
- :func:`sweep_heatmap` — 2-D grid heatmap. Asserts the sweep was a
  two-axis :func:`gmat_sweep.sweep` grid and pivots the per-run metric
  into a matrix.
- :func:`sweep_band_plot` — median-and-band plot of one column over time
  (or run_id), reading a :func:`gmat_sweep.sweep_summary` DataFrame
  directly.

The corner/heatmap helpers consume the v0.2
``(run_id, time)``-MultiIndexed DataFrame :func:`gmat_sweep.sweep`
returns. Per-run parameter values come from either the DataFrame itself
(when the perturbed dotted-path is also a report column) or from
``manifest.entries[i].overrides`` — the resolution order is documented
per helper. :func:`sweep_band_plot` instead consumes the per-``time``
(or per-``run_id``) statistics frame :func:`sweep_summary` returns.

This module imports :mod:`matplotlib` lazily (inside each helper) so
``import gmat_sweep.plotting`` succeeds without the ``[plot]`` extra; the
real :class:`ImportError` fires only on first call. ``pip install
gmat-sweep[plot]`` adds the matplotlib pin.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from gmat_sweep.errors import SweepConfigError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from gmat_sweep.manifest import Manifest

__all__ = ["mc_convergence_plot", "sweep_band_plot", "sweep_corner", "sweep_heatmap"]


def sweep_corner(
    df: pd.DataFrame,
    params: Sequence[str] | None = None,
    metric: str | Callable[[pd.DataFrame], pd.Series[Any]] | None = None,
    *,
    manifest: Manifest | None = None,
    axes: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Render a corner/pair plot of ``params`` coloured by ``metric``.

    The helper builds one point per ``run_id``: each axis pair shows a
    scatter of the two parameter values, and the diagonal carries a
    histogram of each parameter's marginal distribution. The off-diagonal
    scatters share a colormap driven by a per-run scalar derived from
    ``metric``.

    Parameters
    ----------
    df:
        ``(run_id, time)``-MultiIndexed DataFrame as returned by
        :func:`gmat_sweep.sweep`, :func:`gmat_sweep.monte_carlo`, or
        :func:`gmat_sweep.latin_hypercube`. The ``__status`` column is
        used to drop failed and skipped runs.
    params:
        Sequence of dotted-path names to plot on each axis. ``None``
        (default) auto-loads them from ``manifest.parameter_spec`` —
        ``manifest`` must then be supplied.
    metric:
        Per-run scalar that colours the off-diagonal scatters. Pass a
        column name in ``df`` to reduce the column per ``run_id`` via
        ``.last()`` (the final time-step's value, matching the existing
        notebook idiom), or a callable ``df -> Series`` that returns a
        :class:`pandas.Series` indexed by ``run_id``.
    manifest:
        Optional :class:`gmat_sweep.manifest.Manifest`. Required when
        ``params`` is ``None`` or any param is not already a column of
        ``df``; supplies per-run override values via
        ``entry.overrides[param]``.
    axes:
        Optional pre-existing 2-D ndarray of :class:`matplotlib.axes.Axes`
        (shape ``(N, N)`` where ``N == len(params)``). ``None`` (default)
        creates a fresh figure and axes grid sized
        ``(2.5 * N, 2.5 * N)`` inches.
    **kwargs:
        Forwarded to :meth:`matplotlib.axes.Axes.scatter` for the
        off-diagonal panels (e.g. ``s=12``, ``alpha=0.6``, ``cmap=...``).

    Returns
    -------
    numpy.ndarray
        The 2-D ndarray of :class:`matplotlib.axes.Axes` (shape
        ``(N, N)``). Save the figure via ``axes[0, 0].figure.savefig(...)``.

    Raises
    ------
    SweepConfigError
        If ``params`` is ``None`` and ``manifest`` is ``None``, if
        ``metric`` is ``None``, if any param cannot be resolved from
        either ``df`` or ``manifest``, if a callable ``metric`` returns
        a Series not indexed by ``run_id``, or if ``axes`` has the wrong
        shape.
    """
    import matplotlib.pyplot as plt

    resolved_params = _resolve_params(params, manifest)
    if metric is None:
        raise SweepConfigError("sweep_corner requires metric=")
    if len(resolved_params) < 2:
        raise SweepConfigError(f"sweep_corner needs at least 2 params; got {len(resolved_params)}")

    param_values = _gather_param_values(df, resolved_params, manifest)
    metric_series = _resolve_metric(df, metric)

    # Inner-join on run_id so dropped failures (and any run_id the metric
    # callable chose to omit) don't pollute the plot.
    joined = param_values.join(metric_series.rename("__metric"), how="inner")
    status_per_run = _status_per_run(df)
    if status_per_run is not None:
        ok_mask = status_per_run.reindex(joined.index) == "ok"
        dropped = int((~ok_mask).sum())
        if dropped:
            warnings.warn(
                f"sweep_corner: dropped {dropped} non-ok run(s) from the plot",
                RuntimeWarning,
                stacklevel=2,
            )
        joined = joined.loc[ok_mask]

    if joined.empty:
        raise SweepConfigError(
            "sweep_corner: no ok runs left to plot after dropping failed/skipped"
        )

    n = len(resolved_params)
    if axes is None:
        _, axes_arr = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), squeeze=False)
    else:
        axes_arr = np.asarray(axes)
        if axes_arr.shape != (n, n):
            raise SweepConfigError(
                f"sweep_corner: axes shape {axes_arr.shape} does not match len(params)=({n}, {n})"
            )

    metric_arr = joined["__metric"].to_numpy()
    cmap = kwargs.pop("cmap", "viridis")
    scatter_kwargs: dict[str, Any] = {"s": 16, "alpha": 0.7, **kwargs}

    last_scatter = None
    for i, p_row in enumerate(resolved_params):
        for j, p_col in enumerate(resolved_params):
            ax = axes_arr[i, j]
            if i == j:
                ax.hist(joined[p_row].to_numpy(), bins="auto", color="steelblue")
                if i == n - 1:
                    ax.set_xlabel(p_col)
                if j == 0:
                    ax.set_ylabel("count")
            elif i > j:
                last_scatter = ax.scatter(
                    joined[p_col].to_numpy(),
                    joined[p_row].to_numpy(),
                    c=metric_arr,
                    cmap=cmap,
                    **scatter_kwargs,
                )
                if i == n - 1:
                    ax.set_xlabel(p_col)
                if j == 0:
                    ax.set_ylabel(p_row)
            else:
                ax.set_visible(False)

    if last_scatter is not None:
        fig = axes_arr[0, 0].figure
        metric_label = metric if isinstance(metric, str) else "metric"
        fig.colorbar(last_scatter, ax=axes_arr.ravel().tolist(), label=metric_label)

    return cast("NDArray[Any]", axes_arr)


def sweep_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str | Callable[[pd.DataFrame], pd.Series[Any]],
    *,
    manifest: Manifest | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Render a 2-D heatmap of ``z`` over a full-factorial ``(x, y)`` grid.

    Reduces ``z`` to one scalar per ``run_id``, looks each run's ``(x, y)``
    cell up via the same resolution rules as :func:`sweep_corner`, and
    pivots the result into a matrix rendered with
    :meth:`matplotlib.axes.Axes.pcolormesh`. Failed and skipped runs land
    as :data:`numpy.nan` cells; the colormap's ``set_bad`` paints them in
    a distinct light-gray so the gap is obvious.

    Parameters
    ----------
    df:
        ``(run_id, time)``-MultiIndexed DataFrame from a sweep call.
    x, y:
        Dotted-path names of the two grid axes.
    z:
        Per-run metric — same shape as :func:`sweep_corner`'s ``metric``:
        a column in ``df`` (reduced via ``.last()``) or a callable
        returning a :class:`pandas.Series` indexed by ``run_id``.
    manifest:
        Optional :class:`gmat_sweep.manifest.Manifest`. Required when
        ``x`` or ``y`` is not already a column of ``df``. When supplied,
        also asserts the sweep was a two-axis ``grid``-kind sweep — a
        Monte Carlo or Latin hypercube manifest raises
        :class:`SweepConfigError` pointing at :func:`sweep_corner`.
    ax:
        Optional pre-existing :class:`matplotlib.axes.Axes`. ``None``
        (default) creates a fresh figure with size ``(8, 5)`` inches.
    **kwargs:
        Forwarded to :meth:`matplotlib.axes.Axes.pcolormesh` (e.g.
        ``cmap=...``, ``shading="auto"``, ``norm=...``).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes carrying the heatmap. Save the figure via
        ``ax.figure.savefig(...)``.

    Raises
    ------
    SweepConfigError
        If ``manifest`` was supplied and its ``parameter_spec`` is not a
        2-axis grid; if ``x`` or ``y`` cannot be resolved from either
        ``df`` or ``manifest``; or if ``z`` is a callable whose Series
        is not indexed by ``run_id``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if manifest is not None:
        _assert_two_axis_grid(manifest, x, y)

    param_values = _gather_param_values(df, [x, y], manifest)
    metric_series = _resolve_metric(df, z)
    joined = param_values.join(metric_series.rename("__metric"), how="left")

    status_per_run = _status_per_run(df)
    if status_per_run is not None:
        # Mask non-ok runs to NaN so they show through as missing cells.
        aligned = status_per_run.reindex(joined.index)
        non_ok = (aligned != "ok") & aligned.notna()
        joined.loc[non_ok, "__metric"] = np.nan

    pivot = joined.pivot_table(index=y, columns=x, values="__metric", aggfunc="first")
    if pivot.empty:
        raise SweepConfigError(
            "sweep_heatmap: no rows to plot — verify the sweep produced runs at the "
            f"requested (x={x!r}, y={y!r}) axes"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    cmap_input = kwargs.pop("cmap", "viridis")
    cmap = mpl.colormaps[cmap_input] if isinstance(cmap_input, str) else cmap_input
    cmap = cmap.copy()
    cmap.set_bad("lightgray")

    x_edges = _edges_from_centres(np.asarray(pivot.columns, dtype=float))
    y_edges = _edges_from_centres(np.asarray(pivot.index, dtype=float))
    masked = np.ma.masked_invalid(pivot.to_numpy())
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, **kwargs)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    z_label = z if isinstance(z, str) else "metric"
    ax.figure.colorbar(mesh, ax=ax, label=z_label)
    return ax


def _resolve_params(params: Sequence[str] | None, manifest: Manifest | None) -> list[str]:
    if params is not None:
        names = list(params)
        if not names:
            raise SweepConfigError("sweep_corner: params must not be empty")
        return names
    if manifest is None:
        raise SweepConfigError(
            "sweep_corner requires either params= or manifest= to derive the perturbed dotted-paths"
        )
    return _params_from_parameter_spec(manifest.parameter_spec)


def _params_from_parameter_spec(parameter_spec: dict[str, Any]) -> list[str]:
    """Pull the perturbed dotted-paths out of a manifest's parameter_spec.

    Supports all four ``_kind`` shapes produced by the public sweep entry
    points (`grid`, `explicit`, `monte_carlo`, `latin_hypercube`) plus
    older untagged grid manifests.
    """
    kind = parameter_spec.get("_kind")
    if kind is None or kind == "grid":
        return sorted(k for k in parameter_spec if k != "_kind")
    if kind == "explicit":
        return [str(c) for c in parameter_spec["columns"]]
    if kind in {"monte_carlo", "latin_hypercube"}:
        return list(parameter_spec["perturb"].keys())
    raise SweepConfigError(f"unknown parameter_spec _kind: {kind!r} — cannot derive params")


def _gather_param_values(
    df: pd.DataFrame,
    params: Sequence[str],
    manifest: Manifest | None,
) -> pd.DataFrame:
    """Build a ``run_id``-indexed DataFrame of per-run values for ``params``.

    Per-param resolution order: prefer the column already in ``df``
    (reduced via ``.first()`` since perturbed values are constant per
    run); fall back to ``manifest.entries[i].overrides[param]``; raise
    if neither path resolves.
    """
    columns: dict[str, pd.Series[Any]] = {}
    missing: list[str] = []
    for p in params:
        if p in df.columns:
            columns[p] = df.groupby(level="run_id")[p].first()
        elif manifest is not None:
            try:
                columns[p] = pd.Series(
                    {e.run_id: e.overrides[p] for e in manifest.entries},
                    name=p,
                )
            except KeyError:
                missing.append(p)
        else:
            missing.append(p)
    if missing:
        raise SweepConfigError(
            f"sweep_corner/sweep_heatmap: cannot resolve param(s) {missing!r} — "
            "they are not columns of df and (when manifest= is supplied) not in "
            "every entry's overrides. Pass manifest= or merge the values into df."
        )
    out = pd.DataFrame(columns)
    out.index.name = "run_id"
    return out


def _resolve_metric(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], pd.Series[Any]],
) -> pd.Series[Any]:
    if callable(metric):
        result = metric(df)
        if not isinstance(result, pd.Series):
            raise SweepConfigError(
                f"sweep_corner/sweep_heatmap: callable metric must return a "
                f"pandas.Series, got {type(result).__name__}"
            )
        if result.index.name != "run_id":
            raise SweepConfigError(
                "sweep_corner/sweep_heatmap: callable metric must return a Series "
                f"indexed by 'run_id', got index.name={result.index.name!r}"
            )
        return result
    if metric not in df.columns:
        raise SweepConfigError(
            f"sweep_corner/sweep_heatmap: metric={metric!r} is not a column of df "
            "and is not callable"
        )
    return df.groupby(level="run_id")[metric].last()


def _status_per_run(df: pd.DataFrame) -> pd.Series[Any] | None:
    if "__status" not in df.columns:
        return None
    # __status is constant per run (failed/skipped runs land as a single
    # NaN row with status set; ok runs have status="ok" on every row).
    return df.groupby(level="run_id")["__status"].first()


def _assert_two_axis_grid(manifest: Manifest, x: str, y: str) -> None:
    spec = manifest.parameter_spec
    kind = spec.get("_kind")
    if kind not in (None, "grid"):
        raise SweepConfigError(
            f"sweep_heatmap requires a 2-axis grid sweep; manifest is {kind!r}. "
            "Use sweep_corner for non-grid (Monte Carlo, Latin hypercube, "
            "explicit-row) dispersions."
        )
    axes = sorted(k for k in spec if k != "_kind")
    if len(axes) != 2:
        raise SweepConfigError(
            f"sweep_heatmap requires exactly 2 grid axes; manifest has {len(axes)} "
            f"({axes!r}). Use sweep_corner for higher-dimensional sweeps."
        )
    missing = [name for name in (x, y) if name not in axes]
    if missing:
        raise SweepConfigError(
            f"sweep_heatmap: requested axis/axes {missing!r} not in manifest grid axes {axes!r}"
        )


def _edges_from_centres(centres: NDArray[Any]) -> NDArray[Any]:
    """Convert sorted-unique cell centres to ``len(centres)+1`` edges.

    Inner edges are mid-points; the outer edges extrapolate by half the
    nearest-neighbour gap so the outer cells span the same width as the
    nearest interior cell. ``pcolormesh`` with ``shading="flat"`` (the
    default once edges are passed) needs one more edge than centre per
    axis.
    """
    sorted_centres = np.sort(np.unique(centres))
    if sorted_centres.size == 1:
        # Degenerate single-cell axis: pick a unit-width cell so the mesh
        # still renders.
        c = float(sorted_centres[0])
        return np.array([c - 0.5, c + 0.5])
    diffs = np.diff(sorted_centres)
    inner = sorted_centres[:-1] + diffs / 2.0
    first = sorted_centres[0] - diffs[0] / 2.0
    last = sorted_centres[-1] + diffs[-1] / 2.0
    return np.concatenate(([first], inner, [last]))


def sweep_band_plot(
    summary: pd.DataFrame,
    column: str,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot one ``column``'s median and quantile band from a ``sweep_summary`` frame.

    Reads the two-level ``("statistic", "field")`` column index produced
    by :func:`gmat_sweep.sweep_summary`, slices the band — the lowest
    and highest ``q*`` statistics for ``column`` — and draws a
    :meth:`matplotlib.axes.Axes.fill_between` shaded band plus a centre
    line. The centre is the median (``q0.5``) when present, otherwise
    falls back to ``mean``. The x-axis is the summary's row index
    (``time`` for ``by="time"``, ``run_id`` for ``by="run_id"``).

    Parameters
    ----------
    summary
        DataFrame as returned by :func:`gmat_sweep.sweep_summary`. Must
        carry the two-level ``("statistic", "field")`` column index.
    column
        Original data-column name to plot — the ``"field"`` slice under
        every statistic in the summary's column index.
    ax
        Optional pre-existing :class:`matplotlib.axes.Axes`. ``None``
        (default) creates a fresh figure with size ``(8, 4)`` inches.
    **kwargs
        Forwarded to the centre :meth:`matplotlib.axes.Axes.plot` call
        (e.g. ``label=``, ``linestyle=``, ``linewidth=``). The band's
        ``fill_between`` reuses the line colour at ``alpha=0.25``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes carrying the band plot. Save the figure via
        ``ax.figure.savefig(...)``.

    Raises
    ------
    SweepConfigError
        ``summary.columns`` is not a 2-level MultiIndex; ``column`` does
        not appear in the ``"field"`` level; or no centre statistic
        (neither a quantile nor ``mean``) is available for ``column``.
    """
    import matplotlib.pyplot as plt

    if not isinstance(summary.columns, pd.MultiIndex) or summary.columns.nlevels != 2:
        raise SweepConfigError(
            "sweep_band_plot: summary must have a 2-level column MultiIndex "
            "as produced by gmat_sweep.sweep_summary"
        )

    field_level = summary.columns.get_level_values(1)
    if column not in field_level:
        available = sorted(set(field_level))
        raise SweepConfigError(
            f"sweep_band_plot: column={column!r} not found in summary; "
            f"available fields: {available}"
        )

    stat_level = summary.columns.get_level_values(0)
    quantile_stats: list[tuple[float, str]] = []
    for stat in dict.fromkeys(stat_level):
        if isinstance(stat, str) and stat.startswith("q"):
            try:
                quantile_stats.append((float(stat[1:]), stat))
            except ValueError:
                continue
    quantile_stats.sort()

    has_mean = "mean" in set(stat_level)

    centre_stat: str | None = None
    for q_val, stat in quantile_stats:
        if q_val == 0.5:
            centre_stat = stat
            break
    if centre_stat is None and has_mean:
        centre_stat = "mean"
    if centre_stat is None:
        raise SweepConfigError(
            f"sweep_band_plot: column={column!r} has no centre statistic "
            "(need either q=0.5 or mean in include=)"
        )

    band: tuple[str, str] | None = (
        (quantile_stats[0][1], quantile_stats[-1][1]) if len(quantile_stats) >= 2 else None
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    x = summary.index.to_numpy()
    centre = summary[(centre_stat, column)].to_numpy()

    plot_kwargs: dict[str, Any] = {"label": f"{column} ({centre_stat})", **kwargs}
    (line,) = ax.plot(x, centre, **plot_kwargs)

    if band is not None:
        lo = summary[(band[0], column)].to_numpy()
        hi = summary[(band[1], column)].to_numpy()
        ax.fill_between(
            x,
            lo,
            hi,
            color=line.get_color(),
            alpha=0.25,
            label=f"{column} ({band[0]} to {band[1]})",
        )

    x_label = summary.index.name or "index"
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(column)
    return ax


def mc_convergence_plot(
    conv: pd.DataFrame,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a Monte Carlo convergence curve from :func:`gmat_sweep.mc_convergence` output.

    Renders the running mean as a line and the ``±1·SE`` envelope as a
    shaded :meth:`matplotlib.axes.Axes.fill_between` band, with ``n``
    (sample count) on the x-axis. Visualises whether the per-run scalar
    metric has stabilised: a flattening centre line and a shrinking band
    indicate convergence.

    Parameters
    ----------
    conv
        Long-form convergence DataFrame as returned by
        :func:`gmat_sweep.mc_convergence` with ``terminal_only=True``
        (or with a callable metric). Must carry the columns ``n``,
        ``running_mean``, and ``se_mean``. The per-time shape returned
        by ``terminal_only=False`` is not supported in this release —
        filter to a single ``time`` first or aggregate to terminal-only.
    ax
        Optional pre-existing :class:`matplotlib.axes.Axes`. ``None``
        (default) creates a fresh figure with size ``(8, 4)`` inches.
    **kwargs
        Forwarded to the centre :meth:`matplotlib.axes.Axes.plot` call
        (e.g. ``label=``, ``linestyle=``, ``linewidth=``). The band's
        ``fill_between`` reuses the line colour at ``alpha=0.25``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes carrying the convergence plot.

    Raises
    ------
    SweepConfigError
        ``conv`` is missing one of the required columns, or carries the
        per-time ``time`` column from the ``terminal_only=False`` shape.
    """
    import matplotlib.pyplot as plt

    required = {"n", "running_mean", "se_mean"}
    missing = required - set(conv.columns)
    if missing:
        raise SweepConfigError(
            f"mc_convergence_plot: input is missing required column(s) {sorted(missing)}; "
            f"got columns {list(conv.columns)}"
        )
    if "time" in conv.columns:
        raise SweepConfigError(
            "mc_convergence_plot: per-time convergence frames are not supported; "
            "filter to a single time (e.g. conv[conv['time'] == t]) or rerun "
            "mc_convergence(terminal_only=True)"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n = conv["n"].to_numpy()
    mean = conv["running_mean"].to_numpy()
    se = conv["se_mean"].to_numpy()

    plot_kwargs: dict[str, Any] = {"label": "running mean", **kwargs}
    (line,) = ax.plot(n, mean, **plot_kwargs)
    ax.fill_between(
        n,
        mean - se,
        mean + se,
        color=line.get_color(),
        alpha=0.25,
        label="±1·SE",
    )

    ax.set_xlabel("n")
    ax.set_ylabel("running mean")
    return ax
