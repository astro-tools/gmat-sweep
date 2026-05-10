"""Tests for :mod:`gmat_sweep.plotting` — sweep_corner and sweep_heatmap."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

# Skip this module entirely when matplotlib (the [plot] extra) is absent.
# Local checkouts without `uv sync --extra plot` skip cleanly; CI's test job
# installs --extra plot.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.collections import PathCollection, PolyCollection, QuadMesh  # noqa: E402

from gmat_sweep.aggregate import mc_convergence, sweep_summary  # noqa: E402
from gmat_sweep.errors import SweepConfigError  # noqa: E402
from gmat_sweep.manifest import Manifest, ManifestEntry  # noqa: E402
from gmat_sweep.plotting import (  # noqa: E402
    mc_convergence_plot,
    sweep_band_plot,
    sweep_corner,
    sweep_heatmap,
)


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc)


def _make_entry(
    run_id: int,
    overrides: dict[str, Any],
    status: str = "ok",
) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides=overrides,
        status=status,  # type: ignore[arg-type]
        output_paths={},
        started_at=_ts(0),
        ended_at=_ts(1),
        duration_s=1.0,
        stderr=None,
        log_path=None,
    )


def _make_manifest(entries: list[ManifestEntry], parameter_spec: dict[str, Any]) -> Manifest:
    return Manifest(
        script_sha256="0" * 64,
        gmat_sweep_version="0.0.0-test",
        gmat_run_version="unknown",
        gmat_install_version="unknown",
        python_version="3.12.0",
        os_platform="linux",
        sweep_seed=None,
        parameter_spec=parameter_spec,
        run_count=len(entries),
        entries=list(entries),
    )


def _multiindex_df(
    rows_per_run: dict[int, list[dict[str, Any]]],
    *,
    statuses: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build a ``(run_id, time)``-MultiIndexed DataFrame for the helpers."""
    statuses = statuses or {}
    records: list[dict[str, Any]] = []
    for run_id, rows in rows_per_run.items():
        status = statuses.get(run_id, "ok")
        for r in rows:
            records.append({"run_id": run_id, "__status": status, **r})
    df = pd.DataFrame(records)
    return cast(pd.DataFrame, df.set_index(["run_id", "time"]))


# ---------------------------------------------------------------------------
# sweep_corner
# ---------------------------------------------------------------------------


def test_sweep_corner_basic_shape_and_data() -> None:
    """Manifest-driven param resolution; off-diagonals carry one scatter each."""
    df = _multiindex_df(
        {
            0: [{"time": _ts(0), "value": 10.0}, {"time": _ts(1), "value": 11.0}],
            1: [{"time": _ts(0), "value": 20.0}, {"time": _ts(1), "value": 22.0}],
            2: [{"time": _ts(0), "value": 30.0}, {"time": _ts(1), "value": 33.0}],
        }
    )
    manifest = _make_manifest(
        [
            _make_entry(0, {"Sat.SMA": 7000.0, "Sat.ECC": 0.001}),
            _make_entry(1, {"Sat.SMA": 7100.0, "Sat.ECC": 0.002}),
            _make_entry(2, {"Sat.SMA": 7200.0, "Sat.ECC": 0.003}),
        ],
        parameter_spec={
            "_kind": "monte_carlo",
            "perturb": {"Sat.SMA": ("normal", 7100.0, 50.0), "Sat.ECC": ("uniform", 0.001, 0.003)},
            "n": 3,
            "seed": 42,
        },
    )

    axes = sweep_corner(df, metric="value", manifest=manifest)

    assert axes.shape == (2, 2)
    # Off-diagonal has a scatter; opposite corner is hidden.
    bottom_left = axes[1, 0]
    scatters = [c for c in bottom_left.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 1
    offsets = np.asarray(scatters[0].get_offsets())
    assert offsets.shape == (3, 2)
    assert axes[0, 1].get_visible() is False
    # Diagonals have histogram patches.
    assert len(axes[0, 0].patches) > 0
    assert len(axes[1, 1].patches) > 0


def test_sweep_corner_params_resolved_from_df_columns() -> None:
    """When the dotted-path is already a df column, no manifest is needed."""
    rows_per_run = {
        rid: [
            {
                "time": _ts(0),
                "Sat.SMA": 7000.0 + 100.0 * rid,
                "Sat.ECC": 0.001 + 0.0005 * rid,
                "value": 10.0 + rid,
            },
            {
                "time": _ts(1),
                "Sat.SMA": 7000.0 + 100.0 * rid,
                "Sat.ECC": 0.001 + 0.0005 * rid,
                "value": 11.0 + rid,
            },
        ]
        for rid in range(4)
    }
    df = _multiindex_df(rows_per_run)
    axes = sweep_corner(df, params=["Sat.SMA", "Sat.ECC"], metric="value")

    bottom_left = axes[1, 0]
    scatters = [c for c in bottom_left.collections if isinstance(c, PathCollection)]
    offsets = np.asarray(scatters[0].get_offsets())
    assert offsets.shape == (4, 2)


def test_sweep_corner_callable_metric() -> None:
    """Callable metric must return a Series indexed by run_id."""
    df = _multiindex_df(
        {
            0: [{"time": _ts(0), "value": 1.0}, {"time": _ts(1), "value": 2.0}],
            1: [{"time": _ts(0), "value": 3.0}, {"time": _ts(1), "value": 4.0}],
        }
    )
    manifest = _make_manifest(
        [
            _make_entry(0, {"a": 1.0, "b": 10.0}),
            _make_entry(1, {"a": 2.0, "b": 20.0}),
        ],
        parameter_spec={"_kind": "grid", "a": [1.0, 2.0], "b": [10.0, 20.0]},
    )

    def metric(_df: pd.DataFrame) -> pd.Series[Any]:
        s = _df.groupby(level="run_id")["value"].max()
        s.index.name = "run_id"
        return s

    axes = sweep_corner(df, metric=metric, manifest=manifest)
    bottom_left = axes[1, 0]
    scatter = next(c for c in bottom_left.collections if isinstance(c, PathCollection))
    np.testing.assert_array_equal(scatter.get_array(), np.array([2.0, 4.0]))


def test_sweep_corner_drops_failed_runs_with_warning() -> None:
    """Failed runs are excluded; a RuntimeWarning summarises the drop count."""
    df = _multiindex_df(
        {
            0: [{"time": _ts(0), "value": 1.0}],
            1: [{"time": _ts(0), "value": 2.0}],
            2: [{"time": pd.NaT, "value": np.nan}],
        },
        statuses={2: "failed"},
    )
    manifest = _make_manifest(
        [
            _make_entry(0, {"a": 1.0, "b": 10.0}),
            _make_entry(1, {"a": 2.0, "b": 20.0}),
            _make_entry(2, {"a": 3.0, "b": 30.0}, status="failed"),
        ],
        parameter_spec={"_kind": "grid", "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]},
    )

    with pytest.warns(RuntimeWarning, match=r"dropped 1 non-ok run"):
        axes = sweep_corner(df, metric="value", manifest=manifest)
    scatter = next(c for c in axes[1, 0].collections if isinstance(c, PathCollection))
    assert np.asarray(scatter.get_offsets()).shape == (2, 2)


def test_sweep_corner_requires_metric() -> None:
    df = _multiindex_df({0: [{"time": _ts(0), "value": 1.0}]})
    manifest = _make_manifest(
        [_make_entry(0, {"a": 1.0, "b": 2.0})],
        parameter_spec={"_kind": "grid", "a": [1.0], "b": [2.0]},
    )
    with pytest.raises(SweepConfigError, match="requires metric"):
        sweep_corner(df, manifest=manifest)


def test_sweep_corner_requires_params_or_manifest() -> None:
    df = _multiindex_df({0: [{"time": _ts(0), "value": 1.0}]})
    with pytest.raises(SweepConfigError, match="params= or manifest="):
        sweep_corner(df, metric="value")


def test_sweep_corner_callable_metric_must_index_by_run_id() -> None:
    df = _multiindex_df({0: [{"time": _ts(0), "value": 1.0}], 1: [{"time": _ts(0), "value": 2.0}]})
    manifest = _make_manifest(
        [_make_entry(0, {"a": 1.0, "b": 10.0}), _make_entry(1, {"a": 2.0, "b": 20.0})],
        parameter_spec={"_kind": "grid", "a": [1.0, 2.0], "b": [10.0, 20.0]},
    )

    def bad_metric(_df: pd.DataFrame) -> pd.Series[Any]:
        return pd.Series([1.0, 2.0])  # no run_id index

    with pytest.raises(SweepConfigError, match="indexed by 'run_id'"):
        sweep_corner(df, metric=bad_metric, manifest=manifest)


# ---------------------------------------------------------------------------
# sweep_heatmap
# ---------------------------------------------------------------------------


def _grid_df_and_manifest(
    *,
    statuses: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, Manifest]:
    """3-by-2 grid (a in [1, 2, 3], b in [10, 20]) with metric == 10*b + a."""
    statuses = statuses or {}
    rows_per_run: dict[int, list[dict[str, Any]]] = {}
    entries: list[ManifestEntry] = []
    rid = 0
    for a in (1.0, 2.0, 3.0):
        for b in (10.0, 20.0):
            metric = 10.0 * b + a
            status = statuses.get(rid, "ok")
            rows: list[dict[str, Any]]
            if status == "ok":
                rows = [{"time": _ts(0), "value": metric}]
            else:
                rows = [{"time": pd.NaT, "value": np.nan}]
            rows_per_run[rid] = rows
            entries.append(_make_entry(rid, {"a": a, "b": b}, status=status))
            rid += 1
    df = _multiindex_df(rows_per_run, statuses=statuses)
    manifest = _make_manifest(
        entries,
        parameter_spec={"_kind": "grid", "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0]},
    )
    return df, manifest


def test_sweep_heatmap_basic() -> None:
    df, manifest = _grid_df_and_manifest()
    ax = sweep_heatmap(df, x="a", y="b", z="value", manifest=manifest)

    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    mesh = meshes[0]
    array = mesh.get_array()
    assert array is not None
    # 3 a-values by 2 b-values pivoted with a=columns, b=index -> shape (2, 3)
    assert array.shape == (2, 3)
    # Spot-check: (a=2, b=20) cell == 10*20 + 2 == 202
    pivot = pd.DataFrame(
        {
            "a": [e.overrides["a"] for e in manifest.entries],
            "b": [e.overrides["b"] for e in manifest.entries],
            "v": [10.0 * e.overrides["b"] + e.overrides["a"] for e in manifest.entries],
        }
    ).pivot_table(index="b", columns="a", values="v")
    np.testing.assert_array_equal(np.asarray(array), pivot.to_numpy())


def test_sweep_heatmap_failed_cell_is_nan_with_set_bad() -> None:
    """A failed run becomes a NaN cell; the colormap's set_bad is non-default."""
    df, manifest = _grid_df_and_manifest(statuses={2: "failed"})
    ax = sweep_heatmap(df, x="a", y="b", z="value", manifest=manifest)
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))

    array = mesh.get_array()
    # The mesh array is a masked array; assert the failed cell is masked.
    masked_array: np.ma.MaskedArray[Any, Any] = np.ma.asarray(array)
    assert masked_array.mask.any()

    # The colormap should have its bad colour overridden to lightgray (rgba).
    bad = mesh.cmap.get_bad()
    assert bad[3] > 0  # alpha set
    assert tuple(bad[:3]) == pytest.approx(matplotlib.colors.to_rgb("lightgray"), abs=1e-6)


def test_sweep_heatmap_rejects_non_grid_manifest() -> None:
    df = _multiindex_df({0: [{"time": _ts(0), "value": 1.0}]})
    manifest = _make_manifest(
        [_make_entry(0, {"a": 1.0, "b": 2.0})],
        parameter_spec={
            "_kind": "monte_carlo",
            "perturb": {"a": ("normal", 1.0, 0.1), "b": ("normal", 2.0, 0.1)},
            "n": 1,
            "seed": 0,
        },
    )
    with pytest.raises(SweepConfigError, match="sweep_corner"):
        sweep_heatmap(df, x="a", y="b", z="value", manifest=manifest)


def test_sweep_heatmap_rejects_wrong_dimensionality() -> None:
    df = _multiindex_df({0: [{"time": _ts(0), "value": 1.0}]})
    manifest = _make_manifest(
        [_make_entry(0, {"a": 1.0, "b": 2.0, "c": 3.0})],
        parameter_spec={"_kind": "grid", "a": [1.0], "b": [2.0], "c": [3.0]},
    )
    with pytest.raises(SweepConfigError, match="exactly 2 grid axes"):
        sweep_heatmap(df, x="a", y="b", z="value", manifest=manifest)


def test_sweep_heatmap_callable_z() -> None:
    df, manifest = _grid_df_and_manifest()

    def z(_df: pd.DataFrame) -> pd.Series[Any]:
        s = _df.groupby(level="run_id")["value"].mean()
        s.index.name = "run_id"
        return s

    ax = sweep_heatmap(df, x="a", y="b", z=z, manifest=manifest)
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    assert np.asarray(mesh.get_array()).shape == (2, 3)


# ---------------------------------------------------------------------------
# sweep_band_plot
# ---------------------------------------------------------------------------


def _band_summary(n_runs: int = 50, n_steps: int = 4) -> pd.DataFrame:
    """A sweep-summary frame with two columns for sweep_band_plot tests."""
    rows: list[dict[str, Any]] = []
    times = pd.to_datetime([f"2026-05-04T00:00:0{i}" for i in range(n_steps)])
    for run_id in range(n_runs):
        for step, t in enumerate(times):
            rows.append(
                {
                    "run_id": run_id,
                    "time": t,
                    "value": float(run_id) + step * 0.5,
                    "other": float(run_id) - step,
                    "__status": "ok",
                }
            )
    df = pd.DataFrame(rows).set_index(["run_id", "time"])
    return sweep_summary(df)


def test_sweep_band_plot_returns_axes_with_line_and_band() -> None:
    summary = _band_summary()
    ax = sweep_band_plot(summary, "value")

    # One line for the centre and one PolyCollection for the fill_between band.
    assert len(ax.lines) == 1
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 1
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "value"


def test_sweep_band_plot_uses_q_5_as_centre_and_q_band_extremes() -> None:
    summary = _band_summary(n_runs=20, n_steps=3)
    ax = sweep_band_plot(summary, "value")

    centre = ax.lines[0].get_ydata()
    pd.testing.assert_series_equal(
        pd.Series(centre, name=("q0.5", "value")),
        summary[("q0.5", "value")].reset_index(drop=True),
        check_names=False,
        check_index=False,
    )


def test_sweep_band_plot_falls_back_to_mean_without_q_5() -> None:
    rows: list[dict[str, Any]] = []
    for run_id in range(10):
        for step in range(3):
            rows.append(
                {
                    "run_id": run_id,
                    "time": pd.Timestamp(f"2026-05-04T00:00:0{step}"),
                    "v": float(run_id),
                    "__status": "ok",
                }
            )
    df = pd.DataFrame(rows).set_index(["run_id", "time"])
    summary = sweep_summary(df, q=(0.1, 0.9), include=("mean",))

    ax = sweep_band_plot(summary, "v")
    centre = ax.lines[0].get_ydata()
    assert centre == pytest.approx(summary[("mean", "v")].to_numpy())


def test_sweep_band_plot_rejects_unknown_column() -> None:
    summary = _band_summary(n_runs=4, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"not found in summary"):
        sweep_band_plot(summary, "nope")


def test_sweep_band_plot_rejects_flat_column_index() -> None:
    flat = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([0, 1], name="time"))
    with pytest.raises(SweepConfigError, match=r"2-level column MultiIndex"):
        sweep_band_plot(flat, "x")


def test_sweep_band_plot_rejects_summary_without_centre() -> None:
    rows: list[dict[str, Any]] = []
    for run_id in range(5):
        for step in range(2):
            rows.append(
                {
                    "run_id": run_id,
                    "time": pd.Timestamp(f"2026-05-04T00:00:0{step}"),
                    "v": float(run_id),
                    "__status": "ok",
                }
            )
    df = pd.DataFrame(rows).set_index(["run_id", "time"])
    summary = sweep_summary(df, q=(0.1, 0.9), include=())  # no q=0.5, no mean
    with pytest.raises(SweepConfigError, match=r"no centre statistic"):
        sweep_band_plot(summary, "v")


# ---------------------------------------------------------------------------
# mc_convergence_plot
# ---------------------------------------------------------------------------


def _convergence_df(n_runs: int = 30, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    terminal = rng.normal(loc=0.0, scale=1.0, size=n_runs)
    rows: list[dict[str, Any]] = []
    for run_id in range(n_runs):
        rows.append(
            {
                "run_id": run_id,
                "time": pd.Timestamp("2026-05-04T00:00:00"),
                "miss": float(terminal[run_id]),
                "__status": "ok",
            }
        )
    df = pd.DataFrame(rows).set_index(["run_id", "time"])
    return mc_convergence(df, "miss", terminal_only=True)


def test_mc_convergence_plot_returns_axes_with_line_and_band() -> None:
    conv = _convergence_df()
    ax = mc_convergence_plot(conv)

    assert len(ax.lines) == 1
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 1
    assert ax.get_xlabel() == "n"
    assert ax.get_ylabel() == "running mean"


def test_mc_convergence_plot_line_x_is_n_and_y_is_running_mean() -> None:
    conv = _convergence_df(n_runs=10)
    ax = mc_convergence_plot(conv)

    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), conv["n"].to_numpy())
    np.testing.assert_array_equal(line.get_ydata(), conv["running_mean"].to_numpy())


def test_mc_convergence_plot_rejects_per_time_frame() -> None:
    conv = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-04T00:00:00"]),
            "n": [1],
            "running_mean": [0.0],
            "running_std": [float("nan")],
            "se_mean": [float("nan")],
        }
    )
    with pytest.raises(SweepConfigError, match=r"per-time"):
        mc_convergence_plot(conv)


def test_mc_convergence_plot_rejects_missing_columns() -> None:
    conv = pd.DataFrame({"n": [1, 2], "running_mean": [0.0, 0.1]})  # no se_mean
    with pytest.raises(SweepConfigError, match=r"missing required column"):
        mc_convergence_plot(conv)
