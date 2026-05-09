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

from matplotlib.collections import PathCollection, QuadMesh  # noqa: E402

from gmat_sweep.errors import SweepConfigError  # noqa: E402
from gmat_sweep.manifest import Manifest, ManifestEntry  # noqa: E402
from gmat_sweep.plotting import sweep_corner, sweep_heatmap  # noqa: E402


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
