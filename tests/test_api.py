"""Tests for gmat_sweep.api.sweep — end-to-end wrapper over Sweep + LocalJoblibPool."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.api import latin_hypercube, monte_carlo, sweep
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest
from tests.conftest import FakeGmatRun, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _payload_run_hook(value: float = 1.0) -> Any:
    payload = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [value]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- explicit out --------------------------------------------------------


def test_sweep_returns_multiindexed_dataframe(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    assert df.index.names == ["run_id", "time"]
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1]
    assert (out / "manifest.jsonl").exists()


def test_sweep_writes_manifest_with_grid_in_header(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
        seed=42,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.parameter_spec == {
        "_kind": "grid",
        "Sat.SMA": [7000.0, 7100.0],
    }
    assert manifest.sweep_seed == 42
    assert manifest.run_count == 2


def test_sweep_string_path_is_accepted(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(
        str(script),
        grid={"Sat.SMA": [7000.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    assert len(df) == 1


def test_sweep_creates_out_directory_when_missing(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "does" / "not" / "exist"
    assert not out.exists()

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(script, grid={"Sat.SMA": [7000.0]}, backend=LocalJoblibPool(max_workers=1), out=out)

    assert out.is_dir()
    assert (out / "manifest.jsonl").exists()


def test_sweep_resolves_relative_out_to_absolute(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GMAT resolves a relative `working_dir` against its installed
    # `OUTPUT_PATH` (e.g. `/opt/gmat/output/` in the canonical container
    # image), so a relative `out=` would land per-run files somewhere
    # other than where the user pointed. Public entry points must
    # absolutise before per-run dirs are seeded.
    script = _write_script(tmp_path)
    monkeypatch.chdir(tmp_path)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(script, grid={"Sat.SMA": [7000.0]}, backend=LocalJoblibPool(max_workers=1), out="rel-out")

    manifest_path = tmp_path / "rel-out" / "manifest.jsonl"
    manifest = Manifest.load(manifest_path)
    [entry] = manifest.entries
    assert entry.log_path is not None
    assert Path(entry.log_path).is_absolute()


# ---- no out: temp dir lifetime --------------------------------------------


def test_sweep_with_no_out_returns_usable_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """lazy_multiindex materialises Parquet in-memory before returning, so the
    DataFrame is usable even after the temp dir would have been GC'd."""
    script = _write_script(tmp_path)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(value=3.14))

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=None,
    )

    assert df["x"].iloc[0] == 3.14


def test_sweep_with_no_out_cleans_up_temp_dir_after_df_dropped(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the returned DataFrame triggers the weakref finalizer that
    cleans up the sweep-scoped temp directory."""
    import tempfile as _tempfile

    captured_paths: list[Path] = []
    real_tempdir = _tempfile.TemporaryDirectory

    def _spy_tempdir(*args: Any, **kwargs: Any) -> Any:
        td = real_tempdir(*args, **kwargs)
        captured_paths.append(Path(td.name))
        return td

    monkeypatch.setattr("gmat_sweep.api.tempfile.TemporaryDirectory", _spy_tempdir)

    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(script, grid={"Sat.SMA": [7000.0]}, backend=LocalJoblibPool(max_workers=1), out=None)

    assert len(captured_paths) == 1
    sweep_dir = captured_paths[0]
    assert sweep_dir.is_dir()

    del df
    gc.collect()

    assert not sweep_dir.exists()


# ---- failure modes don't raise -------------------------------------------


def test_sweep_with_failing_run_includes_failed_row(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    def _setitem(_key: str, value: Any) -> None:
        if value == 7100.0:
            raise ValueError("rejected")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())

    df = sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    statuses = set(df["__status"].unique())
    assert statuses == {"ok", "failed"}


# ---- empty grid ----------------------------------------------------------


def test_sweep_with_empty_grid_runs_once(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """full_factorial({}) yields a single empty override dict; sweep() honours it."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = sweep(script, grid={}, backend=LocalJoblibPool(max_workers=1), out=out)

    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == [0]


# ---- progress -----------------------------------------------------------


def test_sweep_progress_false_quiet_on_stderr(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """progress=False propagates from sweep() through to Sweep so the tqdm
    progress bar does not paint to stderr — needed when notebooks are
    committed with executed outputs (otherwise each tqdm refresh lands as a
    captured stderr snapshot in the .ipynb)."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
        progress=False,
    )

    captured = capsys.readouterr()
    assert "gmat-sweep" not in captured.err
    assert "%" not in captured.err


# ---- backend= plumbing ---------------------------------------------------


def test_sweep_default_backend_constructs_local_joblib_pool(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``backend=`` is omitted, ``_run_sweep`` builds a LocalJoblibPool
    and dispatches through it. Verified by intercepting the constructor and
    forcing ``max_workers=1`` so the fake ``gmat_run`` is visible to the
    in-process worker."""
    captured: list[LocalJoblibPool] = []

    def _intercept(*_args: Any, **_kwargs: Any) -> LocalJoblibPool:
        instance = LocalJoblibPool(max_workers=1)
        captured.append(instance)
        return instance

    monkeypatch.setattr("gmat_sweep.api.LocalJoblibPool", _intercept)

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(script, grid={"Sat.SMA": [7000.0]}, out=out)

    assert len(captured) == 1
    assert Manifest.load(out / "manifest.jsonl").backend == "LocalJoblibPool"


def test_sweep_user_supplied_backend_records_pool_class_name_on_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A user-supplied pool's ``__class__.__name__`` lands on the manifest
    header verbatim — the contract third-party Pool subclasses rely on."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    class _CustomLabelPool(LocalJoblibPool):
        pass

    with _CustomLabelPool(max_workers=1) as pool:
        sweep(script, grid={"Sat.SMA": [7000.0]}, backend=pool, out=out)

    manifest = Manifest.load(out / "manifest.jsonl")
    assert manifest.backend == "_CustomLabelPool"


def test_sweep_user_supplied_backend_is_not_closed_by_sweep(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Caller owns the supplied pool's lifecycle: sweep() must not call
    close() on it. Two consecutive sweeps share one pool."""
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    pool = LocalJoblibPool(max_workers=1)
    try:
        sweep(script, grid={"Sat.SMA": [7000.0]}, backend=pool, out=tmp_path / "out-a")
        # If sweep() had closed the pool, this second call would fail.
        sweep(script, grid={"Sat.SMA": [7100.0]}, backend=pool, out=tmp_path / "out-b")
    finally:
        pool.close()


def test_monte_carlo_user_backend_records_pool_class_on_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with LocalJoblibPool(max_workers=1) as pool:
        monte_carlo(
            script,
            n=4,
            perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
            seed=42,
            backend=pool,
            out=out,
        )

    assert Manifest.load(out / "manifest.jsonl").backend == "LocalJoblibPool"


def test_latin_hypercube_user_backend_records_pool_class_on_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with LocalJoblibPool(max_workers=1) as pool:
        latin_hypercube(
            script,
            n=4,
            perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
            seed=42,
            backend=pool,
            out=out,
        )

    assert Manifest.load(out / "manifest.jsonl").backend == "LocalJoblibPool"


# ---- module-import-time logger config -----------------------------------


def test_gmat_sweep_logger_default_level_is_warning() -> None:
    """gmat_sweep configures its top-level logger at module import; without it
    every per-run INFO record would land in the parent process's handlers."""
    import logging

    assert logging.getLogger("gmat_sweep").level == logging.WARNING


# ---- explicit-row sweep (samples=DataFrame) ------------------------------


def test_sweep_with_samples_returns_one_run_per_row(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Issue #34 headline acceptance: a 4-row DataFrame produces a sweep
    result whose ``run_id`` cardinality is 4 and whose per-row override is
    applied (verified by reading the override back via the fake setitem
    hook)."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    seen: list[tuple[str, Any]] = []

    def _setitem(key: str, value: Any) -> None:
        seen.append((key, value))

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())

    samples = pd.DataFrame({"Sat.SMA": [7000, 7100, 7200, 7300]})
    df = sweep(script, samples=samples, backend=LocalJoblibPool(max_workers=1), out=out)

    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == [0, 1, 2, 3]
    # Worker side saw the per-row override applied to the live mission.
    assert ("Sat.SMA", 7000) in seen
    assert ("Sat.SMA", 7100) in seen
    assert ("Sat.SMA", 7200) in seen
    assert ("Sat.SMA", 7300) in seen


def test_sweep_with_both_grid_and_samples_raises(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="not both"):
        sweep(
            script,
            grid={"Sat.SMA": [7000.0]},
            samples=pd.DataFrame({"Sat.SMA": [7000.0]}),
            backend=LocalJoblibPool(max_workers=1),
            out=tmp_path / "out",
        )


def test_sweep_with_neither_grid_nor_samples_raises(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="one of grid= or samples="):
        sweep(script, backend=LocalJoblibPool(max_workers=1), out=tmp_path / "out")


def test_sweep_with_samples_non_default_index_raises(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    samples = pd.DataFrame(
        {"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        index=pd.RangeIndex(10, 14),
    )
    with pytest.raises(SweepConfigError, match="default RangeIndex"):
        sweep(script, samples=samples, backend=LocalJoblibPool(max_workers=1), out=tmp_path / "out")


def test_sweep_with_grid_writes_grid_kind_in_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Manifest tags ``parameter_spec`` with ``_kind: "grid"`` from v1
    onward; the materialised axes ride alongside the discriminator."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0], "Sat.ECC": [0.001, 0.002]},
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    assert spec["_kind"] == "grid"
    assert spec["Sat.SMA"] == [7000.0, 7100.0]
    assert spec["Sat.ECC"] == [0.001, 0.002]


def test_sweep_with_samples_writes_explicit_kind_in_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Manifest tags the parameter_spec with ``_kind: "explicit"`` so a
    later loader can distinguish a sample-based sweep from a grid sweep."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, 7100.0],
            "Sat.ECC": [0.001, 0.002],
        }
    )
    sweep(script, samples=samples, backend=LocalJoblibPool(max_workers=1), out=out)

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    assert spec["_kind"] == "explicit"
    assert spec["columns"] == ["Sat.SMA", "Sat.ECC"]
    assert spec["rows"] == [[7000.0, 0.001], [7100.0, 0.002]]


def test_sweep_with_samples_manifest_round_trips_to_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Manifest.load(saved).parameter_spec reconstructs a DataFrame whose
    contents equal the input — the round-trip acceptance criterion."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    samples = pd.DataFrame(
        {
            "Sat.SMA": [7000.0, 7100.0, 7200.0],
            "Sat.ECC": [0.001, 0.002, 0.003],
        }
    )
    sweep(script, samples=samples, backend=LocalJoblibPool(max_workers=1), out=out)

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    reconstructed = pd.DataFrame(spec["rows"], columns=spec["columns"])
    pd.testing.assert_frame_equal(reconstructed, samples)


# ---- monte_carlo ---------------------------------------------------------


def test_monte_carlo_returns_n_run_dataframe(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """Issue #33 headline: ``monte_carlo(n=100, perturb=..., seed=42)``
    returns a ``(run_id, time)``-indexed DataFrame with run_id cardinality
    100."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = monte_carlo(
        script,
        n=100,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    assert df.index.names == ["run_id", "time"]
    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == list(range(100))


def test_monte_carlo_two_calls_at_same_seed_produce_equal_overrides(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Determinism contract: two calls at the same ``(n, perturb, seed)``
    produce manifests whose recorded per-run overrides are identical at
    every run_id."""
    script = _write_script(tmp_path)
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out_a,
    )
    monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out_b,
    )

    a = {e.run_id: e.overrides for e in Manifest.load(out_a / "manifest.jsonl").entries}
    b = {e.run_id: e.overrides for e in Manifest.load(out_b / "manifest.jsonl").entries}
    assert a == b


def test_monte_carlo_different_seed_produces_different_overrides(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out_a,
    )
    monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=43,
        backend=LocalJoblibPool(max_workers=1),
        out=out_b,
    )

    a = {e.run_id: e.overrides for e in Manifest.load(out_a / "manifest.jsonl").entries}
    b = {e.run_id: e.overrides for e in Manifest.load(out_b / "manifest.jsonl").entries}
    assert a != b


def test_monte_carlo_adding_a_perturbed_parameter_preserves_other_draws(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Issue #33 order-independence contract: adding a second perturbed
    parameter to an existing perturb dict does not change the draws of the
    first parameter at any run_id."""
    script = _write_script(tmp_path)
    out_one = tmp_path / "out-one"
    out_two = tmp_path / "out-two"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    monte_carlo(
        script,
        n=20,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out_one,
    )
    monte_carlo(
        script,
        n=20,
        perturb={
            "Sat.SMA": ("normal", 7100.0, 50.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out_two,
    )

    one = {
        e.run_id: e.overrides["Sat.SMA"] for e in Manifest.load(out_one / "manifest.jsonl").entries
    }
    two = {
        e.run_id: e.overrides["Sat.SMA"] for e in Manifest.load(out_two / "manifest.jsonl").entries
    }
    assert one == two


def test_monte_carlo_accepts_pre_frozen_rv(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    from scipy import stats

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = monte_carlo(
        script,
        n=10,
        perturb={"x": stats.beta(2, 5)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == list(range(10))


def test_monte_carlo_failed_run_lands_as_nan_row(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A worker exception inside an MC sweep surfaces as a NaN row with
    ``__status="failed"`` — same contract as the grid path."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    call_count = {"n": 0}

    def _setitem(_key: str, _value: Any) -> None:
        call_count["n"] += 1
        # Fail every other run.
        if call_count["n"] % 2 == 0:
            raise ValueError("rejected")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook())

    df = monte_carlo(
        script,
        n=8,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    statuses = set(df["__status"].unique())
    assert "failed" in statuses


def test_monte_carlo_writes_tagged_parameter_spec(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The manifest header carries a tagged ``parameter_spec`` so a later
    loader can reproduce the same draws from the seed alone."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    monte_carlo(
        script,
        n=10,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    assert spec["_kind"] == "monte_carlo"
    assert spec["n"] == 10
    assert spec["seed"] == 42
    assert spec["perturb"] == {"Sat.SMA": ["normal", 7100.0, 50.0]}
    assert manifest.sweep_seed == 42


def test_monte_carlo_rejects_empty_perturb(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="non-empty perturb"):
        monte_carlo(
            script,
            n=10,
            perturb={},
            seed=42,
            backend=LocalJoblibPool(max_workers=1),
            out=tmp_path / "out",
        )


def test_monte_carlo_rejects_n_less_than_one(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="requires n >= 1"):
        monte_carlo(
            script,
            n=0,
            perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
            seed=42,
            backend=LocalJoblibPool(max_workers=1),
            out=tmp_path / "out",
        )


# ---- latin_hypercube -----------------------------------------------------


def test_latin_hypercube_returns_n_run_dataframe(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Issue #35 headline: ``latin_hypercube(n=64, perturb=..., seed=42)``
    returns a ``(run_id, time)``-indexed DataFrame with run_id cardinality
    64."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    df = latin_hypercube(
        script,
        n=64,
        perturb={
            "Sat.SMA": ("normal", 7100.0, 50.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    assert df.index.names == ["run_id", "time"]
    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == list(range(64))


def test_latin_hypercube_two_calls_at_same_seed_produce_equal_overrides(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    perturb = {
        "Sat.SMA": ("normal", 7100.0, 50.0),
        "Sat.INC": ("uniform", 0.0, 90.0),
    }
    latin_hypercube(
        script, n=16, perturb=perturb, seed=42, backend=LocalJoblibPool(max_workers=1), out=out_a
    )
    latin_hypercube(
        script, n=16, perturb=perturb, seed=42, backend=LocalJoblibPool(max_workers=1), out=out_b
    )

    a = {e.run_id: e.overrides for e in Manifest.load(out_a / "manifest.jsonl").entries}
    b = {e.run_id: e.overrides for e in Manifest.load(out_b / "manifest.jsonl").entries}
    assert a == b


def test_latin_hypercube_writes_tagged_parameter_spec(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """Issue #35: manifest header records ``{"_kind": "latin_hypercube",
    "perturb": <serialised>, "n": n, "seed": seed}`` so the seed plus the
    spec is enough to reproduce the LH samples DataFrame."""
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    latin_hypercube(
        script,
        n=16,
        perturb={
            "Sat.SMA": ("normal", 7100.0, 50.0),
            "Sat.INC": ("uniform", 0.0, 90.0),
        },
        seed=42,
        backend=LocalJoblibPool(max_workers=1),
        out=out,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    spec = manifest.parameter_spec
    assert spec["_kind"] == "latin_hypercube"
    assert spec["n"] == 16
    assert spec["seed"] == 42
    assert spec["perturb"] == {
        "Sat.SMA": ["normal", 7100.0, 50.0],
        "Sat.INC": ["uniform", 0.0, 90.0],
    }
    assert manifest.sweep_seed == 42


def test_latin_hypercube_rejects_empty_perturb(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="non-empty perturb"):
        latin_hypercube(
            script,
            n=10,
            perturb={},
            seed=42,
            backend=LocalJoblibPool(max_workers=1),
            out=tmp_path / "out",
        )


def test_latin_hypercube_rejects_n_less_than_one(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook())

    with pytest.raises(SweepConfigError, match="requires n >= 1"):
        latin_hypercube(
            script,
            n=0,
            perturb={"Sat.SMA": ("normal", 7100.0, 50.0)},
            seed=42,
            backend=LocalJoblibPool(max_workers=1),
            out=tmp_path / "out",
        )
