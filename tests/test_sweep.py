"""Tests for gmat_sweep.sweep — orchestrator wiring, manifest fsync, ctrl-c safety."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest, canonical_script_sha256
from gmat_sweep.spec import RunOutcome, RunSpec
from gmat_sweep.sweep import Sweep
from tests.conftest import FakeGmatRun, FakeResults


def _write_script(tmp_path: Path, name: str = "mission.script") -> Path:
    path = tmp_path / name
    path.write_text("% GMAT script\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _make_runs(script: Path, output_dir: Path, n: int) -> list[RunSpec]:
    return [
        RunSpec(
            script_path=script,
            overrides={"Sat.SMA": 7000.0 + i},
            output_dir=output_dir / f"run-{i}",
            run_id=i,
            seed=None,
            run_options={},
        )
        for i in range(n)
    ]


def _payload_run_hook(rows: int = 1) -> Any:
    payload = pd.DataFrame(
        {
            "time": pd.to_datetime([f"2026-05-04T00:00:0{i}" for i in range(rows)]),
            "x": [float(i) for i in range(rows)],
        }
    )

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": payload})

    return _run


# ---- run() basics ---------------------------------------------------------


def test_sweep_run_returns_self(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0]},
            progress=False,
        )
        assert sweep.run() is sweep


def test_sweep_run_writes_one_manifest_entry_per_run(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=4)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0, 7001.0, 7002.0, 7003.0]},
            progress=False,
        ).run()

    manifest = Manifest.load(output_dir / "manifest.jsonl")
    assert manifest.run_count == 4
    assert len(manifest.entries) == 4
    assert {e.run_id for e in manifest.entries} == {0, 1, 2, 3}
    by_run_id = {e.run_id: e for e in manifest.entries}
    for run_id in range(4):
        entry = by_run_id[run_id]
        assert entry.status == "ok"
        assert entry.overrides == {"Sat.SMA": 7000.0 + run_id}
        assert entry.log_path == output_dir / f"run-{run_id}" / "worker.log"


def test_sweep_manifest_header_carries_script_hash_seed_and_spec(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path, name="m.script")
    expected_sha = canonical_script_sha256(script)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0]},
            sweep_seed=1729,
            progress=False,
        ).run()

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert reloaded.script_sha256 == expected_sha
    assert reloaded.sweep_seed == 1729
    assert reloaded.parameter_spec == {"Sat.SMA": [7000.0]}
    assert reloaded.run_count == 1
    # gmat_sweep_version is the package version pulled at import time —
    # asserting it equals the live module value is the cheapest way to confirm
    # it isn't a stale string literal in sweep.py.
    import gmat_sweep

    assert reloaded.gmat_sweep_version == gmat_sweep.__version__


class _NamedSpecificPool(Pool):
    """A no-IO Pool subclass with a distinct class name.

    Used by the manifest-backend-field test to confirm
    :meth:`Sweep._build_manifest` records ``pool.__class__.__name__`` rather
    than e.g. a hard-coded ``"LocalJoblibPool"`` literal or the abstract
    ``"Pool"``.
    """

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        future: Future[RunOutcome] = Future()
        future.spec = spec  # type: ignore[attr-defined]
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        now = datetime.now(timezone.utc)
        for f in futures:
            spec: RunSpec = f.spec  # type: ignore[attr-defined]
            outcome = RunOutcome.ok(
                run_id=spec.run_id,
                output_paths={},
                started_at=now,
                ended_at=now,
            )
            f.set_result(outcome)
            yield outcome

    def close(self) -> None:
        pass


def test_sweep_manifest_records_pool_class_name_as_backend(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """:meth:`Sweep._build_manifest` records the pool's ``__class__.__name__``
    as the manifest's ``backend`` field. The contract every cross-backend
    consumer (the equivalence suite, the CLI's ``show`` output) relies on.
    """
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=2)

    pool = _NamedSpecificPool()
    Sweep(
        runs=runs,
        backend=pool,
        manifest_path=output_dir / "manifest.jsonl",
        output_dir=output_dir,
        script_path=script,
        parameter_spec={"Sat.SMA": [7000.0, 7001.0]},
        progress=False,
    ).run()

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert reloaded.backend == "_NamedSpecificPool"


def test_sweep_manifest_parent_directory_is_created(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "deeply" / "nested" / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    assert (output_dir / "manifest.jsonl").exists()


# ---- aggregation ---------------------------------------------------------


def test_sweep_to_dataframe_returns_multiindexed_frame(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=3)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=2))

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0, 7001.0, 7002.0]},
            progress=False,
        ).run()

    df = sweep.to_dataframe()
    assert df.index.names == ["run_id", "time"]
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2]
    assert (df["__status"] == "ok").all()


def test_sweep_to_dataframe_marks_failed_run(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=3)

    def _setitem(_key: str, value: Any) -> None:
        if value == 7001.0:
            raise ValueError("rejected by GMAT")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook(rows=1))

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    df = sweep.to_dataframe()
    assert set(df["__status"].unique()) == {"ok", "failed"}
    failed_rows = df.loc[df["__status"] == "failed"]
    assert len(failed_rows) == 1
    assert failed_rows.index.get_level_values("run_id").tolist() == [1]


def test_sweep_to_manifest_requires_run_first(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        )
        with pytest.raises(RuntimeError, match="run"):
            sweep.to_manifest()


# ---- progress -------------------------------------------------------------


def test_sweep_progress_disabled_quiet_on_stderr(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=2)

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={},
            progress=False,
        ).run()

    captured = capsys.readouterr()
    # tqdm draws a progress bar by writing carriage returns and percentages
    # to stderr; with progress=False neither should appear.
    assert "gmat-sweep" not in captured.err
    assert "%" not in captured.err


# ---- ctrl-c safety --------------------------------------------------------


class _InterruptingPool(Pool):
    """Pool that yields N outcomes successfully, then raises KeyboardInterrupt.

    Used to drive the partial-manifest assertion without standing up a real
    subprocess pool — the actual loky path is exercised separately in
    test_backends_joblib.
    """

    def __init__(self, *, yield_count: int) -> None:
        self._submitted: list[RunSpec] = []
        self._yield_count = yield_count

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        self._submitted.append(spec)
        return Future()

    def as_completed(self, _futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        now = datetime.now(timezone.utc)
        for spec in self._submitted[: self._yield_count]:
            yield RunOutcome.ok(
                run_id=spec.run_id,
                output_paths={},
                started_at=now,
                ended_at=now,
            )
        raise KeyboardInterrupt

    def close(self) -> None:
        pass


def test_sweep_keyboard_interrupt_leaves_parsable_partial_manifest(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=4)

    pool = _InterruptingPool(yield_count=2)
    sweep = Sweep(
        runs=runs,
        backend=pool,
        manifest_path=output_dir / "manifest.jsonl",
        output_dir=output_dir,
        script_path=script,
        parameter_spec={},
        progress=False,
    )
    with pytest.raises(KeyboardInterrupt):
        sweep.run()

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert len(reloaded.entries) == 2
    assert {e.run_id for e in reloaded.entries} == {0, 1}


# ---- from_manifest --------------------------------------------------------


def _build_grid_manifest(
    tmp_path: Path,
    fake_gmat_run: FakeGmatRun,
    *,
    n: int = 4,
    failing_value: float | None = None,
) -> tuple[Path, Path]:
    """Run a grid sweep, optionally failing on a specific override value.

    Returns ``(script_path, output_dir)`` so a follow-up resume can point at
    the same script and manifest.
    """
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=n)

    if failing_value is not None:

        def _setitem(_key: str, value: Any) -> None:
            if value == failing_value:
                raise ValueError("rejected by GMAT")

        fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook(rows=1))
    else:
        fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))

    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0 + i for i in range(n)]},
            progress=False,
        ).run()

    return script, output_dir


def test_from_manifest_grid_rebuilds_runs(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """Untagged grid manifests (no ``_kind`` key) keep loading —
    ``_build_runs_from_parameter_spec`` treats a missing ``_kind`` as
    ``"grid"`` for backwards compatibility."""
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=4)
    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))

    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep.from_manifest(
            output_dir / "manifest.jsonl",
            script,
            backend=pool,
            progress=False,
        )

    assert [s.run_id for s in sweep._runs] == [0, 1, 2, 3]
    assert [s.overrides for s in sweep._runs] == [{"Sat.SMA": 7000.0 + i} for i in range(4)]


def test_from_manifest_tagged_grid_kind_rebuilds_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """A grid manifest written by ``sweep(grid=...)`` carries
    ``_kind: "grid"``; ``from_manifest`` dispatches it to the same
    expander as the untagged shape."""
    from gmat_sweep import sweep as sweep_api

    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    sweep_api(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0]},
        backend=LocalJoblibPool(workers=1),
        out=output_dir,
    )

    # Sanity-check the discriminator made it onto the manifest.
    from gmat_sweep import Manifest as _Manifest

    saved = _Manifest.load(output_dir / "manifest.jsonl")
    assert saved.parameter_spec["_kind"] == "grid"

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        rebuilt = Sweep.from_manifest(
            output_dir / "manifest.jsonl",
            script,
            backend=pool,
            progress=False,
        )

    assert [s.run_id for s in rebuilt._runs] == [0, 1, 2]
    assert [s.overrides for s in rebuilt._runs] == [
        {"Sat.SMA": 7000.0},
        {"Sat.SMA": 7100.0},
        {"Sat.SMA": 7200.0},
    ]


def test_from_manifest_explicit_kind_rebuilds_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """An explicit-row sweep round-trips through from_manifest's _kind dispatch."""
    from gmat_sweep import sweep as sweep_api

    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    samples = pd.DataFrame({"Sat.SMA": [7000.0, 7100.0, 7200.0], "Sat.ECC": [0.001, 0.002, 0.003]})

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    sweep_api(script, samples=samples, backend=LocalJoblibPool(workers=1), out=output_dir)

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        rebuilt = Sweep.from_manifest(
            output_dir / "manifest.jsonl",
            script,
            backend=pool,
            progress=False,
        )

    assert [s.overrides for s in rebuilt._runs] == [
        {"Sat.SMA": 7000.0, "Sat.ECC": 0.001},
        {"Sat.SMA": 7100.0, "Sat.ECC": 0.002},
        {"Sat.SMA": 7200.0, "Sat.ECC": 0.003},
    ]


def test_from_manifest_monte_carlo_rebuilds_bit_equal_overrides(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The acceptance criterion: a resumed Monte Carlo sweep produces
    bit-equal draws because expand_monte_carlo_to_run_specs is deterministic
    in (perturb, n, seed)."""
    from gmat_sweep import monte_carlo

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    monte_carlo(
        script,
        n=8,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0), "Sat.INC": ("uniform", 0.0, 90.0)},
        seed=1729,
        backend=LocalJoblibPool(workers=1),
        out=out,
    )

    original_overrides = {
        e.run_id: e.overrides for e in Manifest.load(out / "manifest.jsonl").entries
    }

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        rebuilt = Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False)

    rebuilt_overrides = {s.run_id: s.overrides for s in rebuilt._runs}
    assert rebuilt_overrides == original_overrides


def test_from_manifest_latin_hypercube_rebuilds_bit_equal_overrides(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    from gmat_sweep import latin_hypercube

    script = _write_script(tmp_path)
    out = tmp_path / "out"
    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    latin_hypercube(
        script,
        n=8,
        perturb={"Sat.SMA": ("normal", 7100.0, 50.0), "Sat.INC": ("uniform", 0.0, 90.0)},
        seed=1729,
        backend=LocalJoblibPool(workers=1),
        out=out,
    )

    original = {e.run_id: e.overrides for e in Manifest.load(out / "manifest.jsonl").entries}

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        rebuilt = Sweep.from_manifest(out / "manifest.jsonl", script, backend=pool, progress=False)

    assert {s.run_id: s.overrides for s in rebuilt._runs} == original


def test_from_manifest_script_drift_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=2)
    # Mutate the script: same path, different bytes ⇒ different canonical hash.
    script.write_text("% GMAT script v2\nCreate Spacecraft Sat;\n", encoding="utf-8")

    with LocalJoblibPool(workers=1) as pool:  # noqa: SIM117 — context manager scope is intentional
        with pytest.raises(SweepConfigError, match="script hash mismatch"):
            Sweep.from_manifest(output_dir / "manifest.jsonl", script, backend=pool, progress=False)


def test_from_manifest_script_drift_allowed_warns_and_proceeds(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=2)
    script.write_text("% GMAT script v2\nCreate Spacecraft Sat;\n", encoding="utf-8")

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with (
        LocalJoblibPool(workers=1) as pool,
        pytest.warns(RuntimeWarning, match="script hash mismatch"),
    ):
        sweep = Sweep.from_manifest(
            output_dir / "manifest.jsonl",
            script,
            backend=pool,
            allow_script_drift=True,
            progress=False,
        )
    assert len(sweep._runs) == 2


def test_from_manifest_missing_output_dir_raises(tmp_path: Path) -> None:
    """If the parent of manifest_path no longer exists, succeeded runs'
    Parquet files are gone — raise rather than silently produce a sweep
    that would re-execute everything."""
    fake_path = tmp_path / "vanished" / "manifest.jsonl"
    with LocalJoblibPool(workers=1) as pool:  # noqa: SIM117
        with pytest.raises(SweepConfigError, match="output directory does not exist"):
            Sweep.from_manifest(
                fake_path, tmp_path / "mission.script", backend=pool, progress=False
            )


def test_from_manifest_unknown_kind_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=1)
    # Hand-corrupt the header to introduce an unknown _kind.
    manifest_path = output_dir / "manifest.jsonl"
    raw = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    import json as _json

    header = _json.loads(raw[0])
    header["parameter_spec"] = {"_kind": "halton", "n": 1}
    raw[0] = _json.dumps(header, sort_keys=True) + "\n"
    manifest_path.write_text("".join(raw), encoding="utf-8")

    with LocalJoblibPool(workers=1) as pool:  # noqa: SIM117
        with pytest.raises(SweepConfigError, match="unknown parameter_spec _kind"):
            Sweep.from_manifest(manifest_path, script, backend=pool, progress=False)


# ---- resume ---------------------------------------------------------------


def test_resume_requires_from_manifest(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    """A sweep produced by the regular constructor cannot resume — the
    manifest header on disk hasn't been validated against this Sweep's
    parameters, so appending would silently mix two unrelated runs."""
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    runs = _make_runs(script, output_dir, n=1)
    with LocalJoblibPool(workers=1) as pool:
        sweep = Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0]},
            progress=False,
        )
        with pytest.raises(RuntimeError, match="from_manifest"):
            sweep.resume()


def test_resume_only_runs_failed_and_missing_run_ids(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """After a sweep with one failed run, resume should re-submit only that
    run_id (plus any never-recorded run_ids)."""
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=4, failing_value=7001.0)

    # Record which overrides the resumed pass passes through __setitem__.
    second_pass_overrides: list[Any] = []

    def _setitem(_key: str, value: Any) -> None:
        second_pass_overrides.append(value)

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        Sweep.from_manifest(
            output_dir / "manifest.jsonl", script, backend=pool, progress=False
        ).resume()

    # Only the originally-failed run (Sat.SMA=7001.0) was re-attempted.
    assert second_pass_overrides == [7001.0]

    reloaded = Manifest.load(output_dir / "manifest.jsonl")
    assert {e.run_id for e in reloaded.entries} == {0, 1, 2, 3}
    assert all(e.status == "ok" for e in reloaded.entries)


def test_resume_acceptance_16_runs_with_3_failures(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The headline acceptance criterion: 16-run sweep with 3 deliberately
    failing runs, resumed with the override-rejection patched out, lands a
    DataFrame with run_id cardinality 16 and __status=='ok' for all 16."""
    script = _write_script(tmp_path)
    output_dir = tmp_path / "out"
    n = 16
    runs = _make_runs(script, output_dir, n=n)
    failing = {7003.0, 7007.0, 7011.0}

    def _setitem(_key: str, value: Any) -> None:
        if value in failing:
            raise ValueError("rejected by GMAT")

    fake_gmat_run.install_loader(setitem_hook=_setitem, run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        Sweep(
            runs=runs,
            backend=pool,
            manifest_path=output_dir / "manifest.jsonl",
            output_dir=output_dir,
            script_path=script,
            parameter_spec={"Sat.SMA": [7000.0 + i for i in range(n)]},
            progress=False,
        ).run()

    # Sanity: 3 failed in the first pass.
    first_pass = Manifest.load(output_dir / "manifest.jsonl")
    assert sorted(first_pass.find_failed()) == [3, 7, 11]

    # Patch the override out, resume.
    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        df = (
            Sweep.from_manifest(output_dir / "manifest.jsonl", script, backend=pool, progress=False)
            .resume()
            .to_dataframe()
        )

    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == list(range(n))
    assert (df["__status"] == "ok").all()


def test_resume_writes_duplicate_lines_but_load_dedups(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The append-only file gains a second line for the resumed run_id; the
    in-memory entries list (after load) carries only the resumed entry."""
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=3, failing_value=7001.0)
    manifest_path = output_dir / "manifest.jsonl"
    raw_lines_before = manifest_path.read_text(encoding="utf-8").splitlines()

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        Sweep.from_manifest(manifest_path, script, backend=pool, progress=False).resume()

    raw_lines_after = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines_after) == len(raw_lines_before) + 1  # one new entry line

    reloaded = Manifest.load(manifest_path)
    by_run_id = {e.run_id: e for e in reloaded.entries}
    assert by_run_id[1].status == "ok"
    assert sum(1 for e in reloaded.entries if e.run_id == 1) == 1


def test_resume_with_no_failures_or_missing_is_a_noop(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """All runs already ok ⇒ resume submits nothing and the manifest is
    unchanged byte-for-byte."""
    script, output_dir = _build_grid_manifest(tmp_path, fake_gmat_run, n=3)
    manifest_path = output_dir / "manifest.jsonl"
    bytes_before = manifest_path.read_bytes()

    fake_gmat_run.install_loader(run_hook=_payload_run_hook(rows=1))
    with LocalJoblibPool(workers=1) as pool:
        Sweep.from_manifest(manifest_path, script, backend=pool, progress=False).resume()

    assert manifest_path.read_bytes() == bytes_before
