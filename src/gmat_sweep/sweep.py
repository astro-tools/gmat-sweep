"""Sweep orchestrator: owns the run iterable, backend, manifest, and output dir.

The single public class is :class:`Sweep`. It binds a list of
:class:`gmat_sweep.spec.RunSpec` to a backend :class:`gmat_sweep.backends.base.Pool`,
fans the specs out, drains the resulting outcomes in completion order, and
records each one as a :class:`gmat_sweep.manifest.ManifestEntry` with an
fsynced append so a mid-sweep ``Ctrl-C`` leaves a parseable manifest on disk.

The class does **not** own the pool's lifecycle — wrap the supplied
:class:`Pool` in a ``with`` block at the call site (or call ``close()``)
so worker processes are cleaned up. The thin :func:`gmat_sweep.api.sweep`
wrapper takes care of this for the common case.
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from gmat_sweep.aggregate import (
    lazy_contacts,
    lazy_ephemerides,
    lazy_fused_reports,
    lazy_multiindex,
)
from gmat_sweep.errors import BackendError, SweepConfigError
from gmat_sweep.manifest import Manifest, ManifestEntry, canonical_script_sha256

if TYPE_CHECKING:
    import pandas as pd

    from gmat_sweep.backends.base import Pool
    from gmat_sweep.spec import RunOutcome, RunSpec

__all__ = ["Sweep"]

# Per-run worker log file name. The worker (gmat_sweep.worker.run_one) writes
# this file under each spec's output_dir; the manifest entry records the path
# so a downstream "show me the log for failed run N" lookup is one join away.
_WORKER_LOG_NAME = "worker.log"


class Sweep:
    """Bind run specs, a pool, and a manifest path into a runnable orchestrator.

    Parameters
    ----------
    runs:
        The :class:`RunSpec` instances to dispatch. ``run_id`` values must be
        unique. Order is preserved on the submission side; outcomes return in
        completion order.
    backend:
        A constructed :class:`Pool`. The caller owns its lifecycle — typically
        a ``with LocalJoblibPool(...) as pool:`` block.
    manifest_path:
        Where the JSON Lines manifest will be written. Parent directories are
        created on :meth:`run`.
    output_dir:
        Sweep root the per-run output directories live under. Used as the
        anchor for any relative paths the manifest records.
    script_path:
        The ``.script`` every run loads. Hashed via
        :func:`canonical_script_sha256` for the manifest header.
    parameter_spec:
        The original sweep parameterisation (e.g. the materialised grid) —
        recorded verbatim in the manifest header for reproducibility.
    sweep_seed:
        Optional integer seed recorded on the manifest. ``Sweep`` does not
        consume it directly; the Monte Carlo and Latin hypercube wrappers
        in :mod:`gmat_sweep.api` use it to derive their per-run draws.
    progress:
        ``True`` (default) wraps the drain loop in a :mod:`tqdm` bar.
        Set to ``False`` for non-interactive use (tests, CI logs).
    allow_unisolated_pool:
        Acknowledgement flag for backends whose ``subprocess_isolated`` is
        not :data:`True` (today: only
        :class:`gmat_sweep.backends.debug.DebugPool` with the ``"debug"``
        sentinel). Defaults to :data:`False`, in which case constructing a
        :class:`Sweep` over an unisolated backend raises
        :class:`gmat_sweep.errors.BackendError`. Pass :data:`True` together
        with the matching flag on the pool to opt in to in-process,
        single-run debug dispatch.
    """

    def __init__(
        self,
        *,
        runs: Sequence[RunSpec],
        backend: Pool,
        manifest_path: Path,
        output_dir: Path,
        script_path: Path,
        parameter_spec: Mapping[str, Any],
        sweep_seed: int | None = None,
        progress: bool = True,
        allow_unisolated_pool: bool = False,
    ) -> None:
        if backend.subprocess_isolated is not True and not allow_unisolated_pool:
            raise BackendError(
                f"backend {type(backend).__name__} declares "
                f"subprocess_isolated={backend.subprocess_isolated!r} (not True); "
                "pass allow_unisolated_pool=True to acknowledge in-process or "
                "otherwise unisolated dispatch."
            )
        self._runs: list[RunSpec] = list(runs)
        self._backend = backend
        self._manifest_path = Path(manifest_path)
        self._output_dir = Path(output_dir)
        self._script_path = Path(script_path)
        self._parameter_spec: dict[str, Any] = dict(parameter_spec)
        self._sweep_seed = sweep_seed
        self._progress = progress
        self._manifest: Manifest | None = None
        # Set by :meth:`from_manifest`. Gates :meth:`resume` so a freshly-
        # constructed Sweep can't accidentally append onto an unrelated file.
        self._loaded_from_manifest: bool = False

    def run(self) -> Sweep:
        """Submit every run, drain outcomes in completion order, return ``self``.

        Builds and saves the manifest header up front (one fsync, with the
        parent directory created on demand). For each completed
        :class:`RunOutcome` an entry is appended via
        :meth:`Manifest.append_entry`, which fsyncs each line — a ``Ctrl-C``
        between any two iterations leaves a parseable file containing exactly
        the runs that finished.

        :exc:`KeyboardInterrupt` is not caught; it propagates so the caller's
        ``with``-managed pool exits and cancels still-pending futures.
        """
        self._enforce_debug_pool_single_spec(self._runs)
        manifest = self._build_manifest()
        manifest.save(self._manifest_path)
        self._manifest = manifest

        specs_by_run_id: dict[int, RunSpec] = {s.run_id: s for s in self._runs}
        futures: list[Future[RunOutcome]] = [self._backend.submit(s) for s in self._runs]

        progress_bar = tqdm(
            total=len(self._runs),
            disable=not self._progress,
            desc="gmat-sweep",
            unit="run",
        )
        try:
            for outcome in self._backend.as_completed(futures):
                spec = specs_by_run_id[outcome.run_id]
                entry = ManifestEntry.from_outcome(
                    outcome,
                    overrides=spec.overrides,
                    log_path=spec.output_dir / _WORKER_LOG_NAME,
                )
                manifest.append_entry(entry)
                progress_bar.update(1)
        finally:
            progress_bar.close()

        return self

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        script_path: str | Path,
        *,
        backend: Pool,
        allow_script_drift: bool = False,
        progress: bool = True,
    ) -> Sweep:
        """Rebuild a :class:`Sweep` from a manifest written by a prior run.

        Reads ``manifest_path``, validates that the on-disk script still
        matches the manifest's recorded ``script_sha256``, reconstructs the
        run iterable from the manifest's ``parameter_spec``, and returns a
        :class:`Sweep` whose manifest is pre-bound to the loaded one. The
        returned sweep is suitable input to :meth:`resume`; calling
        :meth:`run` on it would re-execute every run from scratch and is
        not the intended flow.

        Parameters
        ----------
        manifest_path:
            Path to the existing ``manifest.jsonl``. Its parent is treated
            as the sweep's ``output_dir`` and must still exist on disk —
            successful runs' Parquet files are read from there as-is.
        script_path:
            Path to the GMAT ``.script`` to load. The file's canonical
            SHA-256 must equal the manifest's ``script_sha256`` unless
            ``allow_script_drift`` is set.
        backend:
            A constructed :class:`Pool`. The caller owns its lifecycle —
            same contract as the regular constructor.
        allow_script_drift:
            ``False`` (default) raises :class:`SweepConfigError` on a hash
            mismatch with both hashes in the message. ``True`` proceeds
            anyway and emits a :class:`RuntimeWarning`.
        progress:
            Forwarded to the constructor — controls the :mod:`tqdm` bar in
            :meth:`resume`.

        Raises
        ------
        SweepConfigError
            If ``manifest_path``'s parent directory does not exist, the
            script hash drifted and ``allow_script_drift`` is ``False``, or
            the manifest's ``parameter_spec`` carries an unknown ``_kind``.
        """
        manifest_path_obj = Path(manifest_path)
        script_path_obj = Path(script_path)
        output_dir = manifest_path_obj.parent
        if not output_dir.exists():
            raise SweepConfigError(
                f"manifest output directory does not exist: {output_dir} — "
                f"successful runs' Parquet files must still be on disk to be reused"
            )

        manifest = Manifest.load(manifest_path_obj)

        current_sha = canonical_script_sha256(script_path_obj)
        if current_sha != manifest.script_sha256:
            msg = (
                f"script hash mismatch for {script_path_obj}: "
                f"manifest={manifest.script_sha256}, current={current_sha}"
            )
            if not allow_script_drift:
                raise SweepConfigError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        runs = _build_runs_from_parameter_spec(
            manifest.parameter_spec,
            script_path=script_path_obj,
            output_dir=output_dir,
        )

        sweep = cls(
            runs=runs,
            backend=backend,
            manifest_path=manifest_path_obj,
            output_dir=output_dir,
            script_path=script_path_obj,
            parameter_spec=manifest.parameter_spec,
            sweep_seed=manifest.sweep_seed,
            progress=progress,
        )
        sweep._manifest = manifest
        sweep._loaded_from_manifest = True
        return sweep

    def resume(self) -> Sweep:
        """Re-run only the failed and missing runs from the loaded manifest.

        Submits specs for the union of ``manifest.find_failed()`` and ``manifest.find_missing(...)``
        through the bound backend, appends one new :class:`ManifestEntry`
        per outcome (with the same ``run_id`` as the original), and reloads
        the manifest so the in-memory ``entries`` reflect the last-wins
        merge. Returns ``self`` for chaining.

        Raises :exc:`RuntimeError` when called on a :class:`Sweep` not
        produced by :meth:`from_manifest`.
        """
        if not self._loaded_from_manifest or self._manifest is None:
            raise RuntimeError("Sweep.resume requires a Sweep built via Sweep.from_manifest")
        manifest = self._manifest

        expected_run_ids = [s.run_id for s in self._runs]
        to_retry: set[int] = set(manifest.find_failed()) | set(
            manifest.find_missing(expected_run_ids)
        )
        specs_by_run_id: dict[int, RunSpec] = {s.run_id: s for s in self._runs}
        runs_to_submit: list[RunSpec] = [specs_by_run_id[rid] for rid in sorted(to_retry)]
        self._enforce_debug_pool_single_spec(runs_to_submit)

        futures: list[Future[RunOutcome]] = [self._backend.submit(s) for s in runs_to_submit]

        progress_bar = tqdm(
            total=len(runs_to_submit),
            disable=not self._progress,
            desc="gmat-sweep resume",
            unit="run",
        )
        try:
            for outcome in self._backend.as_completed(futures):
                spec = specs_by_run_id[outcome.run_id]
                entry = ManifestEntry.from_outcome(
                    outcome,
                    overrides=spec.overrides,
                    log_path=spec.output_dir / _WORKER_LOG_NAME,
                )
                manifest.append_entry(entry)
                progress_bar.update(1)
        finally:
            progress_bar.close()

        # Reload so the in-memory entries list is deduplicated last-wins.
        # append_entry leaves duplicates in memory; load() folds them.
        self._manifest = Manifest.load(self._manifest_path)
        return self

    def to_manifest(self) -> Manifest:
        """Return the manifest populated by :meth:`run`."""
        if self._manifest is None:
            raise RuntimeError("Sweep.to_manifest requires Sweep.run() to have been called")
        return self._manifest

    def archive(
        self,
        out: str | Path,
        *,
        include_logs: bool = False,
    ) -> Path:
        """Pack the sweep — script, manifest, per-run Parquets — into a ``.zip``.

        The bundle is suitable for archival deposit (Zenodo, JOSS supplementary
        material) or internal handoff. Layout, path-rewrite rules, and the
        accompanying ``MANIFEST.hash`` are documented in
        :mod:`gmat_sweep.archive`.

        Parameters
        ----------
        out:
            Destination ``.zip`` path. Parent directories are created on demand.
        include_logs:
            When ``True``, every per-run ``worker.log`` is bundled and the
            manifest's ``log_path`` field continues to point at it (rewritten
            to bundle-relative form). The default ``False`` drops the logs and
            sets ``log_path`` to ``None`` in the bundled manifest, keeping the
            archive small.

        Returns
        -------
        Path
            The resolved path to the produced ``.zip``.
        """
        from gmat_sweep import __version__ as sweep_version
        from gmat_sweep.archive import _archive_sweep

        return _archive_sweep(
            manifest=self.to_manifest(),
            output_dir=self._output_dir,
            script_path=self._script_path,
            out=Path(out),
            include_logs=include_logs,
            sweep_version=sweep_version,
        )

    def to_dataframe(self, name: str | None = None) -> pd.DataFrame:
        """Aggregate the sweep's ``ReportFile`` outputs into one DataFrame.

        ``name`` selects which report to aggregate when the sweep produced
        multiple ``ReportFile`` resources per run; ``None`` (default) picks
        the sole report when exactly one was produced. See
        :func:`gmat_sweep.aggregate.lazy_multiindex` for the full contract.
        """
        return lazy_multiindex(self.to_manifest(), self._output_dir, name=name)

    def to_ephemerides(self, name: str | None = None) -> pd.DataFrame:
        """Aggregate the sweep's ``EphemerisFile`` outputs into one DataFrame.

        See :func:`gmat_sweep.aggregate.lazy_ephemerides` for the contract.
        """
        return lazy_ephemerides(self.to_manifest(), self._output_dir, name=name)

    def to_contacts(self, name: str | None = None) -> pd.DataFrame:
        """Aggregate the sweep's ``ContactLocator`` outputs into one DataFrame.

        See :func:`gmat_sweep.aggregate.lazy_contacts` for the contract.
        """
        return lazy_contacts(self.to_manifest(), self._output_dir, name=name)

    def to_fused_reports(
        self,
        names: Sequence[str],
        *,
        tolerance: str | pd.Timedelta,
        spool: bool = True,
    ) -> pd.DataFrame:
        """Fuse N ``ReportFile`` outputs per run into one wide MultiIndex-column DataFrame.

        See :func:`gmat_sweep.aggregate.lazy_fused_reports` for the
        contract — this is a thin convenience that binds the sweep's own
        manifest and output directory.
        """
        return lazy_fused_reports(
            self.to_manifest(), self._output_dir, names, tolerance=tolerance, spool=spool
        )

    def _repr_html_(self) -> str:
        import html as _html

        from gmat_sweep._repr_html import build_kv_table, short_sha

        kind = self._parameter_spec.get("_kind", "grid")
        try:
            sha = canonical_script_sha256(self._script_path)
            sha_cell = f"<code>{short_sha(sha)}</code>"
        except OSError:
            sha_cell = "(script not readable)"

        rows: list[tuple[str, str]] = [
            ("script", f"<code>{_html.escape(str(self._script_path))}</code>"),
            ("script_sha256", sha_cell),
            ("run_count", str(len(self._runs))),
            ("parameter_spec._kind", _html.escape(str(kind))),
            ("backend", _html.escape(type(self._backend).__name__)),
        ]
        if self._sweep_seed is not None:
            rows.append(("sweep_seed", str(self._sweep_seed)))

        if self._manifest is None:
            rows.append(("status", "<em>not yet executed</em>"))
        else:
            counts = {"ok": 0, "failed": 0, "skipped": 0}
            for entry in self._manifest.entries:
                counts[entry.status] += 1
            tally = ", ".join(f"{counts[k]} {k}" for k in ("ok", "failed", "skipped"))
            rows.append(("outcomes", _html.escape(tally)))

        return build_kv_table("Sweep", rows)

    def _enforce_debug_pool_single_spec(self, runs: Sequence[RunSpec]) -> None:
        # DebugPool runs every spec on the driver process and dirties GMAT's
        # process-global singletons after the first load; re-isolation
        # in-process is not implemented, so the pool refuses anything other
        # than exactly one spec. Other unisolated pools (none today) would
        # need their own enforcement once they appear.
        if self._backend.subprocess_isolated == "debug" and len(runs) != 1:
            raise BackendError(
                f"DebugPool dispatches a single spec in-process; got {len(runs)}. "
                "Use LocalJoblibPool or ProcessPoolExecutorPool for multi-spec sweeps."
            )

    def _build_manifest(self) -> Manifest:
        # Local import: gmat_sweep.__init__ sets __version__ as part of module
        # load, but importing it at module top level would create a cycle
        # (gmat_sweep imports Sweep). Resolved lazily on first run() call.
        from gmat_sweep import __version__ as sweep_version

        return Manifest(
            script_sha256=canonical_script_sha256(self._script_path),
            gmat_sweep_version=sweep_version,
            gmat_run_version=_gmat_run_version(),
            gmat_install_version=_gmat_install_version(),
            python_version=platform.python_version(),
            os_platform=platform.platform(),
            sweep_seed=self._sweep_seed,
            parameter_spec=self._parameter_spec,
            run_count=len(self._runs),
            backend=self._backend.__class__.__name__,
        )


def _build_runs_from_parameter_spec(
    parameter_spec: Mapping[str, Any],
    *,
    script_path: Path,
    output_dir: Path,
) -> list[RunSpec]:
    """Reconstruct the run iterable a manifest's ``parameter_spec`` describes.

    Dispatches on ``parameter_spec["_kind"]``: ``"grid"`` is the tagged
    grid shape current sweeps emit; ``None`` (no key) is the older
    untagged shape kept for backwards compatibility and dispatched the
    same way. The other three kinds round-trip through their matching
    expander. Resumed Monte Carlo and Latin hypercube runs draw bit-equal
    values to the original sweep because the expanders are deterministic
    in ``(perturb, n, seed)``.
    """
    # Local imports keep gmat_sweep.sweep cycle-free at import time —
    # gmat_sweep.grids depends on gmat_sweep.distributions, which pulls in
    # scipy, and we only pay that cost on resume.
    from gmat_sweep.distributions import _deserialise_perturb
    from gmat_sweep.grids import (
        expand_grid_to_run_specs,
        expand_latin_hypercube_to_run_specs,
        expand_monte_carlo_to_run_specs,
        expand_samples_to_run_specs,
    )

    kind = parameter_spec.get("_kind")
    if kind is None or kind == "grid":
        # Tagged and untagged grid manifests both carry the materialised
        # grid as flat top-level keys: {dotted-path: [values]}.
        grid = {k: v for k, v in parameter_spec.items() if k != "_kind"}
        return expand_grid_to_run_specs(grid, script_path, output_dir)
    if kind == "explicit":
        import pandas as pd

        samples = pd.DataFrame(parameter_spec["rows"], columns=list(parameter_spec["columns"]))
        return expand_samples_to_run_specs(samples, script_path, output_dir)
    if kind == "monte_carlo":
        perturb = _deserialise_perturb(parameter_spec["perturb"])
        seed = parameter_spec.get("seed")
        return expand_monte_carlo_to_run_specs(
            perturb,
            n=int(parameter_spec["n"]),
            seed=None if seed is None else int(seed),
            script_path=script_path,
            output_dir=output_dir,
        )
    if kind == "latin_hypercube":
        perturb = _deserialise_perturb(parameter_spec["perturb"])
        seed = parameter_spec.get("seed")
        return expand_latin_hypercube_to_run_specs(
            perturb,
            n=int(parameter_spec["n"]),
            seed=None if seed is None else int(seed),
            script_path=script_path,
            output_dir=output_dir,
        )
    raise SweepConfigError(
        f"unknown parameter_spec _kind: {kind!r} — "
        f"expected one of 'grid', 'explicit', 'monte_carlo', 'latin_hypercube' "
        f"(or absent, for older untagged grid manifests)"
    )


def _gmat_run_version() -> str:
    """Return ``gmat_run.__version__`` if importable, else ``"unknown"``.

    Importing :mod:`gmat_run` does not bootstrap ``gmatpy`` (the heavy SWIG
    bring-up happens inside :meth:`gmat_run.Mission.load`), so this is safe to
    call from the driver process.
    """
    try:
        import gmat_run
    except ImportError:
        return "unknown"
    return str(getattr(gmat_run, "__version__", "unknown"))


def _gmat_install_version() -> str:
    """Return the resolved GMAT install version, or ``"unknown"`` on any failure.

    :func:`gmat_run.install.locate_gmat` walks the filesystem and reads version
    files — it does not bootstrap ``gmatpy`` and so is cheap from the driver.
    Any failure (gmat-run missing, no install discoverable, version file
    unreadable) maps to ``"unknown"`` so the manifest header is always built.
    """
    try:
        from gmat_run.install import locate_gmat

        info = locate_gmat()
    except Exception:
        return "unknown"
    return info.version or "unknown"
