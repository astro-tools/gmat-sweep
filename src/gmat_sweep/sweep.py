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
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from gmat_sweep.aggregate import lazy_contacts, lazy_ephemerides, lazy_multiindex
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
    ) -> None:
        self._runs: list[RunSpec] = list(runs)
        self._backend = backend
        self._manifest_path = Path(manifest_path)
        self._output_dir = Path(output_dir)
        self._script_path = Path(script_path)
        self._parameter_spec: dict[str, Any] = dict(parameter_spec)
        self._sweep_seed = sweep_seed
        self._progress = progress
        self._manifest: Manifest | None = None

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

    def to_manifest(self) -> Manifest:
        """Return the manifest populated by :meth:`run`."""
        if self._manifest is None:
            raise RuntimeError("Sweep.to_manifest requires Sweep.run() to have been called")
        return self._manifest

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
