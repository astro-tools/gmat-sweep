"""Per-run worker: subprocess-fresh gmat_run import, override application, Parquet output.

The single public entry point is :func:`run_one`. It is the unit of work the
backend pool fans out: one :class:`gmat_sweep.spec.RunSpec` in, one
:class:`gmat_sweep.spec.RunOutcome` out, and *never* a raised exception. Every
failure mode — bootstrap failure, override rejection, GMAT engine error,
Parquet write failure — is caught and turned into
:meth:`RunOutcome.failed` carrying the captured traceback as ``stderr`` so a
single bad run does not abort the parent sweep.

``import gmat_run`` is deferred until inside :func:`run_one` so the driver
process never bootstraps ``gmatpy``. Each backend worker pays the bootstrap
cost exactly once on its first call, in its own subprocess.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from gmat_sweep.spec import RunOutcome

if TYPE_CHECKING:
    from gmat_sweep.spec import RunSpec

__all__ = ["run_one"]


_WORKER_LOG_NAME = "worker.log"


def run_one(spec: RunSpec) -> RunOutcome:
    """Run one mission described by ``spec`` and return a :class:`RunOutcome`.

    Loads the script via :class:`gmat_run.Mission`, applies every override in
    ``spec.overrides`` through the dotted-path setter, executes the mission
    with ``working_dir=spec.output_dir`` plus ``**spec.run_options``, and
    writes each ``ReportFile`` / ``EphemerisFile`` / ``ContactLocator``
    output as a Parquet file under ``spec.output_dir`` named
    ``<kind>__<resource_name>.parquet``. ``RunOutcome.output_paths`` is
    keyed on the same prefixed basename so the aggregator can dispatch by
    output kind. Returns :meth:`RunOutcome.ok` on success.

    Any exception raised inside the function (bootstrap failure, override
    rejection, ``GmatRunError``, Parquet write failure, …) is caught and
    converted to :meth:`RunOutcome.failed` with the formatted traceback as
    ``stderr``. ``KeyboardInterrupt`` is the one exception that still
    propagates so ``Ctrl-C`` reaches the driver.

    A per-run log file is written to ``spec.output_dir / "worker.log"`` for
    both successful and failed runs; the eventual manifest entry references
    it via :attr:`gmat_sweep.manifest.ManifestEntry.log_path`.
    """
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()

    # FileHandler init (and the mkdir that precedes it) can raise on a bad
    # output_dir (EROFS, ENOSPC, permission denied). Keep both inside the
    # try so the worker's "never raises" contract holds — a FileHandler
    # failure here is reported as RunOutcome.failed, not propagated.
    logger = logging.getLogger(f"gmat_sweep.worker.run_{spec.run_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler: logging.FileHandler | None = None
    try:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = spec.output_dir / _WORKER_LOG_NAME
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

        logger.info("run_id=%d script=%s", spec.run_id, spec.script_path)
        logger.info("overrides=%s", spec.overrides)
        if spec.seed is not None:
            logger.info("seed=%d", spec.seed)

        # Lazy import: keeps gmatpy out of the driver process. The first call
        # in any given worker subprocess pays the bootstrap cost; subsequent
        # calls in the same subprocess hit the module cache.
        import gmat_run

        mission = gmat_run.Mission.load(spec.script_path)
        for key, value in spec.overrides.items():
            mission[key] = value
        results = mission.run(working_dir=spec.output_dir, **spec.run_options)

        output_paths: dict[str, Path] = {}
        for name, report_df in results.reports.items():
            output_paths[f"report__{name}"] = _write_one(
                spec.output_dir,
                kind="report",
                name=name,
                df=_synthesize_time_column(report_df),
                logger=logger,
            )
        for name, eph_df in results.ephemerides.items():
            output_paths[f"ephemeris__{name}"] = _write_one(
                spec.output_dir,
                kind="ephemeris",
                name=name,
                df=_synthesize_time_column(eph_df),
                logger=logger,
            )
        for name, con_df in results.contacts.items():
            con_df = con_df.reset_index(drop=True).copy()
            con_df["interval_id"] = range(len(con_df))
            output_paths[f"contact__{name}"] = _write_one(
                spec.output_dir,
                kind="contact",
                name=name,
                df=con_df,
                logger=logger,
            )

        if results.log:
            logger.info("--- GMAT engine log ---\n%s", results.log)

        ended_at = datetime.now(timezone.utc)
        duration_s = time.monotonic() - start_monotonic
        logger.info("status=ok duration_s=%.3f", duration_s)
        return RunOutcome.ok(
            run_id=spec.run_id,
            output_paths=output_paths,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
        )
    except KeyboardInterrupt:  # pragma: no cover - propagates to the driver
        raise
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        duration_s = time.monotonic() - start_monotonic
        tb = traceback.format_exc()
        engine_log = getattr(exc, "log", None)
        stderr = tb if not engine_log else f"{tb}\n--- GMAT engine log ---\n{engine_log}"
        if handler is not None:
            logger.error("status=failed\n%s", stderr)
        return RunOutcome.failed(
            run_id=spec.run_id,
            stderr=stderr,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
        )
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()


def _write_one(
    output_dir: Path,
    *,
    kind: str,
    name: str,
    df: pd.DataFrame,
    logger: logging.Logger,
) -> Path:
    parquet_path = output_dir / f"{kind}__{name}.parquet"
    df.to_parquet(parquet_path)
    logger.info("wrote %s %s -> %s (%d rows)", kind, name, parquet_path, len(df))
    return parquet_path


def _synthesize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    # gmat-run names ReportFile columns after their GMAT field (e.g.
    # "Sat.UTCGregorian"); aggregate.lazy_multiindex needs a column literally
    # named "time" to build the (run_id, time) MultiIndex. Copy — don't rename
    # — the first datetime column so the user's original column names round-trip
    # untouched into the aggregated DataFrame.
    if "time" in df.columns:
        return df
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return cast(pd.DataFrame, df.assign(time=df[col]))
    return df
