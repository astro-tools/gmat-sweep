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
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
    writes each ``ReportFile`` output as a Parquet file under
    ``spec.output_dir``. Returns :meth:`RunOutcome.ok` on success.

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
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = spec.output_dir / _WORKER_LOG_NAME

    logger = logging.getLogger(f"gmat_sweep.worker.run_{spec.run_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    try:
        logger.info("run_id=%d script=%s", spec.run_id, spec.script_path)
        logger.info("overrides=%s", spec.overrides)
        if spec.seed is not None:
            logger.info("seed=%d (reserved for v0.2 Monte Carlo)", spec.seed)

        # Lazy import: keeps gmatpy out of the driver process. The first call
        # in any given worker subprocess pays the bootstrap cost; subsequent
        # calls in the same subprocess hit the module cache.
        import gmat_run

        mission = gmat_run.Mission.load(spec.script_path)
        for key, value in spec.overrides.items():
            mission[key] = value
        results = mission.run(working_dir=spec.output_dir, **spec.run_options)

        output_paths = {}
        for name, df in results.reports.items():
            parquet_path = spec.output_dir / f"{name}.parquet"
            df.to_parquet(parquet_path)
            output_paths[name] = parquet_path
            logger.info("wrote report %s -> %s (%d rows)", name, parquet_path, len(df))

        if results.log:
            logger.info("--- GMAT engine log ---\n%s", results.log)

        ended_at = datetime.now(timezone.utc)
        logger.info("status=ok duration_s=%.3f", (ended_at - started_at).total_seconds())
        return RunOutcome.ok(
            run_id=spec.run_id,
            output_paths=output_paths,
            started_at=started_at,
            ended_at=ended_at,
        )
    except KeyboardInterrupt:  # pragma: no cover - propagates to the driver
        raise
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        tb = traceback.format_exc()
        engine_log = getattr(exc, "log", None)
        stderr = tb if not engine_log else f"{tb}\n--- GMAT engine log ---\n{engine_log}"
        logger.error("status=failed\n%s", stderr)
        return RunOutcome.failed(
            run_id=spec.run_id,
            stderr=stderr,
            started_at=started_at,
            ended_at=ended_at,
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
