"""Internal helper: run a RunSpec in a freshly-spawned Python interpreter.

The execution-backend layer's bridge to :mod:`gmat_sweep._worker_entrypoint`.
Backends that reuse worker processes (Dask, Ray) call
:func:`run_spec_in_subprocess` from inside each task to honour the per-run
fresh-interpreter contract enforced by
:class:`gmat_sweep.backends.base.Pool`. ``LocalJoblibPool`` does not — loky
already gives one fresh interpreter per task and gains nothing from the
extra hop.

Underscore-prefixed module name keeps this internal to the backends layer;
nothing here is re-exported from :mod:`gmat_sweep.backends`.

Failure semantics
-----------------

Run-level failures (script-not-found, override rejection, GMAT engine
error) round-trip through the entrypoint and arrive here as a JSON
``RunOutcome`` with ``status="failed"``. They are *not* transport
failures.

A non-zero subprocess exit, a :class:`subprocess.TimeoutExpired`, or an
:class:`OSError` from :func:`subprocess.run` is classified as a transport
failure: the captured stderr (if any) is folded into a synthetic
:meth:`RunOutcome.failed` so the caller still sees a well-formed outcome.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from gmat_sweep.spec import RunOutcome, RunSpec

__all__ = ["run_spec_in_subprocess"]


def run_spec_in_subprocess(
    spec: RunSpec,
    *,
    python: str | None = None,
    timeout: float | None = None,
) -> RunOutcome:
    """Run ``spec`` in a fresh Python interpreter and return its outcome.

    Writes ``spec.to_dict()`` to a temp file, invokes
    ``python -m gmat_sweep._worker_entrypoint --spec <spec> --outcome <out>``
    via :func:`subprocess.run`, reads the outcome JSON back, and returns
    the :class:`RunOutcome`.

    Parameters
    ----------
    spec:
        The :class:`RunSpec` to run.
    python:
        Python executable to spawn. Defaults to :data:`sys.executable`.
    timeout:
        Forwarded to :func:`subprocess.run`. ``None`` (the default) waits
        indefinitely.
    """
    interpreter = python or sys.executable
    started_at = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory(prefix="gmat-sweep-spec-") as td:
        spec_path = Path(td) / "spec.json"
        outcome_path = Path(td) / "outcome.json"
        spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    interpreter,
                    "-m",
                    "gmat_sweep._worker_entrypoint",
                    "--spec",
                    str(spec_path),
                    "--outcome",
                    str(outcome_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            ended_at = datetime.now(timezone.utc)
            captured = exc.stderr if isinstance(exc.stderr, str) else ""
            return RunOutcome.failed(
                run_id=spec.run_id,
                stderr=f"_worker_entrypoint timed out after {timeout}s\n{captured}".strip(),
                started_at=started_at,
                ended_at=ended_at,
            )
        except OSError as exc:
            ended_at = datetime.now(timezone.utc)
            return RunOutcome.failed(
                run_id=spec.run_id,
                stderr=f"_worker_entrypoint could not be spawned: {exc}",
                started_at=started_at,
                ended_at=ended_at,
            )

        if result.returncode != 0:
            ended_at = datetime.now(timezone.utc)
            return RunOutcome.failed(
                run_id=spec.run_id,
                stderr=(
                    f"_worker_entrypoint exited with status {result.returncode}\n{result.stderr}"
                ).rstrip(),
                started_at=started_at,
                ended_at=ended_at,
            )

        return RunOutcome.from_dict(json.loads(outcome_path.read_text(encoding="utf-8")))
