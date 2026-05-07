"""Internal helper: run a RunSpec in a freshly-spawned Python interpreter.

The execution-backend layer's bridge to :mod:`gmat_sweep._run_subprocess`.
Every backend uses this helper when a pool is constructed with
``reuse_gmat_context=False`` — it is the per-task fresh-bootstrap path
described in :class:`gmat_sweep.backends.base.Pool`. ``LocalJoblibPool``,
``DaskPool``, and ``RayPool`` each route their submission through this
function in that mode; in the default ``reuse_gmat_context=True`` mode they
call :func:`gmat_sweep.worker.run_one` directly inside the worker process
and pay the gmatpy bootstrap cost only on a worker's first task.

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
    ``python -m gmat_sweep._run_subprocess --spec <spec> --outcome <out>``
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
                    "gmat_sweep._run_subprocess",
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
                stderr=f"_run_subprocess timed out after {timeout}s\n{captured}".strip(),
                started_at=started_at,
                ended_at=ended_at,
            )
        except OSError as exc:
            ended_at = datetime.now(timezone.utc)
            return RunOutcome.failed(
                run_id=spec.run_id,
                stderr=f"_run_subprocess could not be spawned: {exc}",
                started_at=started_at,
                ended_at=ended_at,
            )

        if result.returncode != 0:
            ended_at = datetime.now(timezone.utc)
            return RunOutcome.failed(
                run_id=spec.run_id,
                stderr=(
                    f"_run_subprocess exited with status {result.returncode}\n{result.stderr}"
                ).rstrip(),
                started_at=started_at,
                ended_at=ended_at,
            )

        return RunOutcome.from_dict(json.loads(outcome_path.read_text(encoding="utf-8")))
