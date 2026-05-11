"""CLI module — ``python -m gmat_sweep._run_subprocess --spec PATH --outcome PATH``.

Runs one :class:`gmat_sweep.spec.RunSpec` (read from ``--spec`` as JSON)
through :func:`gmat_sweep.worker.run_one` and writes the resulting
:class:`gmat_sweep.spec.RunOutcome` (as JSON) to ``--outcome``. Internal
infrastructure for execution backends that reuse worker processes (Dask,
Ray): each task body spawns this module via
``subprocess.run([sys.executable, "-m", "gmat_sweep._run_subprocess", ...])``
to honour the per-run fresh-interpreter contract enforced by
:class:`gmat_sweep.backends.base.Pool`. ``LocalJoblibPool`` does not need
the hop — loky already gives one fresh interpreter per task.

Underscore-prefixed name marks the module internal; callers go through a
``Pool``, not this module directly. The
:func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess` helper
wraps the spawn / handoff / cleanup so backend implementations don't talk
to argv.

Exit-code contract
------------------

A successful round-trip exits ``0`` regardless of the run's own status: a
failed run is still a successful entrypoint invocation — the outcome JSON
carries ``status="failed"`` and the captured ``stderr``, and the
failure-as-row contract holds at the entrypoint boundary just as it does
inside :func:`gmat_sweep.worker.run_one`.

Non-zero exit codes signal *transport* failures — the entrypoint could
not produce an outcome JSON at all:

- ``2`` — bad CLI arguments (argparse default).
- ``3`` — unreadable ``--spec`` file (missing, permission denied, malformed
  JSON, or a payload that fails :meth:`gmat_sweep.spec.RunSpec.from_dict`).
- ``4`` — unwriteable ``--outcome`` file. Two flavours:
  fail-fast — the ``--outcome`` path is probed with a touch + unlink
  before ``run_one`` is invoked, so a bad path doesn't cost a 30-minute
  GMAT run; and post-run — if ``run_one`` completed but the outcome JSON
  fails to land on disk, the outcome is printed to stdout as a single
  JSON line before exiting ``4``. The parent
  (:func:`gmat_sweep.backends._subprocess.run_spec_in_subprocess`)
  recovers the outcome from stdout in that case.
- Anything else — the OS killed the process (signal, OOM, segfault). The
  parent classifies non-zero exits as transport failures and folds the
  captured stderr into a synthetic
  :meth:`gmat_sweep.spec.RunOutcome.failed`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gmat_sweep.spec import RunSpec
from gmat_sweep.worker import run_one

__all__ = ["main"]


_EXIT_BAD_SPEC = 3
_EXIT_BAD_OUTCOME = 4


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, read spec, run, write outcome — return the process exit code.

    Returns ``0`` on a successful round-trip regardless of the run's own
    status. Non-zero on transport failure per the module-level contract.
    """
    parser = argparse.ArgumentParser(
        prog="python -m gmat_sweep._run_subprocess",
        description="Run one RunSpec in a fresh interpreter; emit a RunOutcome.",
    )
    parser.add_argument("--spec", required=True, help="Path to a RunSpec JSON file")
    parser.add_argument("--outcome", required=True, help="Path to write the RunOutcome JSON")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    outcome_path = Path(args.outcome)

    try:
        spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = RunSpec.from_dict(spec_data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(
            f"_run_subprocess: cannot read spec {spec_path}: {exc}",
            file=sys.stderr,
        )
        return _EXIT_BAD_SPEC

    # Fail-fast on a bad --outcome path. A 30-minute GMAT run that then
    # discovers it cannot write the outcome JSON is the work-loss path
    # this probe closes; we'd rather find out now and exit 4 before
    # paying for run_one. The post-run write below still has its own
    # exception handling for races (disk fills mid-run, path made
    # read-only between probe and write, …).
    try:
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        outcome_path.touch()
        outcome_path.unlink()
    except OSError as exc:
        print(
            f"_run_subprocess: --outcome path is not writable {outcome_path}: {exc}",
            file=sys.stderr,
        )
        return _EXIT_BAD_OUTCOME

    outcome = run_one(spec)

    try:
        with outcome_path.open("w", encoding="utf-8") as f:
            json.dump(outcome.to_dict(), f)
    except OSError as exc:
        # Race after the writability probe (disk filled mid-run, fs
        # remounted read-only, …): emit the outcome on stdout so the
        # parent backend can recover it instead of synthesising a
        # failed outcome that drops the worker's actual run.
        print(json.dumps(outcome.to_dict()))
        print(
            f"_run_subprocess: cannot write outcome {outcome_path}: {exc}",
            file=sys.stderr,
        )
        return _EXIT_BAD_OUTCOME

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
