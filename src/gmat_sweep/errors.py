"""Typed exception hierarchy so downstream code can branch on failure mode.

Every exception gmat-sweep raises inherits from :class:`GmatSweepError`. Leaf
classes carry payload relevant to their failure mode as named attributes
(:attr:`RunFailed.run_id`, :attr:`ManifestCorruptError.path`) rather than
only as the formatted message — pattern-matching downstream code can read
the attribute without re-parsing the string.

:class:`RunFailed` is mostly a typed sentinel for tests and for callers
that opt into raise-on-failure behaviour. The default worker contract
(see issue #6) converts per-run failures into a
:class:`gmat_sweep.spec.RunOutcome` with ``status="failed"`` so a single
bad run does not abort the parent sweep.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BackendError",
    "GmatSweepError",
    "ManifestCorruptError",
    "RunFailed",
    "SweepConfigError",
]


class GmatSweepError(Exception):
    """Base class for every exception raised by gmat-sweep."""


class SweepConfigError(GmatSweepError):
    """Raised when a sweep configuration is invalid before any run starts.

    Covers contradictory arguments, malformed grid specs, and dotted-path
    overrides that fail validation at sweep-construction time. The message
    is the only payload.
    """


class RunFailed(GmatSweepError):
    """Raised when a single run fails inside a worker (raise-on-failure mode).

    Per the gmat-sweep contract a failed run normally lands as a labelled
    row in the parent DataFrame rather than an exception, so this class
    is a typed sentinel for tests and for callers that opt into eager
    failure. The ``run_id`` attribute identifies the offending run;
    ``stderr`` carries the captured worker stderr or Python traceback.
    """

    def __init__(self, message: str, run_id: int, stderr: str | None = None) -> None:
        self.run_id = run_id
        self.stderr = stderr
        super().__init__(message)


class BackendError(GmatSweepError):
    """Raised when an execution backend itself fails.

    Covers worker-pool initialisation failures, lost workers, and backend
    implementations that violate the subprocess-isolation contract
    enforced by the :class:`Pool` ABC.
    """


class ManifestCorruptError(GmatSweepError):
    """Raised when a sweep manifest cannot be parsed.

    The ``path`` attribute points at the offending file so callers can
    surface it in error messages without re-deriving the path. The
    optional ``line_number`` attribute is the 1-indexed line in the file
    that failed to parse — set by :meth:`gmat_sweep.Manifest.load` when the
    failure is localised to one line, ``None`` for whole-file failures
    (e.g. an empty file).
    """

    def __init__(self, message: str, path: Path, line_number: int | None = None) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(message)
