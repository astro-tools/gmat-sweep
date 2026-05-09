"""Run/sweep specs and outcomes — JSON-serialisable units of work crossing the worker boundary.

These three dataclasses are the only objects that travel between the
driver process and worker subprocesses. They are
:func:`dataclasses.dataclass` ``frozen=True, slots=True`` so they cannot
accidentally mutate after handoff; ``frozen`` does not deep-freeze the
dict-typed fields, but it pins the field references and the serialised
shape.

Each class exposes a paired :meth:`to_dict` / :meth:`from_dict` for JSON
encoding. :class:`pathlib.Path` is coerced to ``str``; :class:`datetime`
to ISO-8601 via :meth:`datetime.isoformat`. Values inside ``overrides``,
``run_options``, ``backend_kwargs``, and ``output_paths`` must already be
JSON-encodable (no numpy scalars) — gmat-run handles numpy on the
:meth:`Mission.__setitem__` write side, but the spec is the
serialisation boundary.

``json.dumps(spec.to_dict(), sort_keys=True)`` is the canonical
"bit-equal" comparator across a serialise → deserialise → serialise
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

__all__ = ["RunOutcome", "RunSpec", "RunStatus", "SweepSpec"]


RunStatus = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """A single run's worth of work — script + overrides + run_id + seed.

    A worker reconstructs the full run from this record alone:
    instantiate :class:`gmat_run.Mission` from ``script_path``, apply
    ``overrides`` via the dotted-path setter, run with ``run_options``,
    write outputs under ``output_dir``.
    """

    script_path: Path
    overrides: dict[str, Any]
    output_dir: Path
    run_id: int
    seed: int | None
    run_options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_path": str(self.script_path),
            "overrides": dict(self.overrides),
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "seed": self.seed,
            "run_options": dict(self.run_options),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        return cls(
            script_path=Path(data["script_path"]),
            overrides=dict(data["overrides"]),
            output_dir=Path(data["output_dir"]),
            run_id=int(data["run_id"]),
            seed=None if data["seed"] is None else int(data["seed"]),
            run_options=dict(data["run_options"]),
        )


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """A whole sweep's metadata — script, runs, backend, outputs.

    ``runs`` is a materialised :class:`tuple` of :class:`RunSpec` so the
    spec round-trips through JSON cleanly. ``run_id`` ordering is the
    contract the manifest and resume flow depend on: ``runs[i].run_id ==
    i`` for every well-formed sweep.
    """

    mission_script_path: Path
    runs: tuple[RunSpec, ...]
    backend: str
    backend_kwargs: dict[str, Any]
    output_dir: Path
    manifest_path: Path
    sweep_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_script_path": str(self.mission_script_path),
            "runs": [r.to_dict() for r in self.runs],
            "backend": self.backend,
            "backend_kwargs": dict(self.backend_kwargs),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "sweep_seed": self.sweep_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepSpec:
        return cls(
            mission_script_path=Path(data["mission_script_path"]),
            runs=tuple(RunSpec.from_dict(r) for r in data["runs"]),
            backend=str(data["backend"]),
            backend_kwargs=dict(data["backend_kwargs"]),
            output_dir=Path(data["output_dir"]),
            manifest_path=Path(data["manifest_path"]),
            sweep_seed=None if data["sweep_seed"] is None else int(data["sweep_seed"]),
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """The result of one run after the worker returns.

    ``output_paths`` maps a worker-chosen key (e.g. the parsed
    ReportFile resource name) to the on-disk Parquet artefact written
    under :attr:`RunSpec.output_dir`. Empty for failed and skipped runs.
    ``stderr`` is ``None`` for successful runs and the captured worker
    stderr / traceback string for failed runs. ``duration_s`` is computed
    by the :meth:`ok` / :meth:`failed` helpers from the bookend timestamps
    so the three values cannot disagree.
    """

    run_id: int
    status: RunStatus
    output_paths: dict[str, Path]
    duration_s: float
    stderr: str | None
    started_at: datetime
    ended_at: datetime

    @classmethod
    def ok(
        cls,
        *,
        run_id: int,
        output_paths: dict[str, Path],
        started_at: datetime,
        ended_at: datetime,
    ) -> RunOutcome:
        return cls(
            run_id=run_id,
            status="ok",
            output_paths=dict(output_paths),
            duration_s=(ended_at - started_at).total_seconds(),
            stderr=None,
            started_at=started_at,
            ended_at=ended_at,
        )

    @classmethod
    def failed(
        cls,
        *,
        run_id: int,
        stderr: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> RunOutcome:
        return cls(
            run_id=run_id,
            status="failed",
            output_paths={},
            duration_s=(ended_at - started_at).total_seconds(),
            stderr=stderr,
            started_at=started_at,
            ended_at=ended_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "output_paths": {k: str(v) for k, v in self.output_paths.items()},
            "duration_s": self.duration_s,
            "stderr": self.stderr,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunOutcome:
        return cls(
            run_id=int(data["run_id"]),
            status=cast(RunStatus, data["status"]),
            output_paths={k: Path(v) for k, v in data["output_paths"].items()},
            duration_s=float(data["duration_s"]),
            stderr=None if data["stderr"] is None else str(data["stderr"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]),
        )

    def _repr_html_(self) -> str:
        from gmat_sweep._repr_html import (
            build_kv_table,
            format_paths_html,
            summarise_stderr_html,
        )

        rows: list[tuple[str, str]] = [
            ("run_id", str(self.run_id)),
            ("status", self.status),
            ("duration", f"{self.duration_s:.2f} s"),
            ("started_at", self.started_at.isoformat()),
            ("ended_at", self.ended_at.isoformat()),
            ("output_paths", format_paths_html(self.output_paths)),
            ("stderr", summarise_stderr_html(self.stderr)),
        ]
        return build_kv_table(f"RunOutcome run_id={self.run_id}", rows)
