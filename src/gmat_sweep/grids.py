"""Full-factorial, explicit-row, Latin hypercube, and quasi-Monte-Carlo run generators.

v0.1 ships the full-factorial generator only; explicit-row and Latin hypercube
land in v0.2.

The output of :func:`full_factorial` is byte-for-byte deterministic: keys are
emitted in sorted (lexicographic) order and combinations enumerate in
:func:`itertools.product` order over the materialised input iterables. This is
the determinism guarantee the manifest and resume flow rely on — two runs
serialised through JSON with ``sort_keys=True`` produce identical bytes,
across processes and across machines.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import product
from pathlib import Path
from typing import Any

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.spec import RunSpec

__all__ = ["expand_grid_to_run_specs", "full_factorial"]


def full_factorial(grid: Mapping[str, Iterable[Any]]) -> Iterator[dict[str, Any]]:
    """Yield override dicts for the cartesian product of ``grid``.

    Keys are emitted in lexicographic order; the cartesian product enumerates
    in :func:`itertools.product` order over the input iterables, so the outer
    loop varies the lexicographically-first key slowest and the last key
    fastest. For ``{"a": [1, 2], "b": [10, 20, 30]}`` the six dicts come out as
    ``(a=1, b=10), (a=1, b=20), (a=1, b=30), (a=2, b=10), (a=2, b=20),
    (a=2, b=30)``.

    Each input iterable is materialised once at entry so callers may pass
    generators without surprising exhaustion and so empty-iterable validation
    can run before the cartesian product begins.

    An empty mapping is valid and yields a single empty override dict — the
    cartesian product of nothing has one element.

    Raises :class:`SweepConfigError` if any key is not a :class:`str` or any
    value materialises to an empty sequence.
    """
    materialised: dict[str, tuple[Any, ...]] = {}
    for key, values in grid.items():
        if not isinstance(key, str):
            raise SweepConfigError(f"grid keys must be strings, got {type(key).__name__}: {key!r}")
        materialised[key] = tuple(values)
        if not materialised[key]:
            raise SweepConfigError(f"grid value for {key!r} is empty")

    sorted_keys = sorted(materialised)
    for combo in product(*(materialised[k] for k in sorted_keys)):
        yield dict(zip(sorted_keys, combo, strict=True))


def expand_grid_to_run_specs(
    grid: Mapping[str, Iterable[Any]],
    mission: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build a list of :class:`RunSpec` from a full-factorial expansion of ``grid``.

    Each spec gets a sequential ``run_id`` starting at 0, ``script_path`` set
    to ``mission``, ``output_dir`` set to ``<output_dir>/run-<run_id>``,
    ``seed=None``, and ``run_options={}``. The ordering contract from
    :func:`full_factorial` carries through unchanged: ``specs[i].run_id == i``
    and the override dicts appear in cartesian-product order.

    Raises :class:`SweepConfigError` for the same reasons as
    :func:`full_factorial`.
    """
    mission_path = Path(mission)
    base_output_dir = Path(output_dir)
    return [
        RunSpec(
            script_path=mission_path,
            overrides=overrides,
            output_dir=base_output_dir / f"run-{run_id}",
            run_id=run_id,
            seed=None,
            run_options={},
        )
        for run_id, overrides in enumerate(full_factorial(grid))
    ]
