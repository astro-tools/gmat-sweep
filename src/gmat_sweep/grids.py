"""Full-factorial, explicit-row, Latin hypercube, and quasi-Monte-Carlo run generators.

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
from typing import TYPE_CHECKING, Any

from gmat_sweep.errors import SweepConfigError
from gmat_sweep.spec import RunSpec

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "expand_grid_to_run_specs",
    "expand_samples_to_run_specs",
    "full_factorial",
]


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
    script_path: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build a list of :class:`RunSpec` from a full-factorial expansion of ``grid``.

    Each spec gets a sequential ``run_id`` starting at 0, ``script_path``
    propagated through, ``output_dir`` set to ``<output_dir>/run-<run_id>``,
    ``seed=None``, and ``run_options={}``. The ordering contract from
    :func:`full_factorial` carries through unchanged: ``specs[i].run_id == i``
    and the override dicts appear in cartesian-product order.

    Raises :class:`SweepConfigError` for the same reasons as
    :func:`full_factorial`.
    """
    script_path_obj = Path(script_path)
    base_output_dir = Path(output_dir)
    return [
        RunSpec(
            script_path=script_path_obj,
            overrides=overrides,
            output_dir=base_output_dir / f"run-{run_id}",
            run_id=run_id,
            seed=None,
            run_options={},
        )
        for run_id, overrides in enumerate(full_factorial(grid))
    ]


def expand_samples_to_run_specs(
    samples: pd.DataFrame,
    script_path: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build a list of :class:`RunSpec` from an explicit-row sample DataFrame.

    Each row becomes one :class:`RunSpec` with ``overrides = row.to_dict()``,
    ``run_id`` equal to the row's positional index, ``output_dir`` set to
    ``<output_dir>/run-<run_id>``, ``seed=None``, and ``run_options={}``. The
    DataFrame's column names are dotted-path field names — the same shape
    :func:`expand_grid_to_run_specs` already produces — so analysts may
    pre-build any sampling design (Latin hypercube, Halton/Sobol, custom)
    themselves and hand the result in directly.

    Per-cell NaN is forwarded as-is. ``gmat-run`` is the line that decides
    whether NaN is a valid value for a given dotted path; this expander does
    not second-guess it.

    Validation is strict and runs before any spec is built:

    - ``samples`` must be a :class:`pandas.DataFrame`.
    - All column names must be :class:`str` instances.
    - Column names must be unique — duplicates would silently lose data when
      :meth:`pandas.Series.to_dict` collapses them into a single key.
    - The DataFrame index must equal :class:`pandas.RangeIndex(start=0,
      stop=len(samples))` so ``run_id`` and the row's positional index agree.
    - No column may be entirely NaN (an all-NaN axis carries no signal).

    Any violation raises :class:`SweepConfigError` with a message naming the
    offending column or index.
    """
    import pandas as pd

    if not isinstance(samples, pd.DataFrame):
        raise SweepConfigError(f"samples must be a pandas.DataFrame, got {type(samples).__name__}")

    non_string_cols = [c for c in samples.columns if not isinstance(c, str)]
    if non_string_cols:
        raise SweepConfigError(
            f"samples column names must be strings, got non-string columns: {non_string_cols!r}"
        )

    duplicates = samples.columns[samples.columns.duplicated()].unique().tolist()
    if duplicates:
        raise SweepConfigError(f"samples has duplicate column names: {duplicates!r}")

    expected_index = pd.RangeIndex(start=0, stop=len(samples))
    if not samples.index.equals(expected_index):
        raise SweepConfigError(
            "samples index must be a default RangeIndex starting at 0; "
            f"got {samples.index!r}. Call .reset_index(drop=True) first."
        )

    # Guard the all-NaN check on len > 0: .isna().all() is vacuously True on
    # zero-length columns, which would reject every empty DataFrame with a
    # misleading message.
    if len(samples) > 0:
        all_nan_cols = [c for c in samples.columns if samples[c].isna().all()]
        if all_nan_cols:
            raise SweepConfigError(f"samples has all-NaN columns: {all_nan_cols!r}")

    script_path_obj = Path(script_path)
    base_output_dir = Path(output_dir)
    records = samples.to_dict(orient="records")
    return [
        RunSpec(
            script_path=script_path_obj,
            overrides={str(k): v for k, v in record.items()},
            output_dir=base_output_dir / f"run-{run_id}",
            run_id=run_id,
            seed=None,
            run_options={},
        )
        for run_id, record in enumerate(records)
    ]
