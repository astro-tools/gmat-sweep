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

from gmat_sweep.distributions import (
    DistSpec,
    derive_param_seed,
    derive_run_seeds,
    sample,
    to_rv_frozen,
)
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.spec import RunSpec

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "expand_grid_to_run_specs",
    "expand_latin_hypercube_to_run_specs",
    "expand_monte_carlo_extension_to_run_specs",
    "expand_monte_carlo_to_run_specs",
    "expand_samples_to_run_specs",
    "full_factorial",
    "full_factorial_size",
    "iter_grid_run_specs",
    "latin_hypercube_samples",
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

    Materialises the full cartesian product up front — fine for small
    grids but spends O(N) memory on the spec list before the first
    worker starts. Prefer :func:`iter_grid_run_specs` when the
    cartesian product is large (10⁴+ runs): the streaming variant
    yields the same specs one at a time.

    Raises :class:`SweepConfigError` for the same reasons as
    :func:`full_factorial`.
    """
    return list(iter_grid_run_specs(grid, script_path, output_dir))


def iter_grid_run_specs(
    grid: Mapping[str, Iterable[Any]],
    script_path: str | Path,
    output_dir: str | Path,
) -> Iterator[RunSpec]:
    """Stream :class:`RunSpec` instances from a full-factorial expansion of ``grid``.

    Same per-spec shape as :func:`expand_grid_to_run_specs` (sequential
    ``run_id``, per-run ``output_dir``, ``seed=None``, ``run_options={}``)
    but yields lazily — for a 10⁵-row factorial the driver never holds
    more than one :class:`RunSpec` plus :func:`full_factorial`'s
    iterator state in memory.

    Validation (string keys, non-empty values) still runs eagerly at
    the start of iteration via :func:`full_factorial`, so malformed
    grids fail loudly before any spec is yielded.
    """
    script_path_obj = Path(script_path)
    base_output_dir = Path(output_dir)
    for run_id, overrides in enumerate(full_factorial(grid)):
        yield RunSpec(
            script_path=script_path_obj,
            overrides=overrides,
            output_dir=base_output_dir / f"run-{run_id}",
            run_id=run_id,
            seed=None,
            run_options={},
        )


def full_factorial_size(grid: Mapping[str, Iterable[Any]]) -> int:
    """Return the number of runs a :func:`full_factorial` expansion of ``grid`` produces.

    ``prod(len(list(v)) for v in grid.values())`` — but tolerant of
    generators (materialises each axis once) and aware that the empty
    mapping is the identity (one empty-override run, matching
    :func:`full_factorial`).

    Used by :func:`gmat_sweep.sweep` to size the manifest header's
    ``run_count`` and the progress bar without materialising the
    cartesian product itself.
    """
    materialised: list[tuple[Any, ...]] = []
    for key, values in grid.items():
        if not isinstance(key, str):
            raise SweepConfigError(f"grid keys must be strings, got {type(key).__name__}: {key!r}")
        materialised.append(tuple(values))
        if not materialised[-1]:
            raise SweepConfigError(f"grid value for {key!r} is empty")
    n = 1
    for axis in materialised:
        n *= len(axis)
    return n


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


def expand_monte_carlo_to_run_specs(
    perturb: Mapping[str, DistSpec],
    n: int,
    seed: int | None,
    script_path: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build :class:`RunSpec` instances for a Monte Carlo dispersion sweep.

    For each ``run_id`` ``i in range(n)``:

    1. Derive the run-level seed via :func:`derive_run_seeds(seed, n)[i]
       <gmat_sweep.distributions.derive_run_seeds>` — recorded on
       :attr:`RunSpec.seed`.
    2. For each parameter ``k`` in lexicographically-sorted ``perturb``:
       derive a per-parameter sub-seed via
       :func:`derive_param_seed(run_seed, k)
       <gmat_sweep.distributions.derive_param_seed>` and sample one float
       through :func:`sample(perturb[k], sub_seed)
       <gmat_sweep.distributions.sample>`.

    Per-parameter sub-seeds are derived from the parameter *name*, not its
    position in the mapping, so adding a perturbed parameter to an existing
    sweep does not change the draws of any other parameter at any
    ``run_id``. Two calls at the same ``(perturb, n, seed, script_path)``
    return identical specs.

    Raises :class:`SweepConfigError` if ``perturb`` is empty, ``n < 1``, or
    any parameter spec fails its own validation in
    :func:`to_rv_frozen <gmat_sweep.distributions.to_rv_frozen>`.
    """
    if not perturb:
        raise SweepConfigError("monte_carlo requires a non-empty perturb mapping")
    if n < 1:
        raise SweepConfigError(f"monte_carlo requires n >= 1, got {n}")

    sorted_keys = sorted(perturb)
    # Validate every spec up front so the failure is reported before any
    # work — matches full_factorial's "validation runs before any
    # combination is yielded" contract.
    for k in sorted_keys:
        to_rv_frozen(perturb[k])

    run_seeds = derive_run_seeds(seed, n)
    script_path_obj = Path(script_path)
    base_output_dir = Path(output_dir)

    specs: list[RunSpec] = []
    for run_id, run_seed in enumerate(run_seeds):
        overrides: dict[str, Any] = {}
        for k in sorted_keys:
            sub_seed = derive_param_seed(run_seed, k)
            overrides[k] = sample(perturb[k], sub_seed)
        specs.append(
            RunSpec(
                script_path=script_path_obj,
                overrides=overrides,
                output_dir=base_output_dir / f"run-{run_id}",
                run_id=run_id,
                seed=run_seed,
                run_options={},
            )
        )
    return specs


def expand_monte_carlo_extension_to_run_specs(
    perturb: Mapping[str, DistSpec],
    old_n: int,
    n: int,
    seed: int | None,
    script_path: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build :class:`RunSpec` instances for the *extension* slice of an MC sweep.

    Mirrors :func:`expand_monte_carlo_to_run_specs` but emits only the
    ``[old_n, old_n + n)`` tail. The two key facts that make this slice
    bit-equal to the same indices of a fresh ``monte_carlo(n=old_n+n, ...)``
    call:

    1. :func:`numpy.random.SeedSequence.spawn` is position-deterministic —
       the ``i``-th child depends only on ``(parent, i)``, not on ``n``. So
       :func:`derive_run_seeds(seed, total) <gmat_sweep.distributions.derive_run_seeds>`
       at indices ``[old_n, total)`` matches the same indices of a fresh
       ``derive_run_seeds(seed, total)`` call regardless of how the original
       sweep was sized.
    2. Per-parameter sub-seeds are derived from
       :func:`derive_param_seed(run_seed, name) <gmat_sweep.distributions.derive_param_seed>`,
       which keys on the parameter name. The extension reuses the same
       ``perturb`` mapping by construction (extension does not let the caller
       change distributions), so each parameter's sub-seed at every extended
       ``run_id`` is bit-equal to a fresh sweep at the same total ``n``.

    Together these two facts give the bit-equivalence the
    :func:`gmat_sweep.monte_carlo_extend` contract rests on.

    Raises :class:`SweepConfigError` if ``perturb`` is empty, ``old_n < 0``,
    ``n < 1``, or any parameter spec fails its own validation in
    :func:`to_rv_frozen <gmat_sweep.distributions.to_rv_frozen>`.
    """
    if not perturb:
        raise SweepConfigError("monte_carlo_extend requires a non-empty perturb mapping")
    if old_n < 0:
        raise SweepConfigError(f"monte_carlo_extend requires old_n >= 0, got {old_n}")
    if n < 1:
        raise SweepConfigError(f"monte_carlo_extend requires n >= 1, got {n}")

    sorted_keys = sorted(perturb)
    for k in sorted_keys:
        to_rv_frozen(perturb[k])

    # Spawn over the full [0, old_n + n) range so per-run seeds at indices
    # >= old_n are bit-equal to a fresh n=old_n+n call. SeedSequence.spawn
    # is position-deterministic so we could in principle skip the first
    # old_n entries, but spawn is microseconds and matching the existing
    # expander's call shape verbatim is what proves equivalence.
    run_seeds = derive_run_seeds(seed, old_n + n)
    script_path_obj = Path(script_path)
    base_output_dir = Path(output_dir)

    specs: list[RunSpec] = []
    for run_id in range(old_n, old_n + n):
        run_seed = run_seeds[run_id]
        overrides: dict[str, Any] = {}
        for k in sorted_keys:
            sub_seed = derive_param_seed(run_seed, k)
            overrides[k] = sample(perturb[k], sub_seed)
        specs.append(
            RunSpec(
                script_path=script_path_obj,
                overrides=overrides,
                output_dir=base_output_dir / f"run-{run_id}",
                run_id=run_id,
                seed=run_seed,
                run_options={},
            )
        )
    return specs


def latin_hypercube_samples(
    perturb: Mapping[str, DistSpec],
    n: int,
    seed: int | None,
) -> pd.DataFrame:
    """Build the Latin hypercube samples DataFrame for a stochastic sweep.

    Builds a :class:`scipy.stats.qmc.LatinHypercube` sampler with
    ``d = len(perturb)`` and ``seed = seed``, draws ``n`` unit-cube points,
    then maps each column through ``to_rv_frozen(perturb[k]).ppf(...)`` to
    leave the unit cube.

    Columns are emitted in lexicographic order so the run set is stable
    under ``perturb``-dict reordering. The returned DataFrame has a default
    :class:`pandas.RangeIndex` and is suitable input to
    :func:`expand_samples_to_run_specs`.

    Determinism: two calls with the same ``(perturb, n, seed)`` produce
    bit-equal DataFrames.

    Raises :class:`SweepConfigError` if ``perturb`` is empty, ``n < 1``, or
    any parameter spec fails its own validation in
    :func:`to_rv_frozen <gmat_sweep.distributions.to_rv_frozen>`.
    """
    import pandas as pd
    from scipy.stats import qmc

    if not perturb:
        raise SweepConfigError("latin_hypercube requires a non-empty perturb mapping")
    if n < 1:
        raise SweepConfigError(f"latin_hypercube requires n >= 1, got {n}")

    sorted_keys = sorted(perturb)
    rvs = [to_rv_frozen(perturb[k]) for k in sorted_keys]

    sampler = qmc.LatinHypercube(d=len(sorted_keys), seed=seed)
    unit = sampler.random(n=n)  # shape (n, d), each entry in [0, 1)

    columns: dict[str, Any] = {}
    for col_idx, key in enumerate(sorted_keys):
        columns[key] = rvs[col_idx].ppf(unit[:, col_idx])
    return pd.DataFrame(columns)


def expand_latin_hypercube_to_run_specs(
    perturb: Mapping[str, DistSpec],
    n: int,
    seed: int | None,
    script_path: str | Path,
    output_dir: str | Path,
) -> list[RunSpec]:
    """Build :class:`RunSpec` instances for a Latin hypercube sweep.

    Convenience wrapper that builds the samples DataFrame via
    :func:`latin_hypercube_samples` and forwards to
    :func:`expand_samples_to_run_specs`. Per-run seeds are not populated:
    the draw set is fully determined by ``(perturb, n, seed)`` so
    individual runs do not need their own RNG state.
    """
    samples = latin_hypercube_samples(perturb, n=n, seed=seed)
    return expand_samples_to_run_specs(samples, script_path, output_dir)
