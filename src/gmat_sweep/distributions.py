"""Distribution specs, seeded sampling, and conversion to scipy.stats.rv_frozen.

Internal infrastructure for the upcoming Monte Carlo (#33) and Latin hypercube
(#35) public APIs. None of the symbols here are re-exported from
:mod:`gmat_sweep` — callers reach in via ``gmat_sweep.distributions.*``.

:func:`derive_run_seeds` is **the** contract Monte Carlo replays depend on
(charter §5): spawning child :class:`numpy.random.SeedSequence` instances from
a parent and reading the first 32-bit word of each child's ``generate_state(1)``
yields a list of per-run integer seeds that is identical across calls in the
same process and across fresh processes. Two callers given the same
``parent_seed`` and ``n`` reconstruct the same per-run RNG state; this is what
lets a Monte Carlo sweep be replayed run-by-run from its manifest alone.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from gmat_sweep.errors import SweepConfigError

__all__ = [
    "DistSpec",
    "derive_run_seeds",
    "sample",
    "to_rv_frozen",
]


_NormalSpec: TypeAlias = tuple[Literal["normal"], float, float]
_UniformSpec: TypeAlias = tuple[Literal["uniform"], float, float]
_LognormalSpec: TypeAlias = tuple[Literal["lognormal"], float, float]

DistSpec: TypeAlias = _NormalSpec | _UniformSpec | _LognormalSpec | rv_frozen
"""User-facing distribution specification.

One of three shorthand tuples or a pre-frozen scipy distribution:

* ``("normal", mu, sigma)`` — :func:`scipy.stats.norm` with ``loc=mu, scale=sigma``.
* ``("uniform", lo, hi)`` — :func:`scipy.stats.uniform` with ``loc=lo, scale=hi - lo``.
* ``("lognormal", mu, sigma)`` — :func:`scipy.stats.lognorm` with ``s=sigma, scale=exp(mu)``.
* a pre-frozen :class:`scipy.stats._distn_infrastructure.rv_frozen` — passes
  through :func:`to_rv_frozen` unchanged for callers that need a distribution
  shape outside the three shorthands.
"""


def to_rv_frozen(spec: DistSpec) -> rv_frozen:
    """Coerce a :data:`DistSpec` into a frozen scipy distribution.

    A pre-frozen ``rv_frozen`` is returned unchanged (same object identity).
    Shorthand tuples are validated and mapped to the corresponding
    :mod:`scipy.stats` factory.

    Raises :class:`SweepConfigError` for: an unknown shorthand tag, a tuple of
    the wrong length, non-numeric or non-finite parameters, ``sigma <= 0``
    (normal, lognormal), or ``hi <= lo`` (uniform).
    """
    if isinstance(spec, rv_frozen):
        return spec

    if not isinstance(spec, tuple):
        raise SweepConfigError(
            f"distribution spec must be a shorthand tuple or scipy rv_frozen, "
            f"got {type(spec).__name__}"
        )

    if len(spec) == 0:
        raise SweepConfigError("distribution spec tuple is empty")

    tag = spec[0]
    if tag == "normal":
        mu, sigma = _two_floats(spec, "normal")
        if sigma <= 0:
            raise SweepConfigError(f"'normal' distribution sigma must be > 0, got {sigma!r}")
        return cast(rv_frozen, stats.norm(loc=mu, scale=sigma))
    if tag == "uniform":
        lo, hi = _two_floats(spec, "uniform")
        if hi <= lo:
            raise SweepConfigError(
                f"'uniform' distribution requires hi > lo, got lo={lo!r}, hi={hi!r}"
            )
        return cast(rv_frozen, stats.uniform(loc=lo, scale=hi - lo))
    if tag == "lognormal":
        mu, sigma = _two_floats(spec, "lognormal")
        if sigma <= 0:
            raise SweepConfigError(f"'lognormal' distribution sigma must be > 0, got {sigma!r}")
        return cast(rv_frozen, stats.lognorm(s=sigma, scale=math.exp(mu)))
    raise SweepConfigError(
        f"unknown distribution shorthand tag {tag!r}; "
        f"expected one of 'normal', 'uniform', 'lognormal'"
    )


def _two_floats(spec: tuple[Any, ...], tag: str) -> tuple[float, float]:
    if len(spec) != 3:
        raise SweepConfigError(
            f"{tag!r} distribution spec must be a length-3 tuple, got length {len(spec)}"
        )
    a_raw, b_raw = spec[1], spec[2]
    if isinstance(a_raw, bool) or isinstance(b_raw, bool):
        raise SweepConfigError(
            f"{tag!r} distribution parameters must be numeric, got {a_raw!r} and {b_raw!r}"
        )
    try:
        a = float(a_raw)
        b = float(b_raw)
    except (TypeError, ValueError) as e:
        raise SweepConfigError(
            f"{tag!r} distribution parameters must be numeric, got {a_raw!r} and {b_raw!r}"
        ) from e
    if not (math.isfinite(a) and math.isfinite(b)):
        raise SweepConfigError(
            f"{tag!r} distribution parameters must be finite, got {a!r} and {b!r}"
        )
    return a, b


def derive_run_seeds(parent_seed: int | None, n: int) -> list[int]:
    """Derive ``n`` per-run integer seeds from ``parent_seed``.

    Uses ``numpy.random.SeedSequence(parent_seed).spawn(n)`` and reads the
    first 32-bit word of each child's ``generate_state(1)``. For an integer
    ``parent_seed`` the result is identical across calls in the same process
    and across fresh processes — the Monte Carlo replay contract.

    ``parent_seed=None`` falls back to OS entropy and is **not** reproducible.
    ``n`` must be ``>= 0``; ``n=0`` returns ``[]``.
    """
    if n < 0:
        raise SweepConfigError(f"derive_run_seeds requires n >= 0, got {n}")
    if n == 0:
        return []
    seq = np.random.SeedSequence(parent_seed)
    children = seq.spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]


def sample(spec: DistSpec, seed: int) -> float:
    """Draw one float from ``spec`` using a fresh ``numpy.random.default_rng(seed)``.

    Reproducible per ``(spec, seed)``: each call constructs its own RNG, so
    nothing about the draw depends on global RNG state or call order.
    """
    rv = to_rv_frozen(spec)
    rng = np.random.default_rng(seed)
    return float(rv.rvs(size=1, random_state=rng)[0])
