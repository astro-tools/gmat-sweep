"""Latin hypercube stratification + determinism — v0.2 validation suite (issue #40).

Pins the two contracts ``latin_hypercube_samples`` ships with: the samples are
stratified across the unit cube (one point per stratum per axis — the LH
guarantee that buys the small-``n`` coverage advantage over plain Monte Carlo),
and the draws are bit-equal across two calls at the same ``(perturb, n, seed)``
so a downstream consumer can replay a sweep from its manifest alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gmat_sweep.grids import latin_hypercube_samples


def test_1000_uniform_samples_have_one_point_per_equal_width_stratum() -> None:
    """A 1000-point LH on uniform ``[0, 1]`` places exactly one point in each
    of 1000 equal-width strata — the headline LH guarantee."""
    samples = latin_hypercube_samples(
        perturb={"x": ("uniform", 0.0, 1.0)},
        n=1000,
        seed=42,
    )
    assert len(samples) == 1000
    assert list(samples.columns) == ["x"]

    counts, _edges = np.histogram(samples["x"].to_numpy(), bins=1000, range=(0.0, 1.0))
    assert (counts == 1).all(), (
        f"expected exactly one point per stratum, got distribution: "
        f"min={counts.min()}, max={counts.max()}, "
        f"non-singleton bins={(counts != 1).sum()}"
    )


def test_latin_hypercube_samples_bit_equal_across_two_calls() -> None:
    """Two ``latin_hypercube_samples(perturb, n=64, seed=42)`` calls produce
    bit-equal DataFrames — the determinism contract that lets a sweep be
    replayed from its manifest's ``(perturb, n, seed)`` alone."""
    perturb = {
        "Sat.SMA": ("normal", 7100.0, 50.0),
        "Sat.INC": ("uniform", 0.0, 90.0),
    }
    a = latin_hypercube_samples(perturb=perturb, n=64, seed=42)
    b = latin_hypercube_samples(perturb=perturb, n=64, seed=42)

    pd.testing.assert_frame_equal(a, b)
