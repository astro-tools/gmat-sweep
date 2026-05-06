# Examples

End-to-end Jupyter notebooks that exercise the `gmat-sweep` API against a
local GMAT install. Each notebook is committed with cleared cell outputs
and re-executed in CI on every push, so the rendered docs always reflect
the current code.

You can run them locally after `pip install gmat-sweep[examples]` (the extra
pulls in matplotlib).

- [Single-axis SMA scan](01_sma_scan.ipynb) — fifty runs across
  `np.linspace(7000, 8000, 50)` of `Sat.SMA`, parallel-dispatched through the
  default `LocalJoblibPool`, overlaid on a single altitude-vs-time plot.
- [Two-axis epoch × time-of-flight grid](02_epoch_arrival_grid.ipynb) — a
  cartesian product over `Sat.Epoch` and a script-level `Variable TOF`,
  reshaped into a 2D matrix and contoured by per-run miss distance.
- [Surviving a kill](03_killed_sweep_recovery.ipynb) — launch a sweep as a
  subprocess, send `SIGINT` mid-run, walk through inspecting the partial
  manifest with `gmat-sweep show` and reloading the partial DataFrame from
  disk, then complete the sweep with a programmatic
  `Sweep.from_manifest(...).resume()` call.
- [Monte Carlo dispersion](04_monte_carlo_dispersion.ipynb) — 1000-run Monte
  Carlo around a nominal injection burn over a four-axis perturbation cube
  (parking-orbit coast time and the three VNB delta-V components). Histogram
  of arrival miss distances, 3-sigma covariance ellipse in the (X, Y) plane,
  and a recipe demonstrating the determinism contract via
  `expand_monte_carlo_to_run_specs`.
- [Latin hypercube vs Monte Carlo](05_latin_hypercube.ipynb) — 64-run Latin
  hypercube alongside a 64-run plain Monte Carlo against the same four-axis
  injection perturbation. Pair plot of the unit-cube samples to make the
  stratification visible, and a side-by-side miss-distance histogram for the
  variance-reduction case.
