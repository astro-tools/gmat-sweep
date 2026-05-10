# Examples

End-to-end Jupyter notebooks that exercise the `gmat-sweep` API against a
local GMAT install. Each notebook is committed with cleared cell outputs
and re-executed in CI on every push, so the rendered docs always reflect
the current code.

You can run them locally after `pip install gmat-sweep[examples]` (the extra
pulls in matplotlib, `distributed`, and `ray` so the cluster-backend
notebooks run on a laptop too).

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
- [Dask cluster recipe](06_dask_cluster_recipe.ipynb) — 100-run `Sat.SMA`
  grid sweep dispatched through a `distributed.LocalCluster` with `DaskPool`.
  Same client API, same dashboard, same submit/await flow as a real
  `dask.distributed` cluster.
- [Ray autoscaling recipe](07_ray_autoscaling_recipe.ipynb) — 100-run Monte
  Carlo against the notebook 04 fixture, dispatched through `RayPool` against
  a local `ray.init()`. Same task model as a real autoscaling Ray cluster.
- [Sobol sensitivity](08_sobol_sensitivity.ipynb) — Saltelli/Sobol design
  built by `sobol_sample`, dispatched through `sweep(samples=...)` against
  the notebook 04 fixture, then reduced to first/total-order indices via
  `sobol_analyze` with bootstrap confidence intervals.
- [Archive bundle](09_archive_bundle.ipynb) — pack a finished sweep
  (script, manifest, per-run Parquet files) into a single `.zip` via
  `Sweep.archive()`, inspect the bundle's layout, and re-aggregate the
  per-run DataFrame from the unzipped tree without re-running.
- [Extending a Monte Carlo](10_extending_monte_carlo.ipynb) — anchor a
  100-run `monte_carlo` with `out=`, append 200 more runs via
  `monte_carlo_extend(n=200)`, and assert that the original 100 `run_id`s
  are preserved bit-for-bit in the 300-run aggregated frame.
