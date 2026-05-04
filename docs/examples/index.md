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
  subprocess, send `SIGINT` mid-run, and walk through inspecting the partial
  manifest with `gmat-sweep show` and reloading the partial DataFrame from
  disk. Demonstrates the durability claim; programmatic
  `Sweep.from_manifest(...).resume()` lands in v0.2.
