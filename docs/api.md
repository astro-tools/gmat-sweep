# API reference

Every public symbol re-exported from the `gmat_sweep` top-level package,
auto-generated from docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/).

## Sweep entry points

::: gmat_sweep.sweep

::: gmat_sweep.monte_carlo

::: gmat_sweep.monte_carlo_extend

::: gmat_sweep.latin_hypercube

::: gmat_sweep.latin_hypercube_extend

::: gmat_sweep.Sweep

## Execution backend

::: gmat_sweep.Pool

::: gmat_sweep.LocalJoblibPool

::: gmat_sweep.backends.DaskPool

::: gmat_sweep.backends.RayPool

::: gmat_sweep.backends.KubernetesJobPool

::: gmat_sweep.backends.MPIPool

::: gmat_sweep.backends.ProcessPoolExecutorPool

## Specs and outcomes

::: gmat_sweep.RunSpec

::: gmat_sweep.SweepSpec

::: gmat_sweep.RunOutcome

## Grid expansion

::: gmat_sweep.full_factorial

::: gmat_sweep.expand_grid_to_run_specs

## Manifest

::: gmat_sweep.MANIFEST_SCHEMA_VERSION

::: gmat_sweep.Manifest

::: gmat_sweep.ManifestEntry

::: gmat_sweep.canonical_script_sha256

## Aggregation

::: gmat_sweep.lazy_multiindex

::: gmat_sweep.lazy_ephemerides

::: gmat_sweep.lazy_contacts

::: gmat_sweep.lazy_fused_reports

::: gmat_sweep.sweep_summary

::: gmat_sweep.sweep_diff

::: gmat_sweep.mc_convergence

## Plotting

::: gmat_sweep.plotting.sweep_band_plot

::: gmat_sweep.plotting.mc_convergence_plot

## Exceptions

::: gmat_sweep.GmatSweepError

::: gmat_sweep.SweepConfigError

::: gmat_sweep.RunFailed

::: gmat_sweep.BackendError

::: gmat_sweep.ManifestCorruptError
