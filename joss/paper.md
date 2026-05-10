---
title: 'gmat-sweep: Parallel parameter sweeps and Monte Carlo dispersions over NASA GMAT missions'
tags:
  - Python
  - astrodynamics
  - orbital mechanics
  - mission analysis
  - Monte Carlo
  - parameter sweep
  - parallel computing
  - GMAT
authors:
  - name: Dimitrije Jankovic
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 9 May 2026
bibliography: paper.bib
---

<!--
This file is a v0.4-cycle skeleton. The narrative content is intentionally
stub-level; the paper itself is drafted as part of the v1.0 adoption push
(see issue #94). When expanding this paper, replace the TODO markers below
with real prose, verify every bib entry in paper.bib, and confirm the author
ORCID + affiliation reflect the submitting state at JOSS submission time.
-->

# Summary

`gmat-sweep` is a Python library for running parameter sweeps and Monte
Carlo dispersions over NASA's General Mission Analysis Tool [GMAT,
@gmat-nasa] in parallel. Given a working `.script` mission and either a
parameter grid, an explicit run table, or a perturbation distribution, it
fans the run set across subprocess workers, aggregates each run's outputs
into multi-indexed `pandas` DataFrames, and writes a JSON Lines manifest
so any sweep is reproducible and resumable.

<!-- v1.0 TODO: expand the summary into the standard JOSS three-paragraph form. -->

# Statement of need

Mission-analysis workflows routinely require running the same GMAT mission
hundreds or thousands of times under varied initial conditions, perturbed
parameters, or stochastic dispersions. GMAT's own scripting layer offers
no first-class abstraction for parametric or Monte Carlo execution, and
running independent missions sequentially leaves modern multi-core and
cluster hardware idle.

`gmat-sweep` fills that gap with four entry points (`sweep`,
`monte_carlo`, `latin_hypercube`, and an explicit-row variant of `sweep`)
and a pluggable backend layer that targets local thread/process pools via
`joblib` [@joblib], distributed clusters via Dask [@dask] or Ray [@ray],
HPC schedulers via MPI, and Kubernetes Jobs. Statistical sampling reuses
the well-validated distributions in SciPy [@scipy], and per-run sub-seeds
derive deterministically from a user-supplied root seed so a resumed
sweep samples the same values for any given run identifier.

<!-- v1.0 TODO: contrast with adjacent tooling and expand on the gap. -->

# Functionality overview

## Parameter sweeps

`sweep()` accepts either a full-factorial grid over one or more
dotted-path fields of the GMAT script, or an explicit `pandas` DataFrame
of run parameters. The aggregated output is a `(run_id, time)`-indexed
DataFrame that streams from per-run Parquet files, so sweeps with tens of
thousands of runs do not have to fit in memory.

## Monte Carlo and Latin hypercube

`monte_carlo()` and `latin_hypercube()` build on the same execution
machinery but draw their parameter rows from named distributions
(`normal`, `uniform`, `lognormal`, etc.) before dispatch. The seeding
contract guarantees bit-reproducible draws across runs and resumes.

## Cluster backends

A single `Pool` abstraction lets the same `sweep()` call dispatch to a
local pool, a Dask cluster, a Ray cluster, an MPI communicator, or a
Kubernetes job pool by switching only the `backend=` argument. A
cross-backend equivalence suite asserts that the choice of backend is
purely an execution-only knob with no observable effect on results.

## Reproducibility and resumability

Every sweep emits a JSON Lines manifest fingerprinting the canonical
script SHA-256, software versions, full parameter spec, and per-run
status. A killed sweep restarts from the manifest and re-runs only the
missing or failed entries; the underlying `gmat-run` [@gmat-run]
single-run primitive carries the per-run isolation contract.

<!-- v1.0 TODO: add benchmark figures and a worked example. -->

# Acknowledgements

The author thanks the GMAT development team at NASA Goddard Space Flight
Center for maintaining the underlying mission-analysis platform that
`gmat-sweep` builds upon.

<!-- v1.0 TODO: acknowledge specific reviewers, contributors, and funding sources at submission time. -->

# References
