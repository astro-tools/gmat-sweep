# Benchmarks

Wall-clock and throughput numbers on a 1000-run reference sweep for every
backend that fits the single-machine rig — `LocalJoblibPool`,
`DaskPool`, `RayPool`, `ProcessPoolExecutorPool`, and `MPIPool` — plus
the in-CI throughput regression that catches per-PR slowdowns before
release. `KubernetesJobPool` and `DebugPool` aren't reported here for
reasons noted under [Excluded backends](#excluded-backends).

## Setup

The reference sweep runs `Sat.SMA` swept across `np.linspace(7000, 8000, 1000)`
against the LEO basic mission fixture under
[`tests/data/leo_basic.script`][leo-basic] — a one-Spacecraft, point-mass,
60-second propagate that exercises the full pipeline (script load, propagator,
ReportFile output, Parquet write) without paying for stationkeeping or
stock-sample support files.

| What | Value |
| --- | --- |
| Run count | 1000 |
| Mission script | [`tests/data/leo_basic.script`][leo-basic] |
| Sweep parameter | `Sat.SMA` ∈ `np.linspace(7000, 8000, 1000)` |
| Workers per backend | 8 |
| GMAT version | R2026a |
| `gmat-sweep` version | [`3193e55`](https://github.com/astro-tools/gmat-sweep/commit/3193e55) (post-0.4 `main`) |
| CPU | Intel® Core™ i7-10700 @ 2.90 GHz (8 cores / 16 threads) |
| RAM | 16 GB |
| OS | Linux 6.6.114.1 (WSL2, x86_64) |
| Filesystem | ext4 (WSL2 native, under `/home`) |

The benchmark fixture is committed at
[`tests/data/benchmark_sweep.py`][benchmark-script]; the docs reproduce-command
and the CI throughput regression test share that single sweep definition so the
docs and CI numbers cannot drift.

[leo-basic]: https://github.com/astro-tools/gmat-sweep/blob/main/tests/data/leo_basic.script
[benchmark-script]: https://github.com/astro-tools/gmat-sweep/blob/main/tests/data/benchmark_sweep.py

## Per-backend timings

Wall-clock seconds, median of three runs, min–max range in parentheses.

| Backend | Median (s) | Min (s) | Max (s) |
| --- | --- | --- | --- |
| `LocalJoblibPool(max_workers=8)` | 11.93 | 11.02 | 12.37 |
| `DaskPool(n_workers=8)` (LocalCluster) | 13.55 | 13.30 | 14.48 |
| `RayPool(num_cpus=8)` (local) | 14.06 | 13.71 | 14.68 |
| `MPIPool(max_workers=8)` (`mpi4py.futures`, 9 ranks) | 12.64 | 12.54 | 13.01 |
| `ProcessPoolExecutorPool(max_workers=8)` (Python ≥ 3.11) | 269.86 | 268.37 | 270.34 |

## Throughput

| Backend | Runs/sec | Per-worker runs/sec |
| --- | --- | --- |
| `LocalJoblibPool(max_workers=8)` | 83.83 | 10.48 |
| `DaskPool(n_workers=8)` | 73.80 | 9.23 |
| `RayPool(num_cpus=8)` | 71.11 | 8.89 |
| `MPIPool(max_workers=8)` | 79.14 | 9.89 |
| `ProcessPoolExecutorPool(max_workers=8)` | 3.71 | 0.46 |

## Excluded backends

Two pools that ship with `gmat-sweep` aren't represented in the table
above. Their numbers don't belong on a single-machine reference setup:

- **`KubernetesJobPool`** — every run becomes a separate `Job` / Pod, so
  wall-clock is dominated by per-Pod scheduling, image pull, and PVC
  mount, not by GMAT propagation. A single-machine kind cluster would
  measure something other than what production deployments care about.
  The authoritative single-machine kind numbers live in the
  `backend-k8s` CI cell — it runs the same 50-run scaled fixture under
  `tests/test_backend_throughput.py` against a kind-provisioned cluster
  and asserts the floor in
  [`tests/data/throughput_floor.json`][floor-json] (`"k8s"` key). For
  multi-host production sizing, measure on the target cluster against
  the same `tests/data/benchmark_sweep.py` fixture.
- **`DebugPool`** — the in-process, single-spec backend for
  `breakpoint()`-driven debugging. It refuses to dispatch more than one
  spec by construction (raising `BackendError`), so "throughput" isn't a
  defined metric for it. See
  [`gmat_sweep.backends.debug.DebugPool`][debugpool] for the design and
  the two-flag opt-in required to use it.

[debugpool]: https://github.com/astro-tools/gmat-sweep/blob/main/src/gmat_sweep/backends/debug.py

## Discussion

On this 8-worker / single-machine setup the local joblib pool tops the
table at roughly 83.8 runs/sec. `MPIPool` lands within ~6 % of it
(79.1 runs/sec) — `mpi4py.futures` with pre-allocated ranks reuses worker
processes the same way joblib's loky backend does, so the per-task
dispatch is lean once the gmatpy bootstrap is amortised. `DaskPool` and
`RayPool` follow at 73.8 and 71.1 runs/sec — about 12–15 % below local
because each of their dispatch layers pays per-task scheduling overhead
that loky avoids on a single host. Per-worker throughput tracks the same
ordering (10.5 / 9.9 / 9.2 / 8.9 runs/sec). Across all four of these
backends the per-run GMAT load is amortised identically because
`reuse_gmat_context=True` (the default) keeps a single gmatpy import
alive in each worker for the lifetime of the sweep.

`ProcessPoolExecutorPool` is the outlier at 3.7 runs/sec — about 22×
below local. That gap is structural, not a regression: the pool is
constructed with `max_tasks_per_child=1`, so every task runs in a fresh
worker interpreter and pays the gmatpy bootstrap cost individually. The
guarantee — one fresh interpreter per task, no joblib runtime
dependency, no shared state — is what the backend trades wall-clock for.
For sweeps where many runs share the same script, `LocalJoblibPool` is
the right choice; reach for `ProcessPoolExecutorPool` when the
fresh-interpreter contract or the stdlib-only dependency surface
matters.

The picture changes once the sweep spans more than one machine; see
[Backends](backends.md) for when each backend is worth its overhead.

## How to reproduce

The benchmark fixture is invokable as a module — pass `--backend` and `--scale`
to pick the variant:

```bash
# Full 1000-run benchmark on the local joblib backend
python -m tests.data.benchmark_sweep --scale 1000 --backend local --workers 8

# Same on Dask (LocalCluster)
python -m tests.data.benchmark_sweep --scale 1000 --backend dask --workers 8

# Same on Ray (local runtime)
python -m tests.data.benchmark_sweep --scale 1000 --backend ray --workers 8

# ProcessPoolExecutorPool — stdlib, Python 3.11+
python -m tests.data.benchmark_sweep --scale 1000 --backend process --workers 8

# MPIPool — pre-allocated ranks under mpi4py.futures.
# `-n 9` provisions 1 driver rank + 8 worker ranks for `--workers 8`.
# `--oversubscribe` is needed because 9 > the rig's 8 cores.
mpirun -n 9 --oversubscribe \
  python -m mpi4py.futures -m tests.data.benchmark_sweep --scale 1000 --backend mpi --workers 8
```

Each invocation prints a JSON record with `wall_seconds` and
`throughput_runs_per_sec` on stdout. The 50-run scaled variant is what the CI
throughput regression executes:

```bash
python -m tests.data.benchmark_sweep --scale 50 --backend local --workers 4
```

## CI regression gate

The 50-run scaled fixture runs across three CI cells that together cover
every shipping backend:

- `backend-throughput` covers `LocalJoblibPool`, `DaskPool`, `RayPool`,
  and `ProcessPoolExecutorPool` (Python 3.12 in the cell, so the
  ≥ 3.11 gate passes).
- `backend-mpi` covers `MPIPool` under
  `mpirun -n K --oversubscribe python -m mpi4py.futures -m pytest …`,
  filtered by `-k mpi` to the MPI parametrize rows only.
- `backend-k8s` covers `KubernetesJobPool` against a kind-provisioned
  cluster, filtered by `-k k8s`.

Each cell asserts a per-backend throughput floor. The floor JSON lives
at [`tests/data/throughput_floor.json`][floor-json] and carries an entry
for every backend `tests.data.benchmark_sweep.BACKENDS` enumerates;
updates show as deliberate diffs in PRs, so a tightening or relaxation
is reviewable rather than accidental. A regression below the floor
fails CI with a message naming the backend, the measured rate, and the
floor.

[floor-json]: https://github.com/astro-tools/gmat-sweep/blob/main/tests/data/throughput_floor.json
