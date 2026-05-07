# Benchmarks

Wall-clock and throughput numbers for the three execution backends on a
1000-run reference sweep, plus the in-CI throughput regression that catches
per-PR slowdowns before release.

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
| GMAT version | _TBD_ |
| `gmat-sweep` version | _TBD_ |
| CPU | _TBD_ |
| RAM | _TBD_ |
| OS | _TBD_ |

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
| `LocalJoblibPool(workers=8)` | _TBD_ | _TBD_ | _TBD_ |
| `DaskPool(n_workers=8)` (LocalCluster) | _TBD_ | _TBD_ | _TBD_ |
| `RayPool(num_cpus=8)` (local) | _TBD_ | _TBD_ | _TBD_ |

## Throughput

| Backend | Runs/sec | Per-worker runs/sec |
| --- | --- | --- |
| `LocalJoblibPool(workers=8)` | _TBD_ | _TBD_ |
| `DaskPool(n_workers=8)` | _TBD_ | _TBD_ |
| `RayPool(num_cpus=8)` | _TBD_ | _TBD_ |

## Discussion

_To be written once the numbers are measured. The expected shape: all three
backends land within roughly 10 % of each other on a single machine — Dask and
Ray pay a small dispatch-layer overhead, but per-run GMAT load time dominates.
The picture changes once the sweep spans more than one machine; see
[Backends](backends.md) for when each backend is worth its overhead._

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
```

Each invocation prints a JSON record with `wall_seconds` and
`throughput_runs_per_sec` on stdout. The 50-run scaled variant is what the CI
throughput regression executes:

```bash
python -m tests.data.benchmark_sweep --scale 50 --backend local --workers 4
```

## CI regression gate

The `backend-throughput` CI job runs the 50-run sweep on each of the three
backends and asserts a per-backend throughput floor. The floor JSON lives at
[`tests/data/throughput_floor.json`][floor-json]; updates show as deliberate
diffs in PRs, so a tightening or relaxation is reviewable rather than
accidental. A regression below the floor fails CI with a message naming the
backend, the measured rate, and the floor.

[floor-json]: https://github.com/astro-tools/gmat-sweep/blob/main/tests/data/throughput_floor.json
