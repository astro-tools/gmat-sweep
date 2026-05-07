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
| GMAT version | R2026a |
| `gmat-sweep` version | [`110d4be`](https://github.com/astro-tools/gmat-sweep/commit/110d4be) (post-0.2.0 `main`) |
| CPU | Intel® Core™ i7-10700 @ 2.90 GHz (8 cores / 16 threads) |
| RAM | 16 GB |
| OS | Linux 6.6.87 (WSL2, x86_64) |

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
| `LocalJoblibPool(workers=8)` | 12.10 | 11.85 | 12.51 |
| `DaskPool(n_workers=8)` (LocalCluster) | 14.32 | 14.22 | 14.73 |
| `RayPool(num_cpus=8)` (local) | 14.61 | 14.52 | 14.73 |

## Throughput

| Backend | Runs/sec | Per-worker runs/sec |
| --- | --- | --- |
| `LocalJoblibPool(workers=8)` | 82.62 | 10.33 |
| `DaskPool(n_workers=8)` | 69.85 | 8.73 |
| `RayPool(num_cpus=8)` | 68.46 | 8.56 |

## Discussion

On this 8-worker / single-machine setup the local joblib pool turns in
roughly 82.6 runs/sec, with Dask at 69.9 and Ray at 68.5 — about 15–17 %
below local. Per-worker throughput tracks the same gap (10.3 vs 8.7 vs 8.6
runs/sec). Dask and Ray are within ~2 % of each other; the gap to local is
the dispatch-layer overhead each of them pays per task that joblib's loky
backend avoids on a single host. Per-run GMAT load is amortised identically
across all three backends because `reuse_gmat_context=True` keeps a single
gmatpy import alive in each worker for the lifetime of the sweep.

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
