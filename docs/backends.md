# Backends

Every gmat-sweep run is dispatched through a `Pool` — the abstraction that
takes a `RunSpec`, runs it, and returns a `RunOutcome`. Three concrete
pools ship in the box:

| Pool | Install | When to pick it |
|---|---|---|
| [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] | core (no extras) | Default. One machine, one Python, joblib's loky workers spawn one fresh interpreter per task. The right choice for nearly every laptop or single-box server sweep. |
| [`DaskPool`][gmat_sweep.backends.DaskPool] | `pip install gmat-sweep[dask]` | Multi-host sweeps, or a sweep that fits on one machine but needs to plug into an existing `dask.distributed` cluster (Slurm, Kubernetes, or a long-lived dev scheduler). |
| [`RayPool`][gmat_sweep.backends.RayPool] | `pip install gmat-sweep[ray]` | Multi-host sweeps on a Ray runtime — local, autoscaling, or remote via the Ray Client. |

All three accept the same `reuse_gmat_context` keyword controlling how the
GMAT bootstrap cost is amortised across the runs in a sweep:

- `reuse_gmat_context=True` (the default) — a worker process imports
  `gmat_run` once and reuses the resulting state across many tasks. Bootstrap
  cost is paid once per worker, then amortised. **Safe only when every task
  dispatched through the pool loads the same script** — GMAT relies on
  process-global singletons that cannot be reused across runs that load
  different scripts.
- `reuse_gmat_context=False` — every task spawns a fresh Python interpreter
  that bootstraps `gmatpy` from scratch. Slower but supports arbitrary
  heterogeneous scripts in a single sweep.

The default is right for the common case (one mission, many parameter
combinations). Pass `reuse_gmat_context=False` when you compose a single pool
across calls that load different `.script` files.

## `LocalJoblibPool` — the default

```python
from gmat_sweep import LocalJoblibPool, sweep

df = sweep(
    "mission.script",
    grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
    backend=LocalJoblibPool(workers=4),
    out="./sweep",
)
```

`workers=-1` (the default) uses every core. See
[Choosing a backend](parameter-spec.md#choosing-a-backend) on the parameter
spec page for the full set of LocalJoblibPool patterns (capping
parallelism, sharing one pool across several sweeps).

## `DaskPool` — `dask.distributed`

```python
from gmat_sweep import sweep
from gmat_sweep.backends import DaskPool

with DaskPool(n_workers=4) as pool:
    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="./sweep",
    )
```

With no arguments, `DaskPool` spawns a `distributed.LocalCluster` and a
`Client` connected to it, and tears both down on `close()`. Pass an
existing client to dispatch through a cluster the rest of your code is
already using:

```python
from dask.distributed import Client
from gmat_sweep import sweep
from gmat_sweep.backends import DaskPool

client = Client("tcp://scheduler:8786")
with DaskPool(client=client) as pool:
    df = sweep("mission.script", grid={...}, backend=pool, out="./sweep")
# `client` is still open — DaskPool only closes resources it created.
```

## `RayPool` — Ray

```python
from gmat_sweep import sweep
from gmat_sweep.backends import RayPool

with RayPool(num_cpus=4) as pool:
    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="./sweep",
    )
```

`RayPool` calls `ray.init` for you when constructed. Pass an `address`
to connect to a pre-existing cluster instead — ``"auto"`` for a local
runtime started elsewhere on the same machine, or
``"ray://host:port"`` for a remote Ray Client server:

```python
from gmat_sweep.backends import RayPool

with RayPool(address="ray://head:10001") as pool:
    df = sweep("mission.script", grid={...}, backend=pool, out="./sweep")
```

`RayPool` only calls `ray.shutdown()` on `close()` if its own `__init__`
was what initialised the runtime. If you called `ray.init()` yourself
before constructing the pool, the pool leaves your runtime alone.

## Failed runs

A single failed run never aborts the sweep, regardless of backend. The
worker subprocess catches the exception, the outcome lands in the
manifest with `status="failed"`, and the aggregated DataFrame gets one
NaN-filled row with `__status="failed"`. The
[killed-sweep recovery example](examples/03_killed_sweep_recovery.ipynb)
shows the resume flow end-to-end.

## Backend equivalence guarantee

Every backend is required to produce bit-equal DataFrames and bit-equal
`parameter_spec` / per-`run_id` `overrides` for the same sweep — only the
manifest's `backend` header field is allowed to differ. The contract is
enforced by `tests/test_backend_equivalence.py`, which runs a 16-run grid
sweep, a 32-run Monte Carlo sweep, and a 16-run Latin hypercube sweep on
each backend and asserts every non-`LocalJoblibPool` backend matches the
local-backend reference. The Monte Carlo sweep also pins cross-process
determinism on `DaskPool` (a fresh driver-process Python re-runs the same
sweep and the result must compare bit-equal). The suite is gated as
`integration and slow` and runs on a dedicated Linux / Python 3.12 / GMAT
R2026a CI cell on every PR.

## Cluster recipes

Worked examples for wiring `DaskPool` and `RayPool` into shared cluster
infrastructure live under [Recipes](recipes/index.md). One page per
orchestrator:

- [Slurm with `srun`](recipes/slurm.md) — `DaskPool` over
  `dask-jobqueue.SLURMCluster`, with the `sbatch` allocation and the
  driver script.
- [Kubernetes pod-per-worker](recipes/kubernetes.md) — `DaskPool` over
  the `dask-kubernetes` operator, with a `KubeCluster` spec and
  shared-PVC layout.
- [Ray autoscaling](recipes/ray-autoscaling.md) — `RayPool` over
  `ray up cluster.yaml`, with an autoscaling worker pool and the Ray
  Client driver.

If your orchestrator isn't on the list, the [`Pool`][gmat_sweep.backends.Pool]
ABC is the escape hatch — implement it once and any `sweep()` call
dispatches through it.
