# Backends

Every gmat-sweep run is dispatched through a `Pool` — the abstraction that
takes a `RunSpec`, runs it, and returns a `RunOutcome`. Three concrete
pools ship in the box:

| Pool | Install | When to pick it |
|---|---|---|
| [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] | core (no extras) | Default. One machine, one Python, joblib's loky workers spawn one fresh interpreter per task. The right choice for nearly every laptop or single-box server sweep. |
| [`DaskPool`][gmat_sweep.backends.DaskPool] | `pip install gmat-sweep[dask]` | Multi-host sweeps, or a sweep that fits on one machine but needs to plug into an existing `dask.distributed` cluster (Slurm, Kubernetes, or a long-lived dev scheduler). |
| [`RayPool`][gmat_sweep.backends.RayPool] | `pip install gmat-sweep[ray]` | Multi-host sweeps on a Ray runtime — local, autoscaling, or remote via the Ray Client. |

All three honour the same per-run subprocess-isolation contract: each
`RunSpec` runs in a freshly-spawned Python interpreter, so the `gmatpy`
bootstrap and GMAT's process-global singletons cannot leak between runs.
Loky gives you that for free; Dask and Ray reuse worker processes by
default, so gmat-sweep adds an explicit subprocess hop inside each task.

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

## Cluster recipes

Slurm `srun` recipes for `DaskPool`, Kubernetes pod-per-worker setup, and
Ray autoscaling cluster configuration are tracked separately and will
land alongside the cluster-recipes documentation page.
