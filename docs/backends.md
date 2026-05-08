# Backends

Every gmat-sweep run is dispatched through a `Pool` — the abstraction that
takes a `RunSpec`, runs it, and returns a `RunOutcome`. Five concrete
pools ship in the box:

| Pool | Install | When to pick it |
|---|---|---|
| [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] | core (no extras) | Default. One machine, one Python, joblib's loky workers spawn one fresh interpreter per task. The right choice for nearly every laptop or single-box server sweep. |
| [`DaskPool`][gmat_sweep.backends.DaskPool] | `pip install gmat-sweep[dask]` | Multi-host sweeps, or a sweep that fits on one machine but needs to plug into an existing `dask.distributed` cluster (Slurm, Kubernetes, or a long-lived dev scheduler). |
| [`RayPool`][gmat_sweep.backends.RayPool] | `pip install gmat-sweep[ray]` | Multi-host sweeps on a Ray runtime — local, autoscaling, or remote via the Ray Client. |
| [`KubernetesJobPool`][gmat_sweep.backends.KubernetesJobPool] | `pip install gmat-sweep[k8s]` | Native Kubernetes — every run becomes one `Job`, every Pod is a fresh interpreter. Pick this when you want the cluster to schedule work directly without a Dask or Ray middleware layer. |
| [`MPIPool`][gmat_sweep.backends.MPIPool] | `pip install gmat-sweep[mpi]` (plus a system MPI install) | HPC clusters that already speak MPI — Slurm allocations, mvapich2 / Intel MPI / Open MPI runtimes. Wraps `mpi4py.futures.MPIPoolExecutor`; works with both dynamic-spawn and pre-allocated-rank launches. |

All five accept the same `reuse_gmat_context` keyword controlling how the
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

## `KubernetesJobPool` — native Kubernetes Jobs

```python
from gmat_sweep import sweep
from gmat_sweep.backends import KubernetesJobPool

with KubernetesJobPool(
    image="ghcr.io/your-org/gmat-sweep:<your-tag>",
    pvc_name="gmat-sweep-shared",
    parallelism=32,
) as pool:
    df = sweep(
        "/sweep/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/sweep/sweeps/sma-scan",
    )
```

Each run becomes one `batch/v1` Job. Pods read the spec and write the
outcome through a shared `PersistentVolumeClaim` mounted at the same
path on the driver and the workers. `parallelism=` caps the in-flight
Job count so a 10000-run sweep doesn't stampede the API server. Per-run
resource overrides are supported via the `resources=` kwarg in either
mapping or callable form.

See the [`KubernetesJobPool` recipe](recipes/kubernetes-jobpool.md) for
the full setup: image build, PVC layout, in-cluster vs.
out-of-cluster auth, and the `resources=` knob.

## `MPIPool` — `mpi4py.futures`

`MPIPool` wraps [`mpi4py.futures.MPIPoolExecutor`][mpi4py-futures]. The
`[mpi]` extra pulls in the `mpi4py` Python bindings; the **system MPI
runtime** (Open MPI / Intel MPI / mvapich2) must already be installed
and on `PATH` — `pip install gmat-sweep[mpi]` does **not** install
`mpirun` itself.

[mpi4py-futures]: https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html

`MPIPoolExecutor` supports two launch modes natively, and `MPIPool` does
not second-guess upstream's mode detection. Both work without any
configuration on this side.

### Dynamic spawn — laptop / CI / dev runs

```python
from gmat_sweep import sweep
from gmat_sweep.backends import MPIPool

with MPIPool(max_workers=4) as pool:
    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="./sweep",
    )
```

```bash
gmat-sweep run --backend mpi --backend-arg max_workers=4 \
    --grid "Sat.SMA=7000:8000:5" --out ./sweep mission.script
```

In this mode the executor calls `MPI_Comm_spawn` to launch
`max_workers` worker ranks on demand. No `mpirun` wrapping the driver
is required, but the MPI runtime must be installed locally.

### Pre-allocated ranks — SLURM / HPC

```bash
mpirun -n 8 python -m mpi4py.futures -m gmat_sweep run \
    --backend mpi --grid "Sat.SMA=7000:8000:5" \
    --out ./sweep mission.script
```

Under the `python -m mpi4py.futures` launcher shim, ranks 1..K-1 enter
`mpi4py.futures`'s worker loop *inside the shim*; rank 0 runs
`gmat_sweep` exactly once with no awareness that MPI is involved.
`max_workers` is then optional — it defaults to K-1.

### `--workers` and MPI

`--workers N` is **silently ignored** under `--backend mpi` — rank
count is set either by `mpirun -n K` (pre-allocated mode) or by
`--backend-arg max_workers=N` (dynamic-spawn mode). This matches the
existing behaviour where `--workers` is forwarded to a backend's
canonical kwarg name (`n_workers` for Dask, `num_cpus` for Ray) and
otherwise has no effect.

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
