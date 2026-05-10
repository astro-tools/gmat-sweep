# Backends

Every gmat-sweep run is dispatched through a `Pool` — the abstraction that
takes a `RunSpec`, runs it, and returns a `RunOutcome`. Five concrete
pools ship in the box:

| Pool | Install | When to pick it |
|---|---|---|
| [`LocalJoblibPool`][gmat_sweep.LocalJoblibPool] | core (no extras) | Default. One machine, one Python, joblib's loky workers spawn one fresh interpreter per task. The right choice for nearly every laptop or single-box server sweep. |
| [`ProcessPoolExecutorPool`][gmat_sweep.backends.ProcessPoolExecutorPool] | core (no extras), Python 3.11+ | Stdlib alternative to `LocalJoblibPool`. Wraps `concurrent.futures.ProcessPoolExecutor` with `max_tasks_per_child=1`, so every task runs in a fresh interpreter by construction. Pick when avoiding the `joblib` runtime dependency matters. |
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
    backend=LocalJoblibPool(max_workers=4),
    out="./sweep",
)
```

`max_workers=-1` (the default) uses every core. `workers=` is accepted as
a deprecated alias and emits a `DeprecationWarning`. See
[Choosing a backend](parameter-spec.md#choosing-a-backend) on the parameter
spec page for the full set of LocalJoblibPool patterns (capping
parallelism, sharing one pool across several sweeps).

## `ProcessPoolExecutorPool` — stdlib opt-in

```python
from gmat_sweep import sweep
from gmat_sweep.backends import ProcessPoolExecutorPool

with ProcessPoolExecutorPool(max_workers=4) as pool:
    df = sweep(
        "mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="./sweep",
    )
```

`ProcessPoolExecutorPool` requires Python 3.11+ — `max_tasks_per_child`
landed in `concurrent.futures.ProcessPoolExecutor` in that release.
Importing the pool on Python 3.10 raises `RuntimeError` immediately,
pointing at `LocalJoblibPool` as the 3.10-compatible path.

Each task runs in a fresh worker interpreter by construction
(`max_tasks_per_child=1`), so gmatpy bootstraps once per task.
That makes this the natural choice when avoiding the `joblib` runtime
dependency matters; for a sweep where many runs share the same script,
`LocalJoblibPool`'s reuse path amortises the bootstrap and finishes
faster.

`reuse_gmat_context` is accepted for `Pool` API parity but has no
practical effect on this backend — `max_tasks_per_child=1` already gives
every task a fresh interpreter, so both modes dispatch `run_one`
directly without nesting through a second subprocess.

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
mapping or callable form. A Job that has not reached a terminal status
within `job_deadline_seconds` (default 1 h) is deleted by the driver
and folded into a synthetic `RunOutcome.failed` rather than hanging the
sweep on a stuck-`Pending` Pod; closing the pool mid-sweep deletes any
remaining in-flight Jobs so they don't orphan against the namespace
quota.

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

## `DebugPool` — in-process, single-run, debugger-friendly

`DebugPool` is the off-spec backend. Every spec runs on the driver process
— no subprocess, no parallelism — so a `breakpoint()` placed in user code,
override application, or `gmat_run` itself drops directly into the driver's
debugger and IDE step-through Just Works. **Production sweeps must not use
it**: GMAT's process-global singletons get dirtied by the run, so the pool
accepts exactly one spec and refuses any sweep that submits more.

The isolation violation is the feature, but it is gated behind two
opt-ins: `DebugPool(allow_unisolated_pool=True)` to construct, and
`Sweep(..., allow_unisolated_pool=True)` to dispatch through. Either flag
missing raises `BackendError`. The high-level `sweep()` / `monte_carlo()`
/ `latin_hypercube()` entry points do not surface the flag — drive
`DebugPool` through the `Sweep` class directly.

```python
from pathlib import Path

from gmat_sweep import RunSpec, Sweep
from gmat_sweep.backends.debug import DebugPool

out = Path("./debug-run")
out.mkdir(exist_ok=True)
spec = RunSpec(
    script_path=Path("mission.script"),
    overrides={"Sat.SMA": 7100.0},
    output_dir=out / "run_0",
    run_id=0,
    seed=None,
    run_options={},
)

with DebugPool(allow_unisolated_pool=True) as pool:
    Sweep(
        runs=[spec],
        backend=pool,
        manifest_path=out / "manifest.jsonl",
        output_dir=out,
        script_path=spec.script_path,
        parameter_spec={"_kind": "explicit", "columns": ["Sat.SMA"], "rows": [[7100.0]]},
        allow_unisolated_pool=True,
    ).run()
```

Drop a `breakpoint()` anywhere downstream of `Sweep.run()` — for example
inside the `.script`-driven mission, or in a `gmat_run` plugin — and the
driver's debugger catches it. Switch back to `LocalJoblibPool` (or any
other isolated pool) the moment you want N > 1 runs.

## Failed runs

A single failed run never aborts the sweep, regardless of backend. The
worker subprocess catches the exception, the outcome lands in the
manifest with `status="failed"`, and the aggregated DataFrame gets one
NaN-filled row with `__status="failed"`. The
[killed-sweep recovery example](examples/03_killed_sweep_recovery.ipynb)
shows the resume flow end-to-end.

The same row-not-raise contract holds for *transport-level* failures
that escape the worker entirely — a loky / Ray / Dask / MPI worker
process dying mid-task, a `RayTaskError` from a remote-side raise,
`BrokenProcessPool` from a `ProcessPoolExecutor` worker crash, an MPI
rank disappearing under a `SIGSEGV`, a Kubernetes Pod evicted before it
writes its outcome JSON. Every pool catches the exception at the drain
site and folds it into a synthetic `RunOutcome.failed` whose `stderr`
carries the captured traceback, so a single bad worker never aborts the
sweep.

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
