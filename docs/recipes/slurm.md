# Slurm with `srun`

Wire [`DaskPool`][gmat_sweep.backends.DaskPool] into a Slurm allocation
via [`dask-jobqueue`](https://jobqueue.dask.org/)'s `SLURMCluster`. Each
Dask worker becomes one Slurm task, the sweep dispatches across them, and
the per-run subprocess isolation that `gmat-sweep` already does on a
laptop carries over node-by-node unchanged.

## Prerequisites

- A Slurm cluster you can submit to (`sbatch`, `srun`, `squeue` on PATH).
- A shared filesystem (NFS, Lustre, GPFS, …) mounted at the same path on
  the login node and every compute node. The driver writes the manifest
  there and every worker reads/writes the per-run output dir from there.
- A working GMAT install reachable on every compute node — either the
  same shared filesystem or a node-local install at the same path. GMAT
  discovery is done by [`gmat-run`](https://github.com/astro-tools/gmat-run);
  see its install guide for the search order. `gmat-sweep` does not
  install GMAT.
- Python with `gmat-sweep[dask]` and `dask-jobqueue` installed in an env
  reachable from the worker nodes (a shared `venv`, `conda` env, or
  `module load` line is the usual choice). `dask-jobqueue` is **not** a
  `gmat-sweep` dependency — install it yourself.

## Worked example

### `driver.py`

```python
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from gmat_sweep import sweep
from gmat_sweep.backends import DaskPool

cluster = SLURMCluster(
    cores=1,
    processes=1,
    memory="4 GB",
    walltime="01:00:00",
    job_extra_directives=["--ntasks=1", "--cpus-per-task=1"],
    local_directory="/scratch/$USER/dask-worker-space",
    log_directory="/scratch/$USER/dask-logs",
)
cluster.scale(jobs=8)
client = Client(cluster)

with DaskPool(client=client) as pool:
    df = sweep(
        "/shared/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/shared/sweeps/sma-scan",
    )

print(df.head())
```

`SLURMCluster` submits one Slurm job per Dask worker; `cluster.scale(jobs=8)`
asks for eight. The worker pool is elastic — Dask drops idle workers back
to Slurm and re-requests them as new specs arrive — so the same driver
also works for shorter or much longer sweeps without rewiring.

### Submitting the driver

The driver itself is a single Python process. Submit it as a one-task
allocation that lives long enough to manage the worker pool:

```bash
#!/bin/bash
#SBATCH --job-name=gmat-sweep-driver
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/sweeps/sma-scan/driver.log

srun python -u driver.py
```

`sbatch driver.sbatch` queues it; `squeue -u $USER` shows the driver job
plus the eight worker jobs as Dask scales them up.

## Caveats

### Shared filesystem is non-negotiable

Every worker reads the script, writes per-run Parquet outputs, and
appends manifest entries through the same `out=` directory. That
directory **must** be visible at the same path from the driver and from
every compute node. NFS, Lustre, BeeGFS, and GPFS all work; node-local
SSD scratch does not unless you stage the result back at the end.

If `out=` resolves to a different location on different nodes, runs will
appear to "succeed" but their Parquet files end up scattered across
worker-local disks and the aggregated DataFrame will be empty or partial.

### GMAT must be reachable from every worker

`gmat-run` discovers a local GMAT install at import time. Each Dask
worker is a fresh Python process on a compute node, so each worker has
to find GMAT independently. Two patterns work:

1. Shared-filesystem install — a single GMAT tree under `/shared/gmat/`
   visible to every node. Set `GMAT_BIN_DIR` (or whatever discovery hook
   `gmat-run` exposes) in the worker environment.
2. Node-local install at the same canonical path on every node.

Both are operationally fine; pick whichever your cluster admins prefer.
The discovery itself is `gmat-run`'s problem, not `gmat-sweep`'s — but
a misconfigured worker shows up as every run failing with the same
import error, so it's worth checking in a `srun python -c "import
gmat_run"` before launching the sweep.

### Subprocess isolation still applies inside each worker

Each Dask worker is one Python process. Under
`reuse_gmat_context=True` (the default) each worker imports `gmat_run`
once and dispatches each task through `gmat_sweep.worker.run_one` in
the same interpreter. Under `reuse_gmat_context=False` each task spawns
a child Python via `gmat_sweep.backends._subprocess.run_spec_in_subprocess`
that bootstraps GMAT fresh, runs the script, and exits. No special
Slurm-side configuration is needed in either mode; the per-task fresh
GMAT context that protects sweeps on a laptop protects them on Slurm
in exactly the same way.

The `reuse_gmat_context=True` default still amortises bootstrap across
the runs assigned to a single worker, so worker-level reuse remains the
fast path on Slurm too.

## When this isn't enough

The recipe above covers the common Slurm-via-`dask-jobqueue` setup. For
exotic schedulers, MPI-style launches, or Slurm features that don't fit
the `SLURMCluster` model, write a custom `Pool` against the
[`Pool`][gmat_sweep.backends.Pool] ABC — its only requirement is that
each task runs through the subprocess hop.
