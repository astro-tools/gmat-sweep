# Ray autoscaling

Wire [`RayPool`][gmat_sweep.backends.RayPool] into a Ray cluster brought
up with `ray up cluster.yaml`. The cluster has one head node and an
autoscaling worker pool that grows under sweep load and shrinks when
the queue drains. Each `gmat-sweep` run becomes one Ray task; the
worker hosts that run them are managed by Ray's autoscaler.

## Prerequisites

- A cloud or on-prem provider Ray's autoscaler can drive — AWS, GCP,
  Azure, vSphere, or a [custom node provider](https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html).
  Local clusters work too; `ray up` against a local provider is the
  simplest way to test the recipe end-to-end.
- A container image containing GMAT and `gmat-sweep[ray]` —
  `ghcr.io/astro-tools/gmat`, with a tag matching the GMAT release the
  sweep targets. The same image runs on the head node and every
  worker; see "Worker-image discipline" below.
- A shared filesystem (or object-store path) reachable from every node
  at the same path, holding the script and the `out=` directory.
- `gmat-sweep[ray]` and `ray[default]` installed in the driver env.
  `ray` is **not** a `gmat-sweep` core dependency — install it
  yourself.

## Worked example

### `cluster.yaml`

```yaml
cluster_name: gmat-sweep
provider:
  type: aws
  region: us-east-1
auth:
  ssh_user: ubuntu

available_node_types:
  head:
    resources: {}
    node_config:
      InstanceType: m6i.large
      ImageId: ami-0123456789abcdef0
  worker:
    resources: {}
    node_config:
      InstanceType: m6i.xlarge
      ImageId: ami-0123456789abcdef0
    min_workers: 0
    max_workers: 16

head_node_type: head
docker:
  image: ghcr.io/astro-tools/gmat:<your-tag>
  container_name: ray
  pull_before_run: true

file_mounts:
  /shared: /local/path/to/shared

initialization_commands: []
setup_commands: []

head_start_ray_commands:
  - ray stop
  - ulimit -n 65536; ray start --head --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml --dashboard-host=0.0.0.0
worker_start_ray_commands:
  - ray stop
  - ulimit -n 65536; ray start --address=$RAY_HEAD_IP:6379 --object-manager-port=8076
```

`ray up cluster.yaml` brings up the head node, installs Ray, and pulls
the GMAT image. `min_workers: 0, max_workers: 16` lets the autoscaler
hold zero idle workers and scale up to sixteen as tasks queue. Adjust
the instance types and image to your provider; the GMAT image is the
load-bearing piece.

### Driver

```python
import ray

from gmat_sweep import sweep
from gmat_sweep.backends import RayPool

ray.init(address="ray://<head-public-ip>:10001")

with RayPool() as pool:
    df = sweep(
        "/shared/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/shared/sweeps/sma-scan",
    )

print(df.head())
```

`ray.init(address="ray://...")` connects the driver — typically your
laptop or a CI runner — to the cluster's Ray Client server (port
`10001` by default). `RayPool()` then dispatches tasks to whichever
workers Ray decides to keep around; the autoscaler grows the pool
under load and reaps idle workers a few minutes after the sweep ends.

Because `ray.init` was already called when `RayPool` is constructed,
the pool's `close()` leaves the runtime alone (it only calls
`ray.shutdown()` for runtimes it bootstrapped itself). Running the
driver again — same script, same grid — reuses the same cluster.

### Watching the sweep

The Ray dashboard runs on port `8265` of the head node:
`http://<head-public-ip>:8265`. The Tasks view shows in-flight
`gmat-sweep` tasks; the Cluster view shows the autoscaler bringing
worker nodes up and tearing them down. Pair this with a
`watch -n 5 ls /shared/sweeps/sma-scan/` on the shared filesystem to
watch per-run Parquet files appear.

## Caveats

### Worker-image discipline

Ray serialises the task callable and ships it to workers; if the
driver's `gmat-sweep` and the worker's `gmat-sweep` differ by even a
patch version, deserialisation can succeed silently and produce
inconsistent runs — the manifest's `backend` field doesn't catch this.

Pin one image, one tag, in `cluster.yaml`'s `docker.image`, and use
the **same** image to run the driver if the driver runs in-cluster.
The
[backend equivalence guarantee](../backends.md#backend-equivalence-guarantee)
pins per-backend determinism on a single CI image; it can't pin a
heterogeneous worker pool you assemble yourself.

### Object-store sizing

Ray pre-allocates an in-memory object store on every node — by default
roughly 30% of system memory. Sweeps that produce large per-run
outputs (large Parquet files, sweeps with many time steps) push more
through the object store than a small-output sweep, and Ray will spill
to disk when the in-memory store fills up. Spill is correct but slow;
if you see the dashboard report "spilled X GB" during a sweep, raise
`object_store_memory` in `cluster.yaml`'s worker node config:

```yaml
worker_start_ray_commands:
  - ray stop
  - ulimit -n 65536; ray start --address=$RAY_HEAD_IP:6379 --object-store-memory=8000000000
```

(Eight GB in this example.) The right number is workload-specific —
profile a small sweep first, then size for the real one.

### Subprocess hop inside each Ray task

Each `gmat-sweep` task runs as a Ray actor task. Under the default
`reuse_gmat_context=True`, the task imports `gmat_run` once per worker
and dispatches every subsequent task through
`gmat_sweep.worker.run_one` in the same interpreter. Under
`reuse_gmat_context=False`, the task spawns a child Python via
`gmat_sweep.backends._subprocess.run_spec_in_subprocess` that
bootstraps GMAT fresh and exits — per task, not per worker. Ray's
worker processes themselves are long-lived (the autoscaler manages them
at node granularity, not per task); the isolation contract is the same
one that protects sweeps on every other backend.

The `reuse_gmat_context=True` default still amortises bootstrap across
the runs assigned to a single Ray worker process, so worker-level
reuse is the fast path on Ray too.

### `runtime_env` and Ray's `uv` auto-bootstrap

Recent Ray versions auto-detect a `pyproject.toml`/`uv.lock` near the
driver and try to install a matching env on every worker via
`runtime_env`. For a cluster running a pre-baked GMAT image, that's
unwanted — the image already has the right env, and the auto-bootstrap
re-installs `gmat-sweep` on every worker every time the cluster scales.
`RayPool` disables this hook automatically by setting
`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` at backend-package import time;
if you bypass `RayPool` and call `ray.init` yourself, set the same
env var (`os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"` before
`import ray`) to opt out.

## When this isn't enough

Custom routing (per-task placement constraints, GPU-affinity workers,
multi-tenant priority queues) and any deviation from "one task per
run, run on whichever worker is free" are out of scope for the
`RayPool` recipe. Implement them in a custom `Pool` against the
[`Pool`][gmat_sweep.backends.Pool] ABC; `gmat_sweep/backends/ray.py` is
the working template.
