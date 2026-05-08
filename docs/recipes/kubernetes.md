# Kubernetes pod-per-worker

!!! note "Two Kubernetes paths"
    There are two ways to run a sweep on Kubernetes. **This page** uses
    [`DaskPool`][gmat_sweep.backends.DaskPool] over `dask-kubernetes` —
    a Dask scheduler manages a fixed (or autoscaling) pool of worker
    Pods. The companion page
    [Kubernetes Job-per-run](kubernetes-jobpool.md) uses
    [`KubernetesJobPool`][gmat_sweep.backends.KubernetesJobPool] —
    every run becomes one `batch/v1` Job, no Dask layer. Pick the
    Dask path when your stack already wants a Dask client; pick the
    Job path when you want native cluster scheduling and one less
    middleware layer.

Wire [`DaskPool`][gmat_sweep.backends.DaskPool] into a Kubernetes
cluster via [`dask-kubernetes`](https://kubernetes.dask.org/). Each
Dask worker becomes one Pod, the sweep dispatches across them, and Pod
lifecycle (creation, eviction, restart) is handled by the Dask
Operator.

The Operator path is the recommended setup — `dask-kubernetes` ships a
`KubeCluster` constructor that talks to the operator's CRDs, so you
don't hand-write Pod YAML for each cluster. A pure `kubectl apply`
flow is possible (see "Manual YAML" at the bottom) but is more work
and more error-prone.

## Prerequisites

- A reachable Kubernetes cluster — managed (GKE, EKS, AKS, …) or
  self-hosted. `kubectl get nodes` should succeed from the machine
  running the driver.
- The [Dask Operator](https://kubernetes.dask.org/en/latest/operator_installation.html)
  installed in the cluster (one-time `helm install` per cluster).
- A container image containing GMAT and `gmat-sweep[dask]`. The
  canonical image is `ghcr.io/astro-tools/gmat`; pin a tag matching the
  GMAT release you want every worker to run. (Mismatched worker images
  silently produce inconsistent sweeps — see "Image discipline" below.)
- A `PersistentVolumeClaim` (or equivalent: EFS, GCS Fuse, Azure Files)
  mounted at the same path in every worker Pod, holding the script and
  the `out=` directory.
- Python with `gmat-sweep[dask]` and `dask-kubernetes` installed in
  the driver env. Neither is a `gmat-sweep` dependency.

## Worked example

### Operator-mode driver

```python
from dask.distributed import Client
from dask_kubernetes.operator import KubeCluster, make_cluster_spec

from gmat_sweep import sweep
from gmat_sweep.backends import DaskPool

spec = make_cluster_spec(
    name="gmat-sweep",
    image="ghcr.io/astro-tools/gmat:<your-tag>",
    n_workers=8,
    resources={"requests": {"cpu": "1", "memory": "4Gi"}},
)
spec["spec"]["worker"]["spec"]["volumes"] = [
    {"name": "shared", "persistentVolumeClaim": {"claimName": "gmat-shared"}},
]
spec["spec"]["worker"]["spec"]["containers"][0]["volumeMounts"] = [
    {"name": "shared", "mountPath": "/shared"},
]
spec["spec"]["scheduler"]["spec"]["volumes"] = spec["spec"]["worker"]["spec"]["volumes"]
spec["spec"]["scheduler"]["spec"]["containers"][0]["volumeMounts"] = (
    spec["spec"]["worker"]["spec"]["containers"][0]["volumeMounts"]
)

cluster = KubeCluster(custom_cluster_spec=spec)
client = Client(cluster)

with DaskPool(client=client) as pool:
    df = sweep(
        "/shared/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/shared/sweeps/sma-scan",
    )

print(df.head())
cluster.close()
```

`KubeCluster(custom_cluster_spec=spec)` creates a `DaskCluster` CRD; the
operator reconciles it into a scheduler Pod and `n_workers` worker
Pods. `cluster.close()` deletes the CRD and the operator garbage-collects
the Pods.

You can scale the cluster mid-sweep — `cluster.scale(16)` — but for a
fixed-size sweep the constructor's `n_workers` is enough. For
autoscaling, set `cluster.adapt(minimum=2, maximum=16)` after the
`Client` is wired up.

### Running the driver

The driver itself can run anywhere with `kubectl` access — your laptop,
a CI runner, or a small pod inside the same cluster. Running it
in-cluster avoids the Dask Client → scheduler hop crossing the cluster
boundary, which simplifies networking and is often noticeably faster.
A minimal driver Pod looks like:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gmat-sweep-driver
spec:
  serviceAccountName: gmat-sweep-driver  # has Dask Operator API perms
  restartPolicy: Never
  containers:
    - name: driver
      image: ghcr.io/astro-tools/gmat:<your-tag>
      command: ["python", "-u", "/shared/drivers/driver.py"]
      volumeMounts:
        - { name: shared, mountPath: /shared }
  volumes:
    - name: shared
      persistentVolumeClaim:
        claimName: gmat-shared
```

`kubectl apply -f driver.yaml` queues it. The driver Pod and the
worker Pods all share the same `gmat-shared` PVC, so the script path
and `out=` path resolve identically on every side.

## Caveats

### Image discipline — every Pod runs the same image

Dask serialises the sweep's task callable on the driver and sends it
to workers; the workers must `pickle`-load it under the same
`gmat-sweep` and `gmat-run` versions or the call fails. **The driver
image and the worker image must be the same image, the same tag.** The
[`backend equivalence guarantee`](../backends.md#backend-equivalence-guarantee)
pins this property in CI on a single backend, but it can't catch a
worker pool that's silently running a different image.

The simplest safe pattern: `make_cluster_spec(image="...")` with a
fully-pinned tag, and the driver Pod manifest uses the same string.

### Storage — shared volume at the same path

Same constraint as the [Slurm recipe](slurm.md#shared-filesystem-is-non-negotiable),
expressed Kubernetes-side: the `out=` directory must be on a
`ReadWriteMany` (or appropriately mounted `ReadWriteOncePod` per node)
PVC visible to every worker Pod and the driver. Storage classes that
work in practice include EFS (AWS), Filestore (GCP), Azure Files,
NFS-CSI on self-hosted clusters, and GCS Fuse via the GCS CSI driver.

`emptyDir` and per-Pod `hostPath` mounts do **not** satisfy this — a
sweep run on `emptyDir` writes per-run Parquet files to whichever Pod
ran the run, and the aggregated DataFrame back at the driver will be
empty.

### Image pull policy

For a sweep that scales out under sustained load, set
`imagePullPolicy: IfNotPresent` on the worker container so re-spawned
Pods reuse a cached image instead of re-pulling on every Pod start.
The default `Always` is fine for development but adds 10–60 s to every
new worker on most registries. The image is identified by a fully-pinned
tag, so `IfNotPresent` is safe — there's no "latest" drift to worry
about.

### Pod eviction during a sweep

Kubernetes can evict a worker Pod for any of the usual reasons —
preemption, node draining, OOM, voluntary scale-down. The Dask
scheduler reassigns in-flight tasks to a surviving worker and the
sweep continues, but a task that was mid-flight at eviction time
records as a failure in the manifest.

Recovery is the standard `gmat-sweep` resume flow — the manifest
persists across the eviction (it's on the shared PVC, not the evicted
Pod's local disk) and `Sweep.from_manifest(...).resume()` re-runs the
failed entries on whatever worker pool is healthy at resume time. See
[Resume](../resume.md) for the full call.

## Manual YAML — when the operator isn't an option

Some clusters can't install operators — locked-down policy, no Helm,
shared cluster with strict CRD review. In that case, the older
`KubeCluster.from_yaml(pod_spec.yaml)` path still works against
hand-written Pod templates, no operator required. The trade-off is that
you maintain the full Pod spec (resources, volumes, security context,
labels) yourself, and you lose autoscaling. Prefer the operator path
unless your cluster forbids it.

## When this isn't enough

Multi-region failover, custom CSI drivers with side-effects on `out=`,
or per-Pod side cars (logging, monitoring) that have to run alongside
the worker — all of those exit the recipe and become custom
`Pool` work against the [`Pool`][gmat_sweep.backends.Pool] ABC. The
`DaskPool` source under `gmat_sweep/backends/dask.py` is a working
template.
