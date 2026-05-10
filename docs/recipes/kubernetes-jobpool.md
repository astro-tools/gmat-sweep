# Kubernetes Job-per-run

Wire [`KubernetesJobPool`][gmat_sweep.backends.KubernetesJobPool] into a
Kubernetes cluster. Every run becomes one `batch/v1` Job, every Pod is
a fresh Python interpreter, and the cluster's scheduler decides where
each Job lands. There is no Dask or Ray middleware — the pool talks to
the Kubernetes API directly.

!!! note "Two Kubernetes paths"
    The companion recipe [Kubernetes pod-per-worker](kubernetes.md) wires
    `DaskPool` into a Dask-managed pool of worker Pods. Pick that when
    your stack already wants a Dask client; pick this page when you
    want native scheduling and one less middleware layer.

## Prerequisites

- A reachable Kubernetes cluster — managed (GKE, EKS, AKS, …),
  self-hosted, or local (kind, k3d, Docker Desktop). `kubectl get nodes`
  should succeed from the machine running the driver.
- A container image with `gmat-sweep[k8s]` plus a working GMAT install.
  See [Image](#image) for the build pattern.
- A `PersistentVolumeClaim` visible to both the driver and the workers
  at the same path. Pure `ReadWriteMany` (EFS, Filestore, Azure Files,
  NFS-CSI, GCS Fuse) is the typical answer; a `ReadWriteOncePod` PVC
  works when the driver runs as a Pod on the same node as the workers.
- Python with `gmat-sweep[k8s]` installed in the driver env.

## Image

`KubernetesJobPool` requires `image=` — the pool ships no default. The
canonical image at `ghcr.io/astro-tools/gmat:<tag>` carries GMAT and
`gmatpy` but **not** `gmat-sweep` itself — downstream consumers add
their own packages on top:

```dockerfile
FROM ghcr.io/astro-tools/gmat:R2026a

RUN python3.12 -m pip install --no-cache-dir "gmat-sweep[k8s]==<your-version>"
```

Pin both the GMAT image tag and the `gmat-sweep` version. A mismatch
between driver and worker `gmat-sweep` versions is a serialisation hazard
the Pool ABC's bit-equal contract cannot recover from.

Push the image somewhere both the cluster and CI can reach
(`ghcr.io/<your-org>/gmat-sweep`, ECR, GCR, …). Pass the fully-pinned
reference to `KubernetesJobPool(image=...)`.

## PVC layout

Per-run spec / outcome JSON files travel through the PVC, not over the
Kubernetes API. The pool writes
`<driver_mount_path>/_specs/<run_id>.json` on the driver side; the Pod
reads `<pvc_mount_path>/_specs/<run_id>.json`. Outcome JSON travels back
the same way under `_outcomes/`. Per-run Parquet outputs go into the
`out=` directory of the sweep.

If the driver runs as a Pod, mount the PVC at the same path as the
workers (`/sweep` is the default). Then `pvc_mount_path` and
`driver_mount_path` are the same string and you don't pass
`driver_mount_path` at all.

If the driver runs out-of-cluster (laptop, CI runner), mount the same
storage on the driver side at the same path the Pods see — typically
via an NFS / CIFS mount that exports the PVC's backing storage. Setting
`driver_mount_path` to a different local path is supported, but
`RunSpec.script_path` and `RunSpec.output_dir` must already be Pod-side
paths in either case — the pool does not rewrite them.

A minimal `PersistentVolume` + `PersistentVolumeClaim` pair backed by
NFS:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: gmat-sweep-pv
spec:
  capacity:
    storage: 100Gi
  accessModes: ["ReadWriteMany"]
  nfs:
    server: nfs.internal
    path: /export/gmat-sweep
  storageClassName: gmat-sweep
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gmat-sweep
spec:
  accessModes: ["ReadWriteMany"]
  resources:
    requests:
      storage: 100Gi
  storageClassName: gmat-sweep
  volumeName: gmat-sweep-pv
```

## Authentication

`KubernetesJobPool` autodetects between the two auth paths:

- **In-cluster** — the driver runs as a Pod with a ServiceAccount that
  has RBAC to create/list/watch `batch/v1` Jobs and read Pod logs in the
  target namespace. The pool calls `load_incluster_config()`. Triggered
  when `/var/run/secrets/kubernetes.io/serviceaccount/token` exists.
- **Out-of-cluster** — the driver runs anywhere with a kubeconfig. The
  pool calls `load_kube_config(config_file=...)`, defaulting to
  `~/.kube/config`. Triggered when the in-cluster token file is absent.

Force one path explicitly with `in_cluster=True` / `in_cluster=False`.
Pass `kubeconfig=` to point at a non-default kubeconfig path.

A minimal `Role` for the in-cluster path:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gmat-sweep-driver
  namespace: <your-namespace>
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
```

## Worked example

### In-cluster driver

```python
from gmat_sweep import sweep
from gmat_sweep.backends import KubernetesJobPool

with KubernetesJobPool(
    image="ghcr.io/your-org/gmat-sweep:0.4.0-gmat-R2026a",
    pvc_name="gmat-sweep",
    pvc_mount_path="/sweep",
    namespace="missions",
    parallelism=32,
    default_resources={"cpu": "1", "memory": "4Gi"},
) as pool:
    df = sweep(
        "/sweep/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/sweep/sweeps/sma-scan",
    )

print(df.head())
```

The driver runs as a Pod whose `serviceAccountName` is bound to the
`gmat-sweep-driver` Role above. `parallelism=32` caps in-flight Jobs at
32; the cluster decides where each Pod lands.

### Out-of-cluster driver

```python
from gmat_sweep import sweep
from gmat_sweep.backends import KubernetesJobPool

with KubernetesJobPool(
    image="ghcr.io/your-org/gmat-sweep:0.4.0-gmat-R2026a",
    pvc_name="gmat-sweep",
    pvc_mount_path="/sweep",
    driver_mount_path="/mnt/gmat-sweep",
    in_cluster=False,
    parallelism=8,
) as pool:
    df = sweep(
        "/sweep/missions/mission.script",
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        backend=pool,
        out="/sweep/sweeps/sma-scan",
    )
```

The driver mounts the same NFS export at `/mnt/gmat-sweep`; the Pods
mount the PVC at `/sweep`. Paths inside the script (`script_path`,
`output_dir`) are Pod-side absolute paths.

## Per-run resources

`resources=` accepts a mapping keyed by `run_id` or a callable
`(RunSpec) -> dict`. Both compose with `default_resources=`:

```python
# Per-run map
KubernetesJobPool(
    ...,
    default_resources={"cpu": "1", "memory": "4Gi"},
    resources={
        17: {"cpu": "8", "memory": "16Gi"},        # one heavy run
        42: {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
    },
)

# Callable
def by_spec(spec):
    if spec.overrides.get("Sat.SMA", 0) > 30000:
        return {"cpu": "4", "memory": "8Gi"}
    return None  # fall through to default_resources

KubernetesJobPool(..., resources=by_spec, default_resources={"cpu": "1"})
```

Each resolved value populates `V1ResourceRequirements.requests`. Pass a
dict with both `requests` and `limits` keys to set limits explicitly:

```python
resources={
    0: {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "8Gi"},
    }
}
```

## Local development with kind

For iterating on the pool itself, [kind](https://kind.sigs.k8s.io/) is
the lightest cluster:

```bash
kind create cluster --name gmat-sweep
docker build -t gmat-sweep:dev -f Dockerfile .
kind load docker-image gmat-sweep:dev --name gmat-sweep
kubectl apply -f sweep-pvc.yaml
```

The `extraMounts` field in `kind-config.yaml` bind-mounts a host
directory into the kind node, so a hostPath PV on the node and the
driver-side path resolve to the same files. See `tests/k8s/` for the
configs used in CI.

## Caveats

### `backoffLimit=0` — no silent retries

The pool sets `backoffLimit=0` so a Pod failure maps 1:1 to a
`RunOutcome.failed`. Setting it higher would let Kubernetes silently
re-run a failed Pod, which breaks gmat-sweep's failure-as-row contract.
If you need retries, do them at the sweep layer with
`Sweep.from_manifest(...).resume()`.

### `ttlSecondsAfterFinished=300`

Completed Jobs auto-GC after 5 minutes. Long enough for a kubectl
inspection window, short enough to not leak Job objects across long
sweeps. Override via `ttl_seconds_after_finished=` if you need more.

### `job_deadline_seconds=3600`

Driver-side wall-clock deadline per Job. A Job that has not reached a
terminal status (`succeeded` or `failed`) within this many seconds is
deleted (`propagationPolicy=Background`) and folded into a synthetic
`RunOutcome.failed`. The check exists to break the otherwise-silent
hang on Pods stuck in `Pending` / `ImagePullBackOff` / `Unschedulable`,
which never produce a status event and so never satisfy the watch
loop. Granularity is bounded by the watch reconnect cadence (~60 s);
the deadline is a hang preventer, not a tight SLA. Override via
`job_deadline_seconds=` for shorter (test) or longer (multi-hour
solver) runs.

### `close()` deletes in-flight Jobs

Calling `close()` (whether from a `with` block exit, a `Ctrl-C`, or
explicitly mid-sweep) issues a background-propagation delete for
every Job still in flight at close time. Without this, an aborted
sweep would leave Pods running until the TTL kicked in *after* they
finished — burning your namespace quota and your bill. `ApiException`
on a per-Job delete is swallowed so a Job already reaped by the TTL
controller (or by a concurrent kubectl) doesn't propagate out.

### Pod failure modes

A Pod that finished but didn't write outcome JSON (OOMKill, eviction,
image pull failure) is folded into a synthetic `RunOutcome.failed`
carrying the Pod's logs as `stderr`. Same posture every other backend
takes for transport-layer failures.

### Image discipline

The driver and the Pods must run the same `gmat-sweep` version, the
same `gmat-run` version, and the same image — same as the dask-kubernetes
recipe documents. A pinned image tag is the simplest enforcement.

### Storage shape

The PVC must be visible to every worker Pod **and** to the driver. A
`ReadWriteOnce` PVC with a single-node cluster works in development;
production typically wants `ReadWriteMany` (EFS, Filestore, Azure Files,
NFS-CSI, GCS Fuse) so worker Pods can land anywhere.

## When this isn't enough

CRD-driven sweep management, custom CSI drivers, multi-cluster
federation, or per-Pod sidecars — those exit the recipe and become
custom `Pool` work against the [`Pool`][gmat_sweep.backends.Pool] ABC.
The `KubernetesJobPool` source under
`gmat_sweep/backends/kubernetes.py` is a working template.
