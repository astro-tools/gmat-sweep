# Cluster recipes

Worked examples for wiring `gmat-sweep` into shared cluster infrastructure
— one page per orchestrator, each pairing the cluster-side configuration
with the matching `sweep()` driver.

The recipes document patterns; they don't introduce new APIs. The
underlying [`DaskPool`][gmat_sweep.backends.DaskPool],
[`RayPool`][gmat_sweep.backends.RayPool], and
[`KubernetesJobPool`][gmat_sweep.backends.KubernetesJobPool] surfaces —
and the [`Pool`][gmat_sweep.backends.Pool] ABC — are covered on the
[Backends](../backends.md) page. Reach for a recipe when you've already
decided on the orchestrator and need the wiring that makes a sweep run
on it.

## Choosing a recipe

| Recipe | Pool | When to pick it |
|---|---|---|
| [Slurm with `srun`](slurm.md) | [`DaskPool`][gmat_sweep.backends.DaskPool] via `dask-jobqueue` | An HPC site with a Slurm scheduler; you submit one driver job and let `SLURMCluster` request worker tasks elastically. |
| [Kubernetes pod-per-worker](kubernetes.md) | [`DaskPool`][gmat_sweep.backends.DaskPool] via `dask-kubernetes` | Kubernetes through a Dask scheduler — workers are Pods, the cluster is managed by the Dask Operator. Pick when other code in your stack already wants a Dask client. |
| [Kubernetes Job-per-run](kubernetes-jobpool.md) | [`KubernetesJobPool`][gmat_sweep.backends.KubernetesJobPool] | Kubernetes without Dask — every run is one `batch/v1` Job, scheduled directly. Pick when you want native cluster scheduling and one less middleware layer. |
| [Ray autoscaling](ray-autoscaling.md) | [`RayPool`][gmat_sweep.backends.RayPool] via `ray up` | A Ray cluster — local, on-prem, or cloud — with autoscaling between a head node and an elastic worker pool. |

Each recipe assumes you've followed [Getting started](../getting-started.md)
locally first. The local sweep proves your script and grid are sound;
the recipe then lifts the same call onto cluster workers without
changing the `sweep()` invocation itself — only the `backend=` argument
changes.

## Prerequisites that apply across all three

- A working GMAT install reachable on **every** worker node, not just
  the driver. The discovery is `gmat-run`'s job; misconfigured workers
  surface as every run failing with the same import error.
- A shared output directory at the same path on every worker. Per-run
  Parquet files and the manifest live there; node-local scratch only
  works if you stage results back yourself.
- The matching cluster-orchestrator package installed in the same env
  the workers run from (`dask-jobqueue`, `dask-kubernetes`, or `ray`).
  None of these are `gmat-sweep` dependencies — pick whichever your
  infrastructure uses and install it explicitly.

## When none of these fits

The orchestrators above are the ones with one-shot recipes. For
anything else — AWS Batch, GCP Batch, custom MPI launchers, in-house
schedulers — write a custom `Pool` against the
[`Pool`][gmat_sweep.backends.Pool] ABC. Its contract is small: accept
[`RunSpec`][gmat_sweep.spec.RunSpec]s, route each through the per-task
subprocess hop, and yield [`RunOutcome`][gmat_sweep.spec.RunOutcome]s as
they complete. The shipped pools are exactly that pattern, in different
shapes.

## Looking for the other side?

These recipes wire `gmat-sweep` *onto* cluster infrastructure. For
patterns that turn the sweep's outputs into the inputs downstream
consumers need — visualisation export, cross-tool validation,
external-tool wrapping — see the [Cookbook](../cookbook.md).
