"""KubernetesJobPool: submit each :class:`gmat_sweep.spec.RunSpec` as a Kubernetes Job.

Every run becomes one ``batch/v1`` Job whose Pod runs the existing
``python -m gmat_sweep._run_subprocess`` entrypoint against per-run spec /
outcome JSON files on a shared :class:`PersistentVolumeClaim` mounted into
both the driver and the Pod. Pods are always fresh interpreters, so
``reuse_gmat_context`` is accepted for :class:`Pool` parity but has no
effect — there is no worker process to reuse on this backend.

Submission semantics match :class:`gmat_sweep.backends.dask.DaskPool` and
:class:`gmat_sweep.backends.ray.RayPool`: :meth:`submit` parks the spec
under a placeholder :class:`Future`; :meth:`as_completed` is the dispatch
point and walks completions via ``kubernetes.watch.Watch`` filtered on a
per-sweep label.

Storage / handoff
-----------------

The pool requires a ``ReadWriteMany`` (or otherwise driver-and-Pod-visible)
``PersistentVolumeClaim`` named ``pvc_name``. Per-run spec JSON is written
under ``<driver_mount_path>/_specs/<run_id>.json``; the Pod reads it from
``<pvc_mount_path>/_specs/<run_id>.json``. Outcome JSON travels back via
``_outcomes/<run_id>.json`` on the same volume. ``RunSpec.script_path`` and
``RunSpec.output_dir`` must already resolve inside the Pod's mount of the
same PVC — the pool does not rewrite paths between the driver and the Pod.

Failure semantics
-----------------

A Pod that produced an outcome JSON returns whatever ``RunOutcome`` the
worker wrote, including the failure-as-row contract for runs that GMAT
itself rejected. A Pod that never produced an outcome JSON (OOMKill,
eviction, image pull failure, …) is folded into a synthetic
:meth:`RunOutcome.failed` carrying the captured Pod logs as ``stderr``,
matching the same contract every other backend honours when the transport
layer fails.

Job lifecycle
-------------

Each Job is created with ``backoffLimit=0`` so a Pod failure maps 1:1 to a
``RunOutcome.failed`` (silent retries break determinism), and
``ttlSecondsAfterFinished=300`` so completed Jobs auto-GC after 5 minutes.
:meth:`close` cancels parked futures but does not delete in-flight Jobs —
the TTL is what reaps them.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

if TYPE_CHECKING:
    import kubernetes.client as _k8s_client_typing  # noqa: F401

__all__ = ["KubernetesJobPool"]


_LABEL_SWEEP_ID = "gmat-sweep/sweep-id"
_LABEL_RUN_ID = "gmat-sweep/run-id"
_SPEC_SUBDIR = "_specs"
_OUTCOME_SUBDIR = "_outcomes"
_VOLUME_NAME = "sweep"
_CONTAINER_NAME = "gmat-sweep"
_WATCH_TIMEOUT_SECONDS = 60
_DEFAULT_PARALLELISM = 64
_DEFAULT_BACKOFF_LIMIT = 0
_DEFAULT_TTL_SECONDS = 300
_INCLUSTER_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


_ResourcesArg = Mapping[int, Mapping[str, Any]] | Callable[[RunSpec], Mapping[str, Any]] | None


class KubernetesJobPool(Pool):
    """Distributed pool that submits each :class:`RunSpec` as a Kubernetes Job.

    Parameters
    ----------
    image:
        Fully-qualified container image with ``gmat-sweep[k8s]`` plus a
        working GMAT install. Pods run ``python -m
        gmat_sweep._run_subprocess`` inside this image. **Required** in
        v0.4 — a published default may follow in a later release.
    pvc_name:
        Name of an existing :class:`PersistentVolumeClaim` in
        ``namespace`` that the Pods will mount. Must be visible to the
        driver under ``driver_mount_path`` and to the Pods under
        ``pvc_mount_path``; ``ReadWriteMany`` is the typical access mode,
        but any topology that gives the driver and the Pods a shared
        view of the same files works.
    pvc_mount_path:
        Path inside each Pod where the PVC is mounted. The pool uses it
        to compute the in-Pod spec / outcome paths it passes to
        ``_run_subprocess`` via ``--spec`` / ``--outcome``. Defaults to
        ``/sweep``.
    driver_mount_path:
        Path on the driver side that resolves to the same PVC contents.
        Defaults to ``pvc_mount_path``, which is correct when the driver
        runs as a Pod mounting the PVC at the same path as the workers.
        For an out-of-cluster driver, set this to the local path where
        the PVC's backing storage is mounted (NFS, EFS, GCS Fuse, …).
    namespace:
        Kubernetes namespace for the Jobs and the watch loop. Defaults
        to ``"default"``.
    parallelism:
        Maximum number of in-flight Jobs at any moment. ``None`` means
        no cap (one Job created per spec, all up-front). Defaults to
        ``64`` so a 10000-run sweep does not stampede the API server.
    backoff_limit:
        Forwarded to ``V1JobSpec.backoff_limit``. Defaults to ``0`` so
        Pod failures map 1:1 to outcome failures; silent retries break
        the failure-as-row contract.
    ttl_seconds_after_finished:
        Forwarded to ``V1JobSpec.ttl_seconds_after_finished``. Defaults
        to ``300`` (5 min) — enough for a kubectl-window for inspection
        before the cluster GCs.
    resources:
        Per-run resource overrides keyed by ``RunSpec.run_id``, or a
        callable taking the spec and returning a resources dict. The
        resolved value populates ``V1ResourceRequirements.requests``
        (and ``.limits`` if the user includes a ``"limits"`` key, see
        below). When unresolved for a given run, the pool falls back to
        ``default_resources``.
    default_resources:
        Resources applied to every Pod that ``resources`` does not
        provide a value for. The dict shape is the standard k8s shape:
        ``{"cpu": "1", "memory": "4Gi"}`` for ``requests`` only, or
        ``{"requests": {...}, "limits": {...}}`` for both. ``None``
        leaves resources unset on the Job spec entirely.
    kubeconfig:
        Path to an explicit kubeconfig file. Forwarded to
        ``kubernetes.config.load_kube_config``. Mutually exclusive with
        ``in_cluster=True``.
    in_cluster:
        ``True`` forces in-cluster auth (``load_incluster_config``).
        ``False`` forces out-of-cluster auth (``load_kube_config``).
        ``None`` (default) auto-detects: in-cluster if the
        ServiceAccount token file exists, out-of-cluster otherwise.
    reuse_gmat_context:
        Accepted for :class:`Pool` API parity. Pods are always fresh
        interpreters on this backend, so the flag has no effect.
    """

    def __init__(
        self,
        *,
        image: str,
        pvc_name: str,
        pvc_mount_path: str = "/sweep",
        driver_mount_path: str | Path | None = None,
        namespace: str = "default",
        parallelism: int | None = _DEFAULT_PARALLELISM,
        backoff_limit: int = _DEFAULT_BACKOFF_LIMIT,
        ttl_seconds_after_finished: int = _DEFAULT_TTL_SECONDS,
        resources: _ResourcesArg = None,
        default_resources: Mapping[str, Any] | None = None,
        kubeconfig: str | Path | None = None,
        in_cluster: bool | None = None,
        reuse_gmat_context: bool = True,
    ) -> None:
        try:
            import kubernetes as _kubernetes
        except ImportError as exc:
            raise BackendError(
                "KubernetesJobPool requires the [k8s] extra: pip install gmat-sweep[k8s]"
            ) from exc

        if in_cluster is True and kubeconfig is not None:
            raise BackendError(
                "in_cluster=True conflicts with an explicit kubeconfig path; choose one"
            )
        if parallelism is not None and parallelism < 1:
            raise BackendError(
                f"parallelism must be a positive integer or None, got {parallelism!r}"
            )
        if not image:
            raise BackendError("image is required")
        if not pvc_name:
            raise BackendError("pvc_name is required")

        self._kubernetes = _kubernetes
        self._image = image
        self._pvc_name = pvc_name
        self._pvc_mount_path = pvc_mount_path.rstrip("/") or "/"
        self._driver_mount_path = (
            Path(driver_mount_path) if driver_mount_path is not None else Path(self._pvc_mount_path)
        )
        self._namespace = namespace
        self._parallelism = parallelism
        self._backoff_limit = backoff_limit
        self._ttl_seconds_after_finished = ttl_seconds_after_finished
        self._resources = resources
        self._default_resources = dict(default_resources) if default_resources is not None else None
        self._reuse_gmat_context = reuse_gmat_context

        self._load_kube_config(in_cluster=in_cluster, kubeconfig=kubeconfig)

        self._batch_api = _kubernetes.client.BatchV1Api()
        self._core_api = _kubernetes.client.CoreV1Api()

        self._pending: dict[Future[RunOutcome], RunSpec] = {}
        self._closed = False

    def _load_kube_config(self, *, in_cluster: bool | None, kubeconfig: str | Path | None) -> None:
        config = self._kubernetes.config

        if in_cluster is True:
            config.load_incluster_config()
            return
        if in_cluster is False:
            config.load_kube_config(config_file=str(kubeconfig) if kubeconfig else None)
            return

        if self._in_cluster_token_present():
            try:
                config.load_incluster_config()
                return
            except config.ConfigException:
                pass
        config.load_kube_config(config_file=str(kubeconfig) if kubeconfig else None)

    @staticmethod
    def _in_cluster_token_present() -> bool:
        return Path(_INCLUSTER_TOKEN_PATH).exists()

    def submit(self, spec: RunSpec) -> Future[RunOutcome]:
        if self._closed:
            raise BackendError("KubernetesJobPool is closed; cannot submit new specs")
        future: Future[RunOutcome] = Future()
        self._pending[future] = spec
        return future

    def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
        if self._closed:
            raise BackendError("KubernetesJobPool is closed; cannot drain futures")

        wanted = list(futures)
        specs: list[RunSpec] = []
        future_by_run_id: dict[int, Future[RunOutcome]] = {}
        for f in wanted:
            spec = self._pending.pop(f, None)
            if spec is None:
                raise BackendError(
                    "Future was not submitted to this pool, or has already been drained"
                )
            specs.append(spec)
            future_by_run_id[spec.run_id] = f

        if not specs:
            return

        sweep_id = secrets.token_hex(4)
        label_selector = f"{_LABEL_SWEEP_ID}={sweep_id}"

        self._ensure_io_dirs()

        in_flight: dict[str, tuple[RunSpec, datetime]] = {}
        submit_iter = iter(specs)
        cap = self._parallelism if self._parallelism is not None else len(specs)

        # initial fill
        while len(in_flight) < cap:
            if not self._submit_next(submit_iter, in_flight, sweep_id):
                break

        completion_stream = self._iter_completions(label_selector)
        while in_flight:
            completed_name = next(completion_stream, None)
            if completed_name is None:
                # stream exhausted unexpectedly; surface remaining as transport failures
                for stuck_name, (stuck_spec, started) in list(in_flight.items()):
                    outcome = self._fold_unknown_failure(stuck_spec, stuck_name, started)
                    future_by_run_id[stuck_spec.run_id].set_result(outcome)
                    yield outcome
                    in_flight.pop(stuck_name)
                break

            entry = in_flight.pop(completed_name, None)
            if entry is None:
                continue
            spec, started = entry
            outcome = self._read_outcome(spec, completed_name, started_at=started)
            future_by_run_id[spec.run_id].set_result(outcome)
            yield outcome

            self._submit_next(submit_iter, in_flight, sweep_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for f in self._pending:
            f.cancel()
        self._pending.clear()

    def _ensure_io_dirs(self) -> None:
        (self._driver_mount_path / _SPEC_SUBDIR).mkdir(parents=True, exist_ok=True)
        (self._driver_mount_path / _OUTCOME_SUBDIR).mkdir(parents=True, exist_ok=True)

    def _submit_next(
        self,
        submit_iter: Iterator[RunSpec],
        in_flight: dict[str, tuple[RunSpec, datetime]],
        sweep_id: str,
    ) -> bool:
        spec = next(submit_iter, None)
        if spec is None:
            return False
        job_name = self._submit_job(spec, sweep_id)
        in_flight[job_name] = (spec, datetime.now(timezone.utc))
        return True

    def _submit_job(self, spec: RunSpec, sweep_id: str) -> str:
        client = self._kubernetes.client

        spec_path_driver = self._driver_mount_path / _SPEC_SUBDIR / f"{spec.run_id}.json"
        spec_path_driver.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

        in_pod_spec_path = f"{self._pvc_mount_path}/{_SPEC_SUBDIR}/{spec.run_id}.json"
        in_pod_outcome_path = f"{self._pvc_mount_path}/{_OUTCOME_SUBDIR}/{spec.run_id}.json"

        job_name = self._job_name(sweep_id, spec.run_id)
        labels = {_LABEL_SWEEP_ID: sweep_id, _LABEL_RUN_ID: str(spec.run_id)}

        container = client.V1Container(
            name=_CONTAINER_NAME,
            image=self._image,
            command=[
                "python",
                "-m",
                "gmat_sweep._run_subprocess",
                "--spec",
                in_pod_spec_path,
                "--outcome",
                in_pod_outcome_path,
            ],
            volume_mounts=[
                client.V1VolumeMount(name=_VOLUME_NAME, mount_path=self._pvc_mount_path)
            ],
            resources=self._build_resource_requirements(spec),
        )
        pod_spec = client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            volumes=[
                client.V1Volume(
                    name=_VOLUME_NAME,
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self._pvc_name
                    ),
                )
            ],
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec,
        )
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=self._backoff_limit,
            ttl_seconds_after_finished=self._ttl_seconds_after_finished,
        )
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name, labels=labels),
            spec=job_spec,
        )

        self._batch_api.create_namespaced_job(namespace=self._namespace, body=job)
        return job_name

    def _job_name(self, sweep_id: str, run_id: int) -> str:
        return f"gmat-sweep-{sweep_id}-{run_id:05d}"

    def _build_resource_requirements(self, spec: RunSpec) -> Any | None:
        client = self._kubernetes.client
        resolved = self._resolve_resources(spec)
        if resolved is None:
            return None

        if "requests" in resolved or "limits" in resolved:
            requests = dict(resolved.get("requests", {})) or None
            limits = dict(resolved.get("limits", {})) or None
        else:
            requests = dict(resolved)
            limits = None

        if not requests and not limits:
            return None
        return client.V1ResourceRequirements(requests=requests, limits=limits)

    def _resolve_resources(self, spec: RunSpec) -> Mapping[str, Any] | None:
        resolved: Mapping[str, Any] | None = None
        if callable(self._resources):
            resolved = self._resources(spec)
        elif isinstance(self._resources, Mapping):
            resolved = self._resources.get(spec.run_id)
        if resolved is not None:
            return resolved
        return self._default_resources

    def _iter_completions(self, label_selector: str) -> Iterator[str]:
        watch_module = self._kubernetes.watch
        rest_exc = self._kubernetes.client.exceptions.ApiException
        while True:
            watch = watch_module.Watch()
            try:
                for event in watch.stream(
                    self._batch_api.list_namespaced_job,
                    namespace=self._namespace,
                    label_selector=label_selector,
                    timeout_seconds=_WATCH_TIMEOUT_SECONDS,
                ):
                    job = event["object"]
                    status = getattr(job, "status", None)
                    if status is None:
                        continue
                    succeeded = getattr(status, "succeeded", None) or 0
                    failed = getattr(status, "failed", None) or 0
                    if succeeded >= 1 or failed >= 1:
                        yield job.metadata.name
            except rest_exc:
                # transient API failure: brief backoff and reconnect
                time.sleep(0.5)
                continue
            finally:
                with contextlib.suppress(Exception):
                    watch.stop()
            # Natural end of stream is a server-side timeout — reconnect.

    def _read_outcome(self, spec: RunSpec, job_name: str, *, started_at: datetime) -> RunOutcome:
        outcome_path = self._driver_mount_path / _OUTCOME_SUBDIR / f"{spec.run_id}.json"
        if outcome_path.exists():
            try:
                return RunOutcome.from_dict(json.loads(outcome_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                ended_at = datetime.now(timezone.utc)
                return RunOutcome.failed(
                    run_id=spec.run_id,
                    stderr=f"unreadable outcome JSON at {outcome_path}: {exc}",
                    started_at=started_at,
                    ended_at=ended_at,
                )
        return self._fold_unknown_failure(spec, job_name, started_at)

    def _fold_unknown_failure(
        self, spec: RunSpec, job_name: str, started_at: datetime
    ) -> RunOutcome:
        ended_at = datetime.now(timezone.utc)
        pod_logs = self._fetch_pod_logs(job_name)
        return RunOutcome.failed(
            run_id=spec.run_id,
            stderr=(
                f"Job {job_name} did not produce an outcome JSON "
                f"(pod failed before writing). Pod logs:\n{pod_logs}"
            ).rstrip(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def _fetch_pod_logs(self, job_name: str) -> str:
        client = self._kubernetes.client
        try:
            pods = self._core_api.list_namespaced_pod(
                namespace=self._namespace,
                label_selector=f"job-name={job_name}",
            )
            items = getattr(pods, "items", None) or []
            if not items:
                return "(no Pod found for Job)"
            pod_name = items[0].metadata.name
            log = self._core_api.read_namespaced_pod_log(name=pod_name, namespace=self._namespace)
            return str(log)
        except client.exceptions.ApiException as exc:
            return f"(could not fetch Pod logs: {exc})"
