"""Tests for gmat_sweep.backends.kubernetes.KubernetesJobPool — Job spec, drain, failure paths.

Mocks the ``kubernetes`` client and watch surfaces so the suite runs at
unit speed (no cluster, no Docker). Higher-level submit / drain /
resource-resolution tests monkeypatch the pool's ``_iter_completions``
helper directly so they can drive a deterministic completion sequence;
two dedicated tests exercise the watch loop itself against a mocked
``kubernetes.watch.Watch`` to pin the reconnect contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Generator, Iterable, Iterator
from typing import cast
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gmat_sweep.backends.base import Pool
from gmat_sweep.errors import BackendError
from gmat_sweep.spec import RunOutcome, RunSpec

kubernetes = pytest.importorskip("kubernetes")

from gmat_sweep.backends.kubernetes import KubernetesJobPool  # noqa: E402


def _make_spec(
    *, output_dir: Path, run_id: int = 0, overrides: dict[str, Any] | None = None
) -> RunSpec:
    return RunSpec(
        script_path=Path("/sweep/missions/m.script"),
        overrides=dict(overrides or {}),
        output_dir=output_dir,
        run_id=run_id,
        seed=None,
        run_options={},
    )


def _ok_outcome_dict(run_id: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "status": "ok",
        "output_paths": {},
        "duration_s": 0.5,
        "stderr": None,
        "started_at": now,
        "ended_at": now,
    }


class _BatchStub:
    """Records every Job submission. Real V1Job objects round-trip through it."""

    def __init__(self) -> None:
        self.created: list[tuple[str, Any]] = []

    def create_namespaced_job(self, *, namespace: str, body: Any) -> Any:
        self.created.append((namespace, body))
        return body

    def list_namespaced_job(self, **_kw: Any) -> Any:
        # Reference handed to ``Watch.stream`` — the watch stub never calls it,
        # but ``Watch.stream`` requires a callable to bind.
        return None


class _CoreStub:
    def __init__(self, *, log_text: str = "fake pod logs") -> None:
        self._log_text = log_text

    def list_namespaced_pod(self, *, namespace: str, label_selector: str) -> Any:
        Pod = kubernetes.client.V1Pod
        Meta = kubernetes.client.V1ObjectMeta
        items = [Pod(metadata=Meta(name="pod-x", namespace=namespace, labels={}))]
        return type("PodList", (), {"items": items})()

    def read_namespaced_pod_log(self, *, name: str, namespace: str) -> str:
        return self._log_text


@pytest.fixture
def patch_kube(monkeypatch: pytest.MonkeyPatch) -> _BatchStub:
    """Patch kubernetes.config + BatchV1Api + CoreV1Api with recording stubs."""
    monkeypatch.setattr(kubernetes.config, "load_kube_config", lambda **_kw: None)
    monkeypatch.setattr(kubernetes.config, "load_incluster_config", lambda: None)

    batch = _BatchStub()
    core = _CoreStub()
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", lambda: batch)
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda: core)
    return batch


def _make_pool(*, tmp_path: Path, **kwargs: Any) -> KubernetesJobPool:
    """Construct a pool with sensible defaults for unit tests."""
    defaults: dict[str, Any] = {
        "image": "ghcr.io/astro-tools/example:latest",
        "pvc_name": "sweep-pvc",
        "pvc_mount_path": "/sweep",
        "driver_mount_path": tmp_path,
        "namespace": "default",
        "in_cluster": False,
    }
    defaults.update(kwargs)
    return KubernetesJobPool(**defaults)


def _drive_completions(
    monkeypatch: pytest.MonkeyPatch,
    batch: _BatchStub,
    completion_order: Iterable[int],
) -> None:
    """Replace ``_iter_completions`` to yield Job names in completion_order.

    Reads the names from ``batch.created`` (recorded when the pool calls
    ``create_namespaced_job``) so each yield happens against the live
    submission record without inspecting interpreter frames.
    """

    order = list(completion_order)

    def _fake(self: KubernetesJobPool, _label_selector: str) -> Iterator[str]:
        completed: set[str] = set()
        for rid in order:
            target = _find_job_name(batch, rid, exclude=completed)
            if target is None:
                return
            completed.add(target)
            yield target

    monkeypatch.setattr(KubernetesJobPool, "_iter_completions", _fake, raising=True)


def _find_job_name(batch: _BatchStub, run_id: int, *, exclude: set[str]) -> str | None:
    for _ns, body in batch.created:
        name: str = body.metadata.name
        if name in exclude:
            continue
        if body.metadata.labels.get("gmat-sweep/run-id") == str(run_id):
            return name
    return None


def _write_outcome(driver_mount_path: Path, run_id: int) -> None:
    out_dir = driver_mount_path / "_outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(_ok_outcome_dict(run_id)))


# ---------------------------------------------------------------------------
# ABC parity
# ---------------------------------------------------------------------------


def test_kubernetesjobpool_is_pool_subclass() -> None:
    assert issubclass(KubernetesJobPool, Pool)
    assert KubernetesJobPool.subprocess_isolated is True


def test_subclass_setting_subprocess_isolated_false_rejected() -> None:
    """The Pool ABC's contract still applies to KubernetesJobPool subclasses."""
    with pytest.raises(BackendError):

        class _Bad(KubernetesJobPool):  # pragma: no cover - body never runs
            subprocess_isolated: ClassVar[bool] = False

            def submit(self, spec: RunSpec) -> Future[RunOutcome]:
                return Future()

            def as_completed(self, futures: Iterable[Future[RunOutcome]]) -> Iterator[RunOutcome]:
                return iter([])

            def close(self) -> None:
                pass


# ---------------------------------------------------------------------------
# Lazy import + constructor validation
# ---------------------------------------------------------------------------


def test_lazy_import_raises_backenderror_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `kubernetes` cannot be imported, the constructor raises BackendError."""
    monkeypatch.setitem(sys.modules, "kubernetes", None)
    with pytest.raises(BackendError) as ei:
        KubernetesJobPool(image="x", pvc_name="y")
    assert "[k8s]" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_init_rejects_empty_image(patch_kube: _BatchStub, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="image is required"):
        _make_pool(tmp_path=tmp_path, image="")


def test_init_rejects_empty_pvc_name(patch_kube: _BatchStub, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="pvc_name is required"):
        _make_pool(tmp_path=tmp_path, pvc_name="")


def test_init_rejects_zero_or_negative_parallelism(patch_kube: _BatchStub, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="parallelism"):
        _make_pool(tmp_path=tmp_path, parallelism=0)
    with pytest.raises(BackendError, match="parallelism"):
        _make_pool(tmp_path=tmp_path, parallelism=-2)


def test_init_rejects_in_cluster_with_kubeconfig(patch_kube: _BatchStub, tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="conflicts"):
        _make_pool(tmp_path=tmp_path, in_cluster=True, kubeconfig="/etc/kube.cfg")


# ---------------------------------------------------------------------------
# Auth-path probing
# ---------------------------------------------------------------------------


def test_in_cluster_true_calls_load_incluster_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(kubernetes.config, "load_kube_config", lambda **_kw: calls.append("kube"))
    monkeypatch.setattr(
        kubernetes.config, "load_incluster_config", lambda: calls.append("incluster")
    )
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", lambda: _BatchStub())
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda: _CoreStub())
    _make_pool(tmp_path=tmp_path, in_cluster=True)
    assert calls == ["incluster"]


def test_in_cluster_false_calls_load_kube_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        kubernetes.config,
        "load_kube_config",
        lambda **kw: calls.append(("kube", kw.get("config_file"))),
    )
    monkeypatch.setattr(
        kubernetes.config,
        "load_incluster_config",
        lambda: calls.append(("incluster", None)),
    )
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", lambda: _BatchStub())
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda: _CoreStub())
    _make_pool(tmp_path=tmp_path, in_cluster=False, kubeconfig="/etc/kube.cfg")
    assert calls == [("kube", "/etc/kube.cfg")]


def test_auto_detect_prefers_in_cluster_when_token_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(kubernetes.config, "load_kube_config", lambda **_kw: calls.append("kube"))
    monkeypatch.setattr(
        kubernetes.config, "load_incluster_config", lambda: calls.append("incluster")
    )
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", lambda: _BatchStub())
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda: _CoreStub())
    monkeypatch.setattr(KubernetesJobPool, "_in_cluster_token_present", staticmethod(lambda: True))

    _make_pool(tmp_path=tmp_path, in_cluster=None)
    assert calls == ["incluster"]


def test_auto_detect_falls_back_to_kube_config_when_no_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(kubernetes.config, "load_kube_config", lambda **_kw: calls.append("kube"))
    monkeypatch.setattr(
        kubernetes.config, "load_incluster_config", lambda: calls.append("incluster")
    )
    monkeypatch.setattr(kubernetes.client, "BatchV1Api", lambda: _BatchStub())
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", lambda: _CoreStub())
    monkeypatch.setattr(KubernetesJobPool, "_in_cluster_token_present", staticmethod(lambda: False))

    _make_pool(tmp_path=tmp_path, in_cluster=None)
    assert calls == ["kube"]


# ---------------------------------------------------------------------------
# submit / close basics
# ---------------------------------------------------------------------------


def test_submit_returns_pending_future(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    assert not f.done()
    assert not f.cancelled()
    pool.close()


def test_submit_after_close_raises(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    pool.close()
    with pytest.raises(BackendError):
        pool.submit(_make_spec(output_dir=tmp_path / "run_0"))


def test_as_completed_after_close_raises(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    pool.close()
    with pytest.raises(BackendError):
        list(pool.as_completed([]))


def test_close_is_idempotent(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    pool.close()
    pool.close()


def test_close_cancels_pending_futures(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0"))
    pool.close()
    assert f.cancelled()


# ---------------------------------------------------------------------------
# Job spec construction
# ---------------------------------------------------------------------------


def test_job_spec_carries_required_fields(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each submitted Job has the entrypoint command, PVC volume mount,
    sweep-id + run-id labels, ``backoffLimit=0``, and ``ttl=300``."""
    pool = _make_pool(tmp_path=tmp_path)
    spec = _make_spec(output_dir=tmp_path / "run_42", run_id=42)
    f = pool.submit(spec)

    _write_outcome(tmp_path, run_id=42)
    _drive_completions(monkeypatch, patch_kube, [42])
    list(pool.as_completed([f]))

    assert len(patch_kube.created) == 1
    namespace, job = patch_kube.created[0]
    assert namespace == "default"
    assert job.kind == "Job"
    assert job.api_version == "batch/v1"
    assert job.metadata.labels["gmat-sweep/run-id"] == "42"
    sweep_id = job.metadata.labels["gmat-sweep/sweep-id"]
    assert len(sweep_id) == 8

    spec_field = job.spec
    assert spec_field.backoff_limit == 0
    assert spec_field.ttl_seconds_after_finished == 300

    container = spec_field.template.spec.containers[0]
    assert container.image == "ghcr.io/astro-tools/example:latest"
    assert container.command[0] == "python"
    assert "gmat_sweep._run_subprocess" in " ".join(container.command)
    assert "/sweep/_specs/42.json" in container.command
    assert "/sweep/_outcomes/42.json" in container.command
    assert container.volume_mounts[0].name == "sweep"
    assert container.volume_mounts[0].mount_path == "/sweep"

    volume = spec_field.template.spec.volumes[0]
    assert volume.name == "sweep"
    assert volume.persistent_volume_claim.claim_name == "sweep-pvc"


def test_spec_json_written_to_driver_mount_path(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver writes ``<driver_mount_path>/_specs/<run_id>.json`` before
    submitting the Job; the file's content round-trips through ``RunSpec``."""
    pool = _make_pool(tmp_path=tmp_path)
    spec = _make_spec(output_dir=tmp_path / "run_7", run_id=7, overrides={"Sat.SMA": 7000.0})
    f = pool.submit(spec)
    _write_outcome(tmp_path, run_id=7)
    _drive_completions(monkeypatch, patch_kube, [7])
    list(pool.as_completed([f]))

    spec_path = tmp_path / "_specs" / "7.json"
    assert spec_path.exists()
    round_trip = RunSpec.from_dict(json.loads(spec_path.read_text()))
    assert round_trip.run_id == 7
    assert round_trip.overrides == {"Sat.SMA": 7000.0}


# ---------------------------------------------------------------------------
# Resource resolution: Mapping vs Callable vs None, against default_resources
# ---------------------------------------------------------------------------


def test_resources_none_emits_no_requirements(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    spec = _make_spec(output_dir=tmp_path / "run_0", run_id=0)
    f = pool.submit(spec)
    _write_outcome(tmp_path, run_id=0)
    _drive_completions(monkeypatch, patch_kube, [0])
    list(pool.as_completed([f]))

    container = patch_kube.created[0][1].spec.template.spec.containers[0]
    assert container.resources is None


def test_resources_default_only_applied_to_every_run(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(
        tmp_path=tmp_path,
        default_resources={"cpu": "2", "memory": "4Gi"},
    )
    spec = _make_spec(output_dir=tmp_path / "run_0", run_id=0)
    f = pool.submit(spec)
    _write_outcome(tmp_path, run_id=0)
    _drive_completions(monkeypatch, patch_kube, [0])
    list(pool.as_completed([f]))

    container = patch_kube.created[0][1].spec.template.spec.containers[0]
    assert container.resources.requests == {"cpu": "2", "memory": "4Gi"}
    assert container.resources.limits is None


def test_resources_mapping_overrides_default_per_run_id(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(
        tmp_path=tmp_path,
        default_resources={"cpu": "1"},
        resources={5: {"cpu": "8", "memory": "16Gi"}},
    )
    f0 = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    f5 = pool.submit(_make_spec(output_dir=tmp_path / "run_5", run_id=5))
    _write_outcome(tmp_path, run_id=0)
    _write_outcome(tmp_path, run_id=5)
    _drive_completions(monkeypatch, patch_kube, [0, 5])
    list(pool.as_completed([f0, f5]))

    by_run = {
        body.metadata.labels["gmat-sweep/run-id"]: body.spec.template.spec.containers[0].resources
        for _ns, body in patch_kube.created
    }
    assert by_run["0"].requests == {"cpu": "1"}
    assert by_run["5"].requests == {"cpu": "8", "memory": "16Gi"}


def test_resources_callable_called_per_spec(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    def _resources_for(spec: RunSpec) -> dict[str, str]:
        seen.append(spec.run_id)
        return {"cpu": str(spec.run_id + 1)}

    pool = _make_pool(tmp_path=tmp_path, resources=_resources_for)
    fs = [pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(3)]
    for i in range(3):
        _write_outcome(tmp_path, run_id=i)
    _drive_completions(monkeypatch, patch_kube, [0, 1, 2])
    list(pool.as_completed(fs))

    assert sorted(seen) == [0, 1, 2]
    by_run = {
        body.metadata.labels["gmat-sweep/run-id"]: body.spec.template.spec.containers[0].resources
        for _ns, body in patch_kube.created
    }
    assert by_run["0"].requests == {"cpu": "1"}
    assert by_run["2"].requests == {"cpu": "3"}


def test_resources_requests_and_limits_split_when_caller_specifies(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(
        tmp_path=tmp_path,
        resources={
            0: {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "4", "memory": "8Gi"},
            }
        },
    )
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_0", run_id=0))
    _write_outcome(tmp_path, run_id=0)
    _drive_completions(monkeypatch, patch_kube, [0])
    list(pool.as_completed([f]))

    container = patch_kube.created[0][1].spec.template.spec.containers[0]
    assert container.resources.requests == {"cpu": "1", "memory": "2Gi"}
    assert container.resources.limits == {"cpu": "4", "memory": "8Gi"}


# ---------------------------------------------------------------------------
# Drain semantics: parallelism cap, outcome read, failure folding
# ---------------------------------------------------------------------------


def test_parallelism_cap_limits_in_flight_jobs(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``parallelism=2`` keeps at most two Jobs created before any completes."""
    pool = _make_pool(tmp_path=tmp_path, parallelism=2)
    fs = [pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(5)]
    for i in range(5):
        _write_outcome(tmp_path, run_id=i)

    high_water_mark = 0
    completed: set[str] = set()

    def _fake(self: KubernetesJobPool, _label_selector: str) -> Iterator[str]:
        nonlocal high_water_mark
        for rid in range(5):
            in_flight_now = len(patch_kube.created) - len(completed)
            high_water_mark = max(high_water_mark, in_flight_now)
            target = _find_job_name(patch_kube, rid, exclude=completed)
            if target is None:
                return
            completed.add(target)
            yield target

    monkeypatch.setattr(KubernetesJobPool, "_iter_completions", _fake, raising=True)
    list(pool.as_completed(fs))

    assert high_water_mark == 2
    assert len(patch_kube.created) == 5


def test_parallelism_none_submits_all_up_front(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``parallelism=None`` creates every Job before draining the first."""
    pool = _make_pool(tmp_path=tmp_path, parallelism=None)
    fs = [pool.submit(_make_spec(output_dir=tmp_path / f"run_{i}", run_id=i)) for i in range(4)]
    for i in range(4):
        _write_outcome(tmp_path, run_id=i)

    seen_at_first_completion = 0
    completed: set[str] = set()

    def _fake(self: KubernetesJobPool, _label_selector: str) -> Iterator[str]:
        nonlocal seen_at_first_completion
        seen_at_first_completion = len(patch_kube.created)
        for rid in range(4):
            target = _find_job_name(patch_kube, rid, exclude=completed)
            if target is None:
                return
            completed.add(target)
            yield target

    monkeypatch.setattr(KubernetesJobPool, "_iter_completions", _fake, raising=True)
    list(pool.as_completed(fs))

    assert seen_at_first_completion == 4


def test_outcome_json_round_trips_back_to_runoutcome(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_3", run_id=3))
    _write_outcome(tmp_path, run_id=3)
    _drive_completions(monkeypatch, patch_kube, [3])
    outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].run_id == 3
    assert outcomes[0].status == "ok"
    assert f.done()
    assert f.result().run_id == 3


def test_missing_outcome_json_folded_into_failed_with_pod_logs(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pod that completes without writing outcome JSON gets a synthetic failure."""
    pool = _make_pool(tmp_path=tmp_path)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_9", run_id=9))
    # NB: no _write_outcome — simulates OOMKill / eviction before the
    # worker could write its outcome JSON.
    _drive_completions(monkeypatch, patch_kube, [9])
    outcomes = list(pool.as_completed([f]))

    assert len(outcomes) == 1
    assert outcomes[0].status == "failed"
    stderr = outcomes[0].stderr or ""
    assert "did not produce an outcome JSON" in stderr
    assert "fake pod logs" in stderr


def test_unreadable_outcome_json_folded_into_failed(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    f = pool.submit(_make_spec(output_dir=tmp_path / "run_2", run_id=2))
    out_dir = tmp_path / "_outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2.json").write_text("not valid json {")
    _drive_completions(monkeypatch, patch_kube, [2])
    outcomes = list(pool.as_completed([f]))

    assert outcomes[0].status == "failed"
    assert "unreadable outcome JSON" in (outcomes[0].stderr or "")


def test_as_completed_rejects_unknown_future(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    bogus: Future[RunOutcome] = Future()
    with pytest.raises(BackendError):
        list(pool.as_completed([bogus]))


def test_as_completed_with_no_specs_is_empty(patch_kube: _BatchStub, tmp_path: Path) -> None:
    pool = _make_pool(tmp_path=tmp_path)
    assert list(pool.as_completed([])) == []


# ---------------------------------------------------------------------------
# Watch-loop behavior (driving the real ``_iter_completions`` against a stub)
# ---------------------------------------------------------------------------


class _WatchStub:
    """Simulate ``kubernetes.watch.Watch`` with a scripted event sequence.

    The call counter is **class-level** so it persists across reconnects —
    each call to ``Watch()`` creates a fresh instance, so per-instance
    counters would silently reset and a scripted ``call_n=1`` branch
    would fire forever.
    """

    instances: ClassVar[list[Any]] = []
    call_n: ClassVar[int] = 0

    def __init__(self) -> None:
        self.stopped = False
        type(self).instances.append(self)

    def stream(self, _fn: Any, **_kw: Any) -> Iterator[dict[str, Any]]:
        cls = type(self)
        cls.call_n += 1
        events = cls._events_for_call(cls.call_n)
        yield from events

    def stop(self) -> None:
        self.stopped = True

    @classmethod
    def _events_for_call(cls, _call_n: int) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


def _make_completed_event(job_name: str, *, succeeded: int = 1, failed: int = 0) -> dict[str, Any]:
    Job = kubernetes.client.V1Job
    Meta = kubernetes.client.V1ObjectMeta
    Status = kubernetes.client.V1JobStatus
    return {
        "type": "MODIFIED",
        "object": Job(
            metadata=Meta(name=job_name, namespace="default"),
            status=Status(succeeded=succeeded, failed=failed),
        ),
    }


def test_iter_completions_yields_completed_job_names(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed-status events yield job name; pending events are skipped."""
    Job = kubernetes.client.V1Job
    Meta = kubernetes.client.V1ObjectMeta
    Status = kubernetes.client.V1JobStatus

    events_first_call = [
        # Pending — must be filtered out
        {"type": "ADDED", "object": Job(metadata=Meta(name="job-a"), status=Status())},
        _make_completed_event("job-a"),
        _make_completed_event("job-b", succeeded=0, failed=1),
    ]

    class _Stub(_WatchStub):
        @classmethod
        def _events_for_call(cls, call_n: int) -> list[dict[str, Any]]:
            if call_n == 1:
                return events_first_call
            raise AssertionError(f"unexpected Watch reconnect (call_n={call_n})")

    _Stub.instances = []
    _Stub.call_n = 0
    monkeypatch.setattr(kubernetes.watch, "Watch", _Stub)

    pool = _make_pool(tmp_path=tmp_path)
    gen = cast(Generator[str, None, None], pool._iter_completions("gmat-sweep/sweep-id=test"))
    try:
        assert next(gen) == "job-a"
        assert next(gen) == "job-b"
    finally:
        gen.close()


def test_iter_completions_reconnects_after_api_exception(
    patch_kube: _BatchStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient ApiException during stream() triggers a Watch() reconnect."""

    class _Stub(_WatchStub):
        @classmethod
        def _events_for_call(cls, call_n: int) -> list[dict[str, Any]]:
            if call_n == 1:
                raise kubernetes.client.exceptions.ApiException("boom")
            if call_n == 2:
                return [_make_completed_event("job-x")]
            raise AssertionError(f"unexpected Watch reconnect (call_n={call_n})")

    _Stub.instances = []
    _Stub.call_n = 0
    monkeypatch.setattr(kubernetes.watch, "Watch", _Stub)
    monkeypatch.setattr("gmat_sweep.backends.kubernetes.time.sleep", lambda _s: None)

    pool = _make_pool(tmp_path=tmp_path)
    gen = cast(Generator[str, None, None], pool._iter_completions("gmat-sweep/sweep-id=test"))
    try:
        assert next(gen) == "job-x"
        # Two Watch instances — proves a reconnect happened
        assert len(_Stub.instances) >= 2
    finally:
        gen.close()


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_importing_kubernetes_module_does_not_import_gmatpy() -> None:
    """Loading gmat_sweep.backends.kubernetes in a fresh interpreter must not import gmatpy.

    Each Pod is its own interpreter; if anything in the import chain
    triggered gmatpy in the *driver*, the subprocess-isolation contract
    would be silently violated even though the fan-out itself is clean.
    """
    code = (
        "import sys\n"
        "import gmat_sweep.backends.kubernetes  # noqa: F401\n"
        "assert 'gmatpy' not in sys.modules, sorted(m for m in sys.modules if 'gmat' in m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
