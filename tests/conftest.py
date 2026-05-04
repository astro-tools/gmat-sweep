"""Shared pytest fixtures for the gmat-sweep test suite."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest


@dataclass
class FakeResults:
    """Stand-in for :class:`gmat_run.results.Results` — only the bits the worker reads."""

    reports: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    log: str = ""


@dataclass
class FakeMission:
    """Stand-in for :class:`gmat_run.mission.Mission` — only the bits the worker calls.

    Each callable hook can be swapped out per test to inject failures at the
    matching point. The default hooks succeed and return an empty
    :class:`FakeResults`.
    """

    script_path: Path
    overrides_log: list[tuple[str, Any]] = field(default_factory=list)
    setitem_hook: Callable[[str, Any], None] | None = None
    run_hook: Callable[..., FakeResults] | None = None
    run_kwargs_log: list[dict[str, Any]] = field(default_factory=list)

    def __setitem__(self, key: str, value: Any) -> None:
        self.overrides_log.append((key, value))
        if self.setitem_hook is not None:
            self.setitem_hook(key, value)

    def run(self, **kwargs: Any) -> FakeResults:
        self.run_kwargs_log.append(kwargs)
        if self.run_hook is not None:
            return self.run_hook(**kwargs)
        return FakeResults()


class FakeGmatRunError(Exception):
    """Stand-in for :class:`gmat_run.errors.GmatRunError` — carries the engine ``log``."""

    def __init__(self, message: str, *, log: str = "") -> None:
        super().__init__(message)
        self.log = log


@dataclass
class FakeGmatRun:
    """Container for the fake module installed at ``sys.modules['gmat_run']``.

    Tests call :meth:`install_loader` (or :meth:`install_failing_loader`) to
    swap out the ``Mission.load`` callable per scenario; ``last_mission`` is
    populated as a side effect of the default loader so tests can assert on
    the calls the worker made through the mission interface.
    """

    module: ModuleType
    last_mission: FakeMission | None = None

    def install_loader(
        self,
        *,
        setitem_hook: Callable[[str, Any], None] | None = None,
        run_hook: Callable[..., FakeResults] | None = None,
    ) -> None:
        def _load(path: Any, **_: Any) -> FakeMission:
            mission = FakeMission(
                script_path=Path(path), setitem_hook=setitem_hook, run_hook=run_hook
            )
            self.last_mission = mission
            return mission

        self.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]

    def install_failing_loader(self, exc: BaseException) -> None:
        def _load(_path: Any, **_: Any) -> FakeMission:
            raise exc

        self.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]


@pytest.fixture
def fake_gmat_run(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGmatRun]:
    """Install a fake ``gmat_run`` module into :data:`sys.modules` for the test.

    The default ``Mission.load`` returns a :class:`FakeMission` whose
    ``__setitem__`` and ``run`` succeed; tests call ``install_loader`` /
    ``install_failing_loader`` on the yielded :class:`FakeGmatRun` to inject
    behaviour.
    """
    container = FakeGmatRun(module=ModuleType("gmat_run"))
    container.install_loader()
    container.module.GmatRunError = FakeGmatRunError  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gmat_run", container.module)
    yield container
