"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from hypothesis import HealthCheck, settings

from crdt_coedit_agents.core.config import get_settings
from crdt_coedit_agents.crdt.hlc import HybridLogicalClock
from crdt_coedit_agents.ops.signing import ReplicaIdentity
from crdt_coedit_agents.replica import Replica

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=2_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
DOC_ID = "test-doc"


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def seeded_identity(index: int) -> ReplicaIdentity:
    """A deterministic replica identity. Tests only: real keys come from the OS CSPRNG."""
    return ReplicaIdentity.from_seed(bytes([index]) * 32)


def manual_clock(start: int = 1_700_000_000_000) -> Callable[[], int]:
    """A wall clock that advances one millisecond per reading, so runs are reproducible."""
    state = {"now": start}

    def now_ms() -> int:
        state["now"] += 1
        return state["now"]

    return now_ms


def make_replica(index: int, doc_id: str = DOC_ID, *, start_ms: int = 1_700_000_000_000) -> Replica:
    """A replica with a deterministic identity and a deterministic clock."""
    identity = seeded_identity(index)
    clock = HybridLogicalClock(identity.replica_id, now_ms=manual_clock(start_ms))
    return Replica(identity, doc_id, clock=clock)


@pytest.fixture
def replica() -> Replica:
    return make_replica(1)


@pytest.fixture
def replicas() -> list[Replica]:
    return [make_replica(index) for index in range(1, 4)]
