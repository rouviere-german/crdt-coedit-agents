"""Hybrid logical clocks (Kulkarni et al., "Logical Physical Clocks", OPODIS 2014).

A hybrid logical clock timestamp is ``(wall_ms, counter)``. It tracks physical time closely enough
to be readable by humans while still respecting causality: if event *a* happens before event *b* on
the same replica, or *b* is the receipt of a message carrying *a*, then ``ts(a) < ts(b)``.

Timestamps are totally ordered by ``(wall_ms, counter, replica_id)``. The replica id breaks ties
between concurrent events, which is what the LWW register needs to pick a deterministic winner.

Clocks are bounded: a remote timestamp more than :data:`MAX_DRIFT_MS` ahead of local wall time is
refused rather than dragging the local clock into the future, which a malicious replica could
otherwise use to win every LWW write forever.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from crdt_coedit_agents.crdt.ids import REPLICA_ID_PATTERN

__all__ = [
    "MAX_COUNTER",
    "MAX_DRIFT_MS",
    "ClockDriftError",
    "HybridLogicalClock",
    "Timestamp",
    "check_drift",
    "wall_clock_ms",
]

MAX_DRIFT_MS = 60_000
MAX_COUNTER = 2**16 - 1


class ClockDriftError(ValueError):
    """A remote timestamp is too far ahead of local wall time to be accepted."""


@dataclass(frozen=True, slots=True, order=True)
class Timestamp:
    """A hybrid logical clock reading. Ordering is ``(wall_ms, counter, replica_id)``."""

    wall_ms: int
    counter: int
    replica_id: str

    def __post_init__(self) -> None:
        if self.wall_ms < 0:
            raise ValueError("wall_ms must be non-negative")
        if not 0 <= self.counter <= MAX_COUNTER:
            raise ValueError(f"counter must be within 0..{MAX_COUNTER}")
        if len(self.replica_id) != 32:
            raise ValueError(f"replica id must match {REPLICA_ID_PATTERN}")

    def encode(self) -> str:
        """Lexicographically ordered wire form, so sorting text sorts timestamps."""
        return f"{self.wall_ms:015d}-{self.counter:05d}-{self.replica_id}"

    @classmethod
    def decode(cls, text: str) -> Timestamp:
        parts = text.split("-")
        if len(parts) != 3 or len(parts[0]) != 15 or len(parts[1]) != 5:
            raise ValueError(f"malformed timestamp: {text!r}")
        return cls(wall_ms=int(parts[0]), counter=int(parts[1]), replica_id=parts[2])


class HybridLogicalClock:
    """A replica's hybrid logical clock. Not thread-safe; one clock per replica."""

    __slots__ = ("_counter", "_now_ms", "_replica_id", "_wall_ms")

    def __init__(self, replica_id: str, *, now_ms: Callable[[], int] | None = None) -> None:
        self._replica_id = replica_id
        self._now_ms = now_ms if now_ms is not None else _wall_clock_ms
        self._wall_ms = 0
        self._counter = 0

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def peek(self) -> Timestamp:
        """The last issued timestamp, without advancing the clock."""
        return Timestamp(self._wall_ms, self._counter, self._replica_id)

    def tick(self) -> Timestamp:
        """Issue a timestamp for a local event. Strictly greater than every previous reading."""
        physical = self._now_ms()
        if physical > self._wall_ms:
            self._wall_ms, self._counter = physical, 0
        else:
            self._counter = _bump(self._counter)
        return Timestamp(self._wall_ms, self._counter, self._replica_id)

    def observe(self, remote: Timestamp) -> Timestamp:
        """Merge a received timestamp and issue one that is greater than both sides.

        Raises :class:`ClockDriftError` when ``remote`` is more than :data:`MAX_DRIFT_MS` ahead of
        local wall time, so a replica cannot fast-forward its peers.
        """
        physical = self._now_ms()
        check_drift(remote, physical)
        wall = max(self._wall_ms, remote.wall_ms, physical)
        if wall == self._wall_ms == remote.wall_ms:
            counter = _bump(max(self._counter, remote.counter))
        elif wall == self._wall_ms:
            counter = _bump(self._counter)
        elif wall == remote.wall_ms:
            counter = _bump(remote.counter)
        else:
            counter = 0
        self._wall_ms, self._counter = wall, counter
        return Timestamp(wall, counter, self._replica_id)


def check_drift(remote: Timestamp, now_ms: int) -> None:
    """Raise :class:`ClockDriftError` when ``remote`` is too far ahead of ``now_ms``.

    This is the bound itself, separated from :class:`HybridLogicalClock` so that components which
    do not author operations — the sync server, for one — can refuse a far-future timestamp before
    it reaches a log or a last-writer-wins register.
    """
    if remote.wall_ms > now_ms + MAX_DRIFT_MS:
        raise ClockDriftError(
            f"remote timestamp is {remote.wall_ms - now_ms} ms ahead of local time "
            f"(limit {MAX_DRIFT_MS} ms)"
        )


def wall_clock_ms() -> int:
    """Local wall-clock time in milliseconds since the epoch."""
    return _wall_clock_ms()


def _bump(counter: int) -> int:
    if counter >= MAX_COUNTER:
        raise ClockDriftError(
            f"hybrid logical clock counter overflowed {MAX_COUNTER} within one millisecond"
        )
    return counter + 1


def _wall_clock_ms() -> int:
    return time.time_ns() // 1_000_000
