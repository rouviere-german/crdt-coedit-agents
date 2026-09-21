"""Hybrid logical clocks: causality, total order and the drift bound."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crdt_coedit_agents.crdt.hlc import (
    MAX_COUNTER,
    MAX_DRIFT_MS,
    ClockDriftError,
    HybridLogicalClock,
    Timestamp,
)
from tests.conftest import manual_clock, seeded_identity

pytestmark = pytest.mark.unit

REPLICA_A = "a" * 32
REPLICA_B = "b" * 32


def frozen_clock(replica_id: str, now: int = 1_000) -> HybridLogicalClock:
    return HybridLogicalClock(replica_id, now_ms=lambda: now)


def test_local_ticks_are_strictly_increasing_even_within_one_millisecond() -> None:
    clock = frozen_clock(REPLICA_A)
    stamps = [clock.tick() for _ in range(5)]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 5
    assert {stamp.wall_ms for stamp in stamps} == {1_000}
    assert [stamp.counter for stamp in stamps] == [0, 1, 2, 3, 4]


def test_a_tick_after_the_wall_clock_advances_resets_the_counter() -> None:
    clock = HybridLogicalClock(REPLICA_A, now_ms=manual_clock(start=10))
    first, second = clock.tick(), clock.tick()
    assert (first.wall_ms, first.counter) == (11, 0)
    assert (second.wall_ms, second.counter) == (12, 0)


def test_observing_a_remote_timestamp_produces_one_greater_than_both() -> None:
    sender = frozen_clock(REPLICA_A, now=5_000)
    receiver = frozen_clock(REPLICA_B, now=1_000)
    remote = sender.tick()
    local = receiver.observe(remote)
    assert local > remote
    assert local.replica_id == REPLICA_B
    assert receiver.tick() > local


def test_a_message_chain_keeps_happens_before_ordering() -> None:
    first = frozen_clock(REPLICA_A, now=1_000)
    second = frozen_clock(REPLICA_B, now=1_000)
    a1 = first.tick()
    b1 = second.observe(a1)
    a2 = first.observe(b1)
    assert a1 < b1 < a2


def test_a_remote_clock_too_far_ahead_is_refused() -> None:
    clock = frozen_clock(REPLICA_A, now=1_000)
    tolerable = Timestamp(1_000 + MAX_DRIFT_MS, 0, REPLICA_B)
    assert clock.observe(tolerable).wall_ms == 1_000 + MAX_DRIFT_MS

    fresh = frozen_clock(REPLICA_A, now=1_000)
    with pytest.raises(ClockDriftError, match="ahead of local time"):
        fresh.observe(Timestamp(1_000 + MAX_DRIFT_MS + 1, 0, REPLICA_B))


def test_the_counter_cannot_overflow_silently() -> None:
    clock = frozen_clock(REPLICA_A)
    with pytest.raises(ClockDriftError, match="overflowed"):
        clock.observe(Timestamp(1_000, MAX_COUNTER, REPLICA_B))


def test_peek_reports_the_last_issued_reading_without_advancing() -> None:
    clock = frozen_clock(REPLICA_A)
    issued = clock.tick()
    assert clock.peek() == issued
    assert clock.peek() == issued
    assert clock.replica_id == REPLICA_A


@given(
    st.integers(min_value=0, max_value=10**12),
    st.integers(min_value=0, max_value=MAX_COUNTER),
    st.sampled_from([REPLICA_A, REPLICA_B]),
)
def test_encoding_round_trips_and_sorts_like_the_timestamp(
    wall_ms: int, counter: int, replica_id: str
) -> None:
    stamp = Timestamp(wall_ms, counter, replica_id)
    assert Timestamp.decode(stamp.encode()) == stamp
    later = Timestamp(wall_ms + 1, counter, replica_id)
    assert stamp.encode() < later.encode()


def test_malformed_timestamps_are_rejected() -> None:
    for text in ("", "1-2", "x-y-z", "000000000000001-00002"):
        with pytest.raises(ValueError, match=r"(malformed timestamp|invalid literal)"):
            Timestamp.decode(text)
    with pytest.raises(ValueError, match="non-negative"):
        Timestamp(-1, 0, REPLICA_A)
    with pytest.raises(ValueError, match="counter must be"):
        Timestamp(0, MAX_COUNTER + 1, REPLICA_A)
    with pytest.raises(ValueError, match="replica id"):
        Timestamp(0, 0, "short")
    # Right length, wrong alphabet: the error message names a regex, so apply it.
    with pytest.raises(ValueError, match="replica id"):
        Timestamp(0, 0, "Z" * 32)
    with pytest.raises(ValueError, match="replica id"):
        Timestamp(0, 0, "A" * 32)


def test_a_real_identity_produces_a_usable_replica_id() -> None:
    identity = seeded_identity(9)
    clock = HybridLogicalClock(identity.replica_id)
    assert clock.tick().replica_id == identity.replica_id
