"""Dots and version vectors: identity, the join, and the delta-state diff."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crdt_coedit_agents.crdt.ids import Dot, VersionVector, format_dot, parse_dot

pytestmark = pytest.mark.unit

A = "a" * 32
B = "b" * 32


def test_a_dot_round_trips_through_its_wire_form() -> None:
    dot = Dot(replica_id=A, seq=42)
    assert format_dot(dot) == f"{A}.42"
    assert parse_dot(format_dot(dot)) == dot


@pytest.mark.parametrize(
    "text", ["", "x.1", f"{A}.0", f"{A}.", f"{A}.01", f"{A}.1.2", f"{A[:31]}.1"]
)
def test_malformed_dots_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match="malformed dot"):
        parse_dot(text)


def test_dots_validate_their_parts() -> None:
    with pytest.raises(ValueError, match="replica id"):
        Dot(replica_id="nope", seq=1)
    with pytest.raises(ValueError, match="start at 1"):
        Dot(replica_id=A, seq=0)


def test_observe_only_advances_on_the_next_sequence_number() -> None:
    vector = VersionVector()
    assert vector.observe(Dot(replica_id=A, seq=1)) is True
    assert vector.observe(Dot(replica_id=A, seq=3)) is False
    assert vector.get(A) == 1
    assert vector.observe(Dot(replica_id=A, seq=2)) is True
    assert vector.get(A) == 2


def test_membership_covers_every_earlier_operation() -> None:
    vector = VersionVector({A: 3})
    assert Dot(replica_id=A, seq=1) in vector
    assert Dot(replica_id=A, seq=3) in vector
    assert Dot(replica_id=A, seq=4) not in vector
    assert Dot(replica_id=B, seq=1) not in vector


def test_the_join_is_the_element_wise_maximum() -> None:
    left = VersionVector({A: 3, B: 1})
    right = VersionVector({A: 1, B: 5})
    assert left.join(right).as_dict() == {A: 3, B: 5}
    assert left.join(right) == right.join(left)
    assert left.join(left) == left


def test_dominance_and_missing_ranges_describe_the_delta_to_send() -> None:
    ahead = VersionVector({A: 5, B: 2})
    behind = VersionVector({A: 2})
    assert ahead.dominates(behind)
    assert not behind.dominates(ahead)
    assert behind.missing_ranges(ahead) == {A: (3, 5), B: (1, 2)}
    assert ahead.missing_ranges(behind) == {}


def test_vectors_reject_impossible_entries() -> None:
    with pytest.raises(ValueError, match="replica id"):
        VersionVector({"short": 1})
    with pytest.raises(ValueError, match="non-negative"):
        VersionVector({A: -1})
    assert len(VersionVector({A: 0})) == 0


def test_copies_are_independent_and_comparable() -> None:
    vector = VersionVector({A: 1})
    clone = vector.copy()
    clone.observe(Dot(replica_id=A, seq=2))
    assert vector.get(A) == 1
    assert clone.get(A) == 2
    assert vector != clone
    assert vector != "not a vector"
    assert hash(vector) == hash(VersionVector({A: 1}))
    assert repr(vector).startswith("VersionVector(")


@given(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=20))
def test_from_dots_accepts_any_order_of_a_contiguous_run(sequence: list[int]) -> None:
    contiguous = list(range(1, max(sequence) + 1))
    dots = (Dot(replica_id=A, seq=seq) for seq in reversed(contiguous))
    assert VersionVector.from_dots(dots).get(A) == max(sequence)
