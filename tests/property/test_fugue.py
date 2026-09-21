"""Fugue's defining property: concurrent runs never interleave.

Two people typing at the same spot should end up with two paragraphs, not a zipper. This is the
property RGA gets right for forward typing and wrong for backward typing, and that Logoot-style
list CRDTs get wrong in both directions; Fugue's two-sided tree gets both right, so both are
tested here.

Runs are tracked by the dots of the operations that created them, not by the characters, so the
check is exact and does not depend on picking marker characters the base text happens not to use.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from crdt_coedit_agents.crdt.ids import parse_dot
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica
from tests.conftest import make_replica

pytestmark = [pytest.mark.property, pytest.mark.unit]

BASE = st.text(alphabet="abcdefgh", min_size=1, max_size=12)
RUN_LENGTH = st.integers(min_value=2, max_value=6)
POSITION = st.integers(min_value=0, max_value=20)


def _forward_run(replica: Replica, anchor: str | None, length: int) -> list[SignedOp]:
    """Typing left to right: each character is anchored to the one before it."""
    return replica.insert_text("x" * length, after=anchor)


def _backward_run(replica: Replica, anchor: str | None, length: int) -> list[SignedOp]:
    """Typing right to left: every character is inserted at the same anchor."""
    ops: list[SignedOp] = []
    for _ in range(length):
        ops.extend(replica.insert_text("y", after=anchor))
    return ops


def _block_positions(replica: Replica, ops: list[SignedOp]) -> list[int]:
    scope = replica.state.main_scope()
    order = [node.anchor_id for node in replica.state.text.visible_nodes(scope)]
    index = {anchor: position for position, anchor in enumerate(order)}
    return sorted(index[str(op.dot)] for op in ops)


def _is_contiguous(positions: list[int]) -> bool:
    return positions == list(range(positions[0], positions[0] + len(positions)))


def _setup(base: str, position: int) -> tuple[Replica, Replica, str | None]:
    left = make_replica(1)
    right = make_replica(2, start_ms=1_700_000_010_000)
    left.insert_text(base)
    right.receive(list(left.ops()))
    anchors = left.view().anchors
    anchor = anchors[position % len(anchors)] if anchors else None
    return left, right, anchor


@given(BASE, POSITION, RUN_LENGTH, RUN_LENGTH)
def test_concurrent_forward_runs_do_not_interleave(
    base: str, position: int, first: int, second: int
) -> None:
    left, right, anchor = _setup(base, position)
    left_ops = _forward_run(left, anchor, first)
    right_ops = _forward_run(right, anchor, second)
    left.receive(right_ops)
    right.receive(left_ops)

    assert left.state_hash() == right.state_hash()
    assert _is_contiguous(_block_positions(left, left_ops))
    assert _is_contiguous(_block_positions(left, right_ops))


@given(BASE, POSITION, RUN_LENGTH, RUN_LENGTH)
def test_concurrent_backward_runs_do_not_interleave(
    base: str, position: int, first: int, second: int
) -> None:
    left, right, anchor = _setup(base, position)
    left_ops = _backward_run(left, anchor, first)
    right_ops = _backward_run(right, anchor, second)
    left.receive(right_ops)
    right.receive(left_ops)

    assert left.state_hash() == right.state_hash()
    assert _is_contiguous(_block_positions(left, left_ops))
    assert _is_contiguous(_block_positions(left, right_ops))


@given(BASE, POSITION, RUN_LENGTH, RUN_LENGTH)
def test_a_forward_run_and_a_backward_run_do_not_interleave(
    base: str, position: int, first: int, second: int
) -> None:
    left, right, anchor = _setup(base, position)
    left_ops = _forward_run(left, anchor, first)
    right_ops = _backward_run(right, anchor, second)
    left.receive(right_ops)
    right.receive(left_ops)

    assert left.state_hash() == right.state_hash()
    assert _is_contiguous(_block_positions(left, left_ops))
    assert _is_contiguous(_block_positions(left, right_ops))


@given(BASE, POSITION, RUN_LENGTH, RUN_LENGTH, RUN_LENGTH)
def test_three_concurrent_runs_stay_in_three_blocks(
    base: str, position: int, first: int, second: int, third: int
) -> None:
    left, right, anchor = _setup(base, position)
    other = make_replica(3, start_ms=1_700_000_020_000)
    other.receive(list(left.ops()))

    runs = [
        _forward_run(left, anchor, first),
        _backward_run(right, anchor, second),
        _forward_run(other, anchor, third),
    ]
    everything = [op for run in runs for op in run]
    for replica in (left, right, other):
        replica.receive(everything)

    assert len({replica.state_hash() for replica in (left, right, other)}) == 1
    for run in runs:
        assert _is_contiguous(_block_positions(left, run))


@given(BASE, POSITION, st.text(alphabet="pqrs", min_size=1, max_size=6))
def test_a_local_insert_lands_immediately_after_its_anchor(
    base: str, position: int, addition: str
) -> None:
    replica = make_replica(1)
    replica.insert_text(base)
    anchors = replica.view().anchors
    anchor = anchors[position % len(anchors)]
    at = anchors.index(anchor)
    replica.insert_text(addition, after=anchor)
    assert replica.view().text == base[: at + 1] + addition + base[at + 1 :]


@given(BASE, POSITION)
def test_deleting_is_idempotent_and_commutes_with_a_concurrent_insert(
    base: str, position: int
) -> None:
    assume(len(base) >= 2)
    left, right, anchor = _setup(base, position)
    assert anchor is not None
    deletes = left.delete_run(anchor)
    inserts = right.insert_text("z", after=anchor)

    left.receive(inserts)
    right.receive(deletes)
    # Re-delivering the same operations changes nothing.
    left.receive(deletes + inserts)
    right.receive(inserts + deletes)

    assert left.state_hash() == right.state_hash()
    assert "z" in left.view().text
    assert str(parse_dot(anchor)) not in left.view().anchors
