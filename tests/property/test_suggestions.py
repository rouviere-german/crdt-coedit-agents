"""What a suggestion does to the document, under concurrent human editing.

Two claims, both randomized:

1. **Rejecting leaves no trace.** After a suggestion is rejected, the main view is byte-identical
   to replaying the same history with that suggestion's operations removed altogether. This is the
   strong form: not "the filter hides it" but "the document is what it would have been".
2. **Accepting lands next to the anchor.** An accepted run appears as one contiguous block whose
   preceding visible element is the anchor the agent addressed, or — when that anchor was deleted
   concurrently — the nearest element before it that survived.

Claim 2 is stated for concurrent edits that are *not* at the same anchor. Two writers inserting at
one position are genuinely concurrent, and which of them lands first is a tie-break, not a bug; the
non-interleaving properties in ``test_fugue.py`` cover that case instead.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crdt_coedit_agents.crdt.document import DocumentState
from crdt_coedit_agents.ops.apply import apply_op
from crdt_coedit_agents.ops.model import SignedOp, SuggestionDecide
from crdt_coedit_agents.replica import Replica, new_suggestion_id
from tests.conftest import make_replica

pytestmark = [pytest.mark.property, pytest.mark.unit]

SUGGESTION = "sg-" + "a" * 16
BASE = st.text(alphabet="abcdefghij", min_size=4, max_size=16)
ADDITION = st.text(alphabet="XYZ", min_size=1, max_size=6)
POSITION = st.integers(min_value=0, max_value=30)
DELETES = st.lists(st.integers(min_value=0, max_value=30), max_size=3)


def _replay_without(replica: Replica, suggestion_id: str) -> str:
    """Materialize the history as if the suggestion had never been made."""
    state = DocumentState(doc_id=replica.doc_id)
    for signed in replica.ops():
        payload = signed.op.payload
        if signed.op.suggestion_id == suggestion_id:
            continue
        if isinstance(payload, SuggestionDecide) and payload.suggestion_id == suggestion_id:
            continue
        apply_op(state, signed)
    return state.view().text


def _scenario(
    base: str, position: int, addition: str, deletes: list[int]
) -> tuple[Replica, Replica, str, list[SignedOp]]:
    """A human writes, an agent suggests, the human keeps editing, then everything merges."""
    human = make_replica(1)
    agent = make_replica(2, start_ms=1_700_000_500_000)
    human.insert_text(base)
    agent.receive(list(human.ops()))

    anchors = agent.view().anchors
    # Never the last element: appending at the end would be an edit at the same anchor.
    anchor = anchors[position % (len(anchors) - 1)]
    suggestion_ops = agent.insert_text(addition, after=anchor, suggestion_id=SUGGESTION)

    human.append_text("!")
    visible = human.view().anchors
    for index in deletes:
        current = human.view().anchors
        if current:
            human.delete_run(current[index % len(current)])
    del visible

    human.receive(list(agent.ops()))
    agent.receive(list(human.ops()))
    return human, agent, anchor, suggestion_ops


@given(BASE, POSITION, ADDITION, DELETES)
def test_a_rejected_suggestion_leaves_the_main_view_byte_identical(
    base: str, position: int, addition: str, deletes: list[int]
) -> None:
    human, agent, _, _ = _scenario(base, position, addition, deletes)
    human.decide(SUGGESTION, accept=False)
    agent.receive(list(human.ops()))

    assert human.state_hash() == agent.state_hash()
    assert human.view().text == _replay_without(human, SUGGESTION)


@given(BASE, POSITION, ADDITION, DELETES)
def test_a_pending_suggestion_is_invisible_in_the_main_view(
    base: str, position: int, addition: str, deletes: list[int]
) -> None:
    human, _, _, _ = _scenario(base, position, addition, deletes)
    assert human.view().text == _replay_without(human, SUGGESTION)
    assert SUGGESTION in human.state.pending_suggestions()
    assert addition in human.view(SUGGESTION).text


@given(BASE, POSITION, ADDITION, DELETES)
def test_an_accepted_suggestion_lands_after_its_anchor_or_its_surviving_predecessor(
    base: str, position: int, addition: str, deletes: list[int]
) -> None:
    human, agent, anchor, suggestion_ops = _scenario(base, position, addition, deletes)
    human.decide(SUGGESTION, accept=True)
    agent.receive(list(human.ops()))
    assert human.state_hash() == agent.state_hash()

    scope = human.state.main_scope()
    tree = human.state.text
    visible = [node.anchor_id for node in tree.visible_nodes(scope)]
    run = [str(op.dot) for op in suggestion_ops]

    positions = sorted(visible.index(item) for item in run)
    assert positions == list(range(positions[0], positions[0] + len(run)))

    # The nearest element at or before the anchor that is still visible.
    structural = [str(dot) for dot in tree.traversal()]
    anchor_index = structural.index(anchor)
    predecessor = next(
        (item for item in reversed(structural[: anchor_index + 1]) if item in visible), None
    )
    if predecessor is None:
        assert positions[0] == 0
    else:
        assert visible[positions[0] - 1] == predecessor


@given(BASE, POSITION, ADDITION, DELETES)
def test_accepting_and_then_rejecting_is_the_last_writer_that_wins(
    base: str, position: int, addition: str, deletes: list[int]
) -> None:
    """The accept flag is an ordinary register: decisions commute and the newest one wins."""
    human, agent, _, _ = _scenario(base, position, addition, deletes)
    human.decide(SUGGESTION, accept=True)
    human.decide(SUGGESTION, accept=False)
    agent.receive(list(human.ops()))
    # Delivering the two decisions in the other order reaches the same answer.
    third = make_replica(3, start_ms=1_700_001_000_000)
    third.receive(list(reversed(list(human.ops()))))

    assert human.state.status(SUGGESTION) == "rejected"
    assert third.state.status(SUGGESTION) == "rejected"
    assert third.state_hash() == human.state_hash()


@given(BASE, POSITION, ADDITION)
def test_two_suggestions_at_one_anchor_stay_separate(
    base: str, position: int, addition: str
) -> None:
    human = make_replica(1)
    first_agent = make_replica(2, start_ms=1_700_000_500_000)
    second_agent = make_replica(3, start_ms=1_700_001_000_000)
    human.insert_text(base)
    for agent in (first_agent, second_agent):
        agent.receive(list(human.ops()))

    anchor = human.view().anchors[position % len(human.view().anchors)]
    first_id, second_id = new_suggestion_id(), new_suggestion_id()
    first_ops = first_agent.insert_text(addition, after=anchor, suggestion_id=first_id)
    second_ops = second_agent.insert_text(addition, after=anchor, suggestion_id=second_id)

    for ops in (first_ops, second_ops):
        human.receive(ops)
    assert human.view().text == base

    human.decide(first_id, accept=True)
    with_first = human.view().text
    assert addition in with_first
    # The second suggestion is still invisible, and accepting the first did not pull it in.
    assert with_first.count(addition[0]) == base.count(addition[0]) + addition.count(addition[0])
