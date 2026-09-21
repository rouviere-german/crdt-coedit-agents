"""Hypothesis strategies that build states a real system could actually reach.

The algebra laws only mean something on reachable states, so nothing here hand-builds a document.
Instead a small pool of *signed operations* is produced by real replicas doing real edits, and each
state is what a replica would hold after receiving some subset of that pool. That models exactly
the situation the laws are about: several replicas, different subsets, arbitrary order.

It also keeps operation identity honest. Every dot names one operation, which is what the
signature layer and the operation log enforce in production; a strategy that invented two different
nodes with the same dot would be testing a state the system refuses to enter.
"""

from __future__ import annotations

from collections.abc import Sequence

from hypothesis import strategies as st

from crdt_coedit_agents.crdt.document import DocumentState
from crdt_coedit_agents.ops.apply import apply_op
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica
from tests.conftest import make_replica

DOC_ID = "prop-doc"
SUGGESTIONS: tuple[str | None, ...] = (None, "sg-" + "1" * 16, "sg-" + "2" * 16)
KEYS = ("title", "status", "owner")
TAGS = ("draft", "urgent", "review")
WORDS = ("alpha", "beta", "gamma", "delta")

_ACTIONS = ("insert", "append", "delete", "field", "tag_add", "tag_remove", "decide")


def _act(replica: Replica, action: str, choice: int, scope: str | None) -> list[SignedOp]:
    """One edit, chosen from a small vocabulary. Returns the operations it produced."""
    view = replica.view(scope)
    match action:
        case "insert":
            anchor = view.anchors[choice % len(view.anchors)] if view.anchors else None
            return replica.insert_text(
                WORDS[choice % len(WORDS)][:2], after=anchor, suggestion_id=scope
            )
        case "append":
            return replica.append_text(WORDS[choice % len(WORDS)][:2], suggestion_id=scope)
        case "delete":
            if not view.anchors:
                return []
            return replica.delete_run(view.anchors[choice % len(view.anchors)], suggestion_id=scope)
        case "field":
            return [
                replica.set_field(
                    KEYS[choice % len(KEYS)], WORDS[choice % len(WORDS)], suggestion_id=scope
                )
            ]
        case "tag_add":
            return [replica.add_tag(TAGS[choice % len(TAGS)], suggestion_id=scope)]
        case "tag_remove":
            return [replica.remove_tag(TAGS[choice % len(TAGS)], suggestion_id=scope)]
    suggestion = SUGGESTIONS[1 + choice % 2]
    assert suggestion is not None
    return [replica.decide(suggestion, accept=choice % 2 == 0)]


@st.composite
def operation_pool(draw: st.DrawFn, *, max_edits: int = 14) -> list[SignedOp]:
    """Signed operations produced by three replicas editing the same document independently."""
    replicas = [make_replica(index, DOC_ID) for index in (1, 2, 3)]
    edits = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=2),
                st.sampled_from(_ACTIONS),
                st.integers(min_value=0, max_value=7),
                st.sampled_from(SUGGESTIONS),
            ),
            max_size=max_edits,
        )
    )
    pool: list[SignedOp] = []
    for replica_index, action, choice, scope in edits:
        replica = replicas[replica_index]
        scope_for_action = None if action == "decide" else scope
        pool.extend(_act(replica, action, choice, scope_for_action))
    return pool


def _subset(pool: Sequence[SignedOp], keep: Sequence[bool]) -> DocumentState:
    state = DocumentState(doc_id=DOC_ID)
    for wanted, signed in zip(keep, pool, strict=False):
        if wanted:
            apply_op(state, signed)
    return state


@st.composite
def document_states(draw: st.DrawFn) -> DocumentState:
    """One replica's state: some subset of a pool of real operations."""
    pool = draw(operation_pool())
    keep = draw(st.lists(st.booleans(), min_size=len(pool), max_size=len(pool)))
    return _subset(pool, keep)


@st.composite
def state_triples(draw: st.DrawFn) -> tuple[DocumentState, DocumentState, DocumentState]:
    """Three replicas that saw different subsets of the same operations."""
    pool = draw(operation_pool())
    subsets = [
        draw(st.lists(st.booleans(), min_size=len(pool), max_size=len(pool))) for _ in range(3)
    ]
    first, second, third = (_subset(pool, keep) for keep in subsets)
    return first, second, third


def merge_all(states: Sequence[DocumentState]) -> DocumentState:
    merged = states[0]
    for state in states[1:]:
        merged = merged.join(state)
    return merged
