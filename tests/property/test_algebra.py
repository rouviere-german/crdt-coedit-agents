"""The three semilattice laws, for every CRDT in the document and for the document itself.

If ``join`` is commutative, associative and idempotent, then any two replicas that have seen the
same set of operations hold the same state, whatever order the operations arrived in and however
often they were re-delivered. Everything else in this repository rests on that, so it is checked
directly rather than inferred from a few examples.

The laws are stated over states with *consistent operation identity*: each dot names one
operation. That is not an assumption the tests make for convenience — the Ed25519 envelope and the
unique ``(doc_id, replica_id, author_seq)`` index on the operation log enforce it, and
``tests/unit/test_fugue.py::test_two_nodes_with_the_same_dot_are_refused`` covers the violation.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from crdt_coedit_agents.crdt.document import DocumentState
from tests.property.strategies import document_states, state_triples

pytestmark = [pytest.mark.property, pytest.mark.unit]


@given(state_triples())
def test_text_join_is_commutative_associative_and_idempotent(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, third = (state.text for state in states)
    assert first.join(second) == second.join(first)
    assert first.join(second).join(third) == first.join(second.join(third))
    assert first.join(first) == first
    # The list order is part of the value, not just the node set.
    assert first.join(second).traversal() == second.join(first).traversal()


@given(state_triples())
def test_field_join_is_commutative_associative_and_idempotent(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, third = (state.fields for state in states)
    assert first.join(second) == second.join(first)
    assert first.join(second).join(third) == first.join(second.join(third))
    assert first.join(first) == first


@given(state_triples())
def test_tag_join_is_commutative_associative_and_idempotent(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, third = (state.tags for state in states)
    assert first.join(second) == second.join(first)
    assert first.join(second).join(third) == first.join(second.join(third))
    assert first.join(first) == first


@given(state_triples())
def test_document_join_is_commutative(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, _ = states
    assert first.join(second).state_hash() == second.join(first).state_hash()


@given(state_triples())
def test_document_join_is_associative(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, third = states
    assert (
        first.join(second).join(third).state_hash() == first.join(second.join(third)).state_hash()
    )


@given(document_states())
def test_document_join_is_idempotent(value: DocumentState) -> None:
    assert value.join(value).state_hash() == value.state_hash()


@given(state_triples())
def test_joining_does_not_mutate_either_side(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    first, second, _ = states
    before_first, before_second = first.state_hash(), second.state_hash()
    first.join(second)
    assert first.state_hash() == before_first
    assert second.state_hash() == before_second


@given(state_triples())
def test_the_join_of_all_three_is_what_every_replica_reaches(
    states: tuple[DocumentState, DocumentState, DocumentState],
) -> None:
    """Whatever order replicas merge in, they land on the same state and the same view."""
    first, second, third = states
    orders = [
        first.join(second).join(third),
        third.join(first).join(second),
        second.join(third).join(first),
        first.join(second.join(third)),
    ]
    hashes = {merged.state_hash() for merged in orders}
    texts = {merged.view().text for merged in orders}
    assert len(hashes) == 1
    assert len(texts) == 1
