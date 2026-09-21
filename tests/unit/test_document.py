"""The composed document: scopes, views, digests and the serialized form."""

from __future__ import annotations

import pytest

from crdt_coedit_agents.crdt.document import (
    STATE_VERSION,
    DocumentState,
    Scope,
    SuggestionStatus,
)
from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import Dot
from crdt_coedit_agents.crdt.orset import Tag, TagRemoval
from crdt_coedit_agents.ops.model import TextInsert, UnsignedOp
from tests.conftest import make_replica, seeded_identity

pytestmark = pytest.mark.unit

SUGGESTION = "sg-" + "1" * 16
OTHER = "sg-" + "2" * 16


def build() -> DocumentState:
    replica = make_replica(1)
    replica.insert_text("hello")
    replica.set_field("title", "Notes")
    replica.add_tag("draft")
    return replica.state


def test_the_main_view_is_json_like_and_hashes_its_content() -> None:
    view = build().view()
    assert view.as_json() == {
        "doc_id": "test-doc",
        "scope": "main",
        "text": "hello",
        "fields": {"title": "Notes"},
        "tags": ["draft"],
    }
    assert len(view.content_hash) == 64
    assert len(view.anchors) == 5


def test_a_suggestion_is_pending_until_it_is_decided() -> None:
    replica = make_replica(1)
    replica.insert_text("hi")
    replica.insert_text("!", after=replica.view().anchors[-1], suggestion_id=SUGGESTION)
    state = replica.state

    assert state.pending_suggestions() == (SUGGESTION,)
    assert state.status(SUGGESTION) == SuggestionStatus.PENDING
    assert state.view().text == "hi"

    replica.decide(SUGGESTION, accept=True)
    assert state.status(SUGGESTION) == SuggestionStatus.ACCEPTED
    assert state.accepted_suggestions() == frozenset({SUGGESTION})
    assert state.pending_suggestions() == ()
    assert state.view().text == "hi!"

    replica.decide(SUGGESTION, accept=False)
    assert state.status(SUGGESTION) == SuggestionStatus.REJECTED
    assert state.view().text == "hi"


def test_known_suggestions_finds_every_place_a_suggestion_can_appear() -> None:
    replica = make_replica(1)
    replica.insert_text("ab")
    anchors = replica.view().anchors
    replica.insert_text("c", after=anchors[0], suggestion_id=SUGGESTION)
    replica.delete_run(anchors[1], suggestion_id=OTHER)
    replica.set_field("status", "open", suggestion_id=SUGGESTION)
    replica.add_tag("agent", suggestion_id=OTHER)
    assert replica.state.known_suggestions() == frozenset({SUGGESTION, OTHER})


def test_editing_scopes_admit_main_accepted_and_the_suggestion_itself() -> None:
    replica = make_replica(1)
    replica.insert_text("hi")
    inside = replica.insert_text("!", after=replica.view().anchors[-1], suggestion_id=OTHER)
    replica.decide(OTHER, accept=True)
    state = replica.state

    main = state.main_scope()
    assert main(None, None)
    assert main(OTHER, inside[0].dot)
    assert not main(SUGGESTION, inside[0].dot)

    editing = state.editing_scope(SUGGESTION)
    assert editing(None, None)
    assert editing(OTHER, inside[0].dot)
    assert editing(SUGGESTION, None)
    assert editing.name == f"suggestion:{SUGGESTION}"
    assert state.editing_scope(None) == main


def test_a_scope_admits_an_accepted_suggestion_only_inside_its_seal() -> None:
    author = "b" * 32
    scope = Scope(
        name="x", accepted=frozenset({SUGGESTION}), seals={SUGGESTION: {author: ((3, 5),)}}
    )
    assert scope(None, None) is True
    assert scope(SUGGESTION, Dot(replica_id=author, seq=4)) is True
    assert scope(SUGGESTION, Dot(replica_id=author, seq=6)) is False
    assert scope(SUGGESTION, Dot(replica_id="c" * 32, seq=4)) is False
    # An element that does not say which operation made it cannot be proved to be sealed in.
    assert scope(SUGGESTION, None) is False
    assert scope(OTHER, Dot(replica_id=author, seq=4)) is False


def test_the_suggestion_digest_covers_only_that_suggestion() -> None:
    replica = make_replica(1)
    replica.insert_text("ab")
    anchors = replica.view().anchors
    replica.insert_text("XY", after=anchors[0], suggestion_id=SUGGESTION)
    replica.delete_run(anchors[1], suggestion_id=SUGGESTION)
    replica.set_field("status", "review", suggestion_id=SUGGESTION)
    replica.add_tag("agent", suggestion_id=SUGGESTION)
    state = replica.state

    digest = state.suggestion_digest(SUGGESTION)
    assert state.suggestion_text(SUGGESTION) == "XY"
    assert state.suggestion_deletions(SUGGESTION) == 1

    # A concurrent human edit does not change what the reviewer approved.
    replica.append_text("zzz")
    assert state.suggestion_digest(SUGGESTION) == digest
    # The agent adding more to the same suggestion does.
    replica.insert_text("!", after=anchors[0], suggestion_id=SUGGESTION)
    assert state.suggestion_digest(SUGGESTION) != digest


def test_the_state_hash_ignores_arrival_order_but_not_content() -> None:
    first = make_replica(1)
    second = make_replica(2, start_ms=1_700_000_010_000)
    first.insert_text("abc")
    second.insert_text("xyz")
    left, right = first.state.join(second.state), second.state.join(first.state)
    assert left.state_hash() == right.state_hash()

    changed = left.copy()
    changed.tags.add(Tag(element="new", dot=Dot(replica_id="c" * 32, seq=1)))
    assert changed.state_hash() != left.state_hash()


def test_the_state_round_trips_through_its_json_form() -> None:
    replica = make_replica(1)
    replica.insert_text("hello")
    replica.delete_run(replica.view().anchors[0])
    replica.set_field("title", {"nested": [1, 2]})
    replica.add_tag("draft")
    replica.remove_tag("draft")
    replica.insert_text("!", after=replica.view().anchors[-1], suggestion_id=SUGGESTION)
    replica.decide(SUGGESTION, accept=True)

    restored = DocumentState.from_json(replica.state.to_json())
    assert restored.state_hash() == replica.state_hash()
    assert restored.view().text == replica.view().text


def test_a_state_from_an_unknown_version_is_refused() -> None:
    payload = build().to_json()
    payload["v"] = STATE_VERSION + 1
    with pytest.raises(ValueError, match="unsupported document state version"):
        DocumentState.from_json(payload)
    payload["v"] = 1
    with pytest.raises(ValueError, match="unsupported document state version"):
        DocumentState.from_json(payload)
    payload["v"] = STATE_VERSION
    payload["doc_id"] = 5
    with pytest.raises(ValueError, match="doc_id must be a string"):
        DocumentState.from_json(payload)


def test_states_of_different_documents_cannot_be_joined() -> None:
    with pytest.raises(ValueError, match="different documents"):
        DocumentState(doc_id="a").join(DocumentState(doc_id="b"))


def test_size_bytes_counts_text_fields_and_tags() -> None:
    state = build()
    assert state.size_bytes() == len("hello") + len('"Notes"') + len("draft")


def test_a_suggestion_cannot_anchor_itself_inside_another_suggestion() -> None:
    """The probe a malicious peer would run: ``parent`` comes off the wire, so it is checked.

    Anchoring suggestion OTHER inside the still-undecided SUGGESTION is impossible through
    ``editing_scope``, but trivial by hand. Rejecting SUGGESTION and accepting OTHER must still
    leave the main view exactly as it would be had neither suggestion existed.
    """
    owner = make_replica(1)
    owner.insert_text("HELLO")
    agent = make_replica(2, start_ms=1_700_000_010_000)
    agent.receive(list(owner.ops()))
    inside = agent.insert_text("X", after=agent.view().anchors[-1], suggestion_id=SUGGESTION)

    identity = seeded_identity(2)
    forged_anchor = identity.sign(
        UnsignedOp(
            doc_id=owner.doc_id,
            replica_id=identity.replica_id,
            seq=inside[-1].op.seq + 1,
            ts=Timestamp(1_700_000_010_050, 0, identity.replica_id).encode(),
            suggestion_id=OTHER,
            payload=TextInsert(parent=str(inside[-1].dot), side="R", value="Y"),
        )
    )

    owner.receive([*inside, forged_anchor])
    owner.decide(SUGGESTION, accept=False)
    owner.decide(OTHER, accept=True)

    assert owner.view().text == "HELLO"
    counterfactual = make_replica(3, start_ms=1_700_000_020_000)
    counterfactual.receive(list(owner.ops()))
    assert counterfactual.view().text == "HELLO"


def test_a_withheld_operation_is_outside_the_seal_even_below_the_highest_sequence() -> None:
    """The seal names an exact set, not a high-water mark.

    An agent that holds one operation back until after the decision must not have it admitted just
    because a later operation of the same suggestion was reviewed.
    """
    owner = make_replica(1)
    owner.insert_text("Base.")
    agent = make_replica(2, start_ms=1_700_000_010_000)
    agent.receive(list(owner.ops()))
    # Three independent insertions, each anchored to a different main element, so withholding one
    # does not orphan the others.
    anchors = agent.view().anchors
    first = agent.insert_text("X", after=anchors[0], suggestion_id=SUGGESTION)
    withheld = agent.insert_text("Y", after=anchors[1], suggestion_id=SUGGESTION)
    third = agent.insert_text("Z", after=anchors[2], suggestion_id=SUGGESTION)

    owner.receive([*first, *third])
    owner.decide(SUGGESTION, accept=True)
    reviewed = owner.view().text
    assert "X" in reviewed
    assert "Z" in reviewed
    assert "Y" not in reviewed
    assert len(owner.state.seal_of(SUGGESTION)) == 2  # two runs, with the gap preserved

    owner.receive(withheld)

    assert owner.view().text == reviewed


def test_the_seal_records_every_kind_of_element_a_suggestion_can_contribute() -> None:
    replica = make_replica(1)
    replica.insert_text("ab")
    anchors = replica.view().anchors
    replica.insert_text("c", after=anchors[0], suggestion_id=SUGGESTION)
    replica.delete_run(anchors[1], suggestion_id=SUGGESTION)
    replica.set_field("status", "open", suggestion_id=SUGGESTION)
    replica.add_tag("agent", suggestion_id=SUGGESTION)
    replica.remove_tag("agent", suggestion_id=SUGGESTION)

    authors = replica.state.suggestion_authors(SUGGESTION)

    assert len(authors) == 5
    assert replica.state.suggestion_seal(SUGGESTION) == (f"{replica.replica_id}.3-7",)


def test_an_element_without_a_recorded_author_cannot_be_sealed() -> None:
    state = DocumentState(doc_id="test-doc")
    state.tags.add(Tag(element="x", dot=Dot(replica_id="a" * 32, seq=1), suggestion_id=SUGGESTION))
    state.tags.remove(
        TagRemoval(dot=Dot(replica_id="a" * 32, seq=1), suggestion_id=SUGGESTION, author=None)
    )
    with pytest.raises(ValueError, match="no recorded author"):
        state.suggestion_seal(SUGGESTION)
