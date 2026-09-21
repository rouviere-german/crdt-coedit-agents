"""The composed document: scopes, views, digests and the serialized form."""

from __future__ import annotations

import pytest

from crdt_coedit_agents.crdt.document import DocumentState, Scope, SuggestionStatus
from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import Dot
from crdt_coedit_agents.crdt.lww import LWWEntry
from crdt_coedit_agents.crdt.orset import Tag
from tests.conftest import make_replica

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
    state = build()
    state.decisions.set(OTHER, LWWEntry(value=True, timestamp=Timestamp(1, 0, "a" * 32)))
    main = state.main_scope()
    assert main(None)
    assert main(OTHER)
    assert not main(SUGGESTION)

    editing = state.editing_scope(SUGGESTION)
    assert editing(None)
    assert editing(OTHER)
    assert editing(SUGGESTION)
    assert editing.name == f"suggestion:{SUGGESTION}"
    assert state.editing_scope(None) == main


def test_a_scope_is_a_plain_callable_filter() -> None:
    scope = Scope(name="x", accepted=frozenset({SUGGESTION}))
    assert scope(None) is True
    assert scope(SUGGESTION) is True
    assert scope(OTHER) is False


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
    second = make_replica(2, start_ms=1_700_000_500_000)
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
    payload["v"] = 2
    with pytest.raises(ValueError, match="unsupported document state version"):
        DocumentState.from_json(payload)
    payload["v"] = 1
    payload["doc_id"] = 5
    with pytest.raises(ValueError, match="doc_id must be a string"):
        DocumentState.from_json(payload)


def test_states_of_different_documents_cannot_be_joined() -> None:
    with pytest.raises(ValueError, match="different documents"):
        DocumentState(doc_id="a").join(DocumentState(doc_id="b"))


def test_size_bytes_counts_text_fields_and_tags() -> None:
    state = build()
    assert state.size_bytes() == len("hello") + len('"Notes"') + len("draft")
