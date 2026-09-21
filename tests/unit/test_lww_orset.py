"""The two simple CRDTs: last-writer-wins registers and the observed-remove set."""

from __future__ import annotations

import pytest

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import Dot
from crdt_coedit_agents.crdt.lww import LWWEntry, LWWMap
from crdt_coedit_agents.crdt.orset import ORSet, Tag, TagRemoval

pytestmark = pytest.mark.unit

A = "a" * 32
B = "b" * 32
SUGGESTION = "sg-" + "1" * 16


def stamp(wall_ms: int, replica_id: str = A, counter: int = 0) -> Timestamp:
    return Timestamp(wall_ms, counter, replica_id)


def main_only(scope: str | None, author: Dot | None = None) -> bool:
    del author
    return scope is None


def with_suggestion(scope: str | None, author: Dot | None = None) -> bool:
    del author
    return scope is None or scope == SUGGESTION


# --------------------------------------------------------------------- LWW


def test_the_later_write_wins_and_an_older_one_is_ignored() -> None:
    register = LWWMap()
    assert register.set("title", LWWEntry("first", stamp(1))) is True
    assert register.set("title", LWWEntry("second", stamp(2))) is True
    assert register.set("title", LWWEntry("stale", stamp(1))) is False
    assert register.materialize() == {"title": "second"}


def test_equal_wall_times_are_broken_by_counter_then_replica_id() -> None:
    register = LWWMap()
    register.set("k", LWWEntry("a", stamp(1, A, counter=0)))
    register.set("k", LWWEntry("b", stamp(1, A, counter=1)))
    assert register.materialize()["k"] == "b"

    other = LWWMap()
    other.set("k", LWWEntry("from-a", stamp(1, A)))
    other.set("k", LWWEntry("from-b", stamp(1, B)))
    assert other.materialize()["k"] == "from-b"


def test_writing_none_removes_the_key_from_the_view_but_keeps_the_tombstone() -> None:
    register = LWWMap()
    register.set("status", LWWEntry("open", stamp(1)))
    register.set("status", LWWEntry(None, stamp(2)))
    assert register.materialize() == {}
    assert register.entry("status") is not None
    # A concurrent later write brings it back; the delete did not lose the register.
    register.set("status", LWWEntry("closed", stamp(3)))
    assert register.materialize() == {"status": "closed"}


def test_a_suggested_write_is_a_separate_register_so_rejecting_restores_the_main_value() -> None:
    register = LWWMap()
    register.set("status", LWWEntry("draft", stamp(1)))
    register.set("status", LWWEntry("in-review", stamp(9)), SUGGESTION)

    assert register.materialize(main_only) == {"status": "draft"}
    assert register.materialize(with_suggestion) == {"status": "in-review"}


def test_the_visible_winner_is_chosen_among_visible_scopes_only() -> None:
    register = LWWMap()
    register.set("owner", LWWEntry("ana", stamp(5)))
    register.set("owner", LWWEntry("bo", stamp(3)), SUGGESTION)
    # The suggestion is older, so even when it is visible the main write still wins.
    assert register.materialize(with_suggestion) == {"owner": "ana"}
    assert register.winner("missing", with_suggestion) is None


def test_registers_compare_and_copy_by_value() -> None:
    register = LWWMap()
    register.set("k", LWWEntry(1, stamp(1)))
    clone = register.copy()
    assert clone == register
    assert register != "not a map"
    clone.set("k", LWWEntry(2, stamp(2)))
    assert clone != register
    assert len(register) == 1
    assert ("k", None) in register
    assert register.keys() == frozenset({"k"})
    assert repr(register).startswith("LWWMap(")
    assert [key for key, _ in register.items()] == [("k", None)]


# ------------------------------------------------------------------- OR-Set


def dot(seq: int, replica_id: str = A) -> Dot:
    return Dot(replica_id=replica_id, seq=seq)


def test_an_element_is_present_while_any_of_its_tags_survives() -> None:
    tags = ORSet()
    tags.add(Tag(element="draft", dot=dot(1)))
    tags.add(Tag(element="draft", dot=dot(2)))
    tags.remove(TagRemoval(dot=dot(1)))
    assert tags.materialize() == ["draft"]
    tags.remove(TagRemoval(dot=dot(2)))
    assert tags.materialize() == []


def test_a_concurrent_add_beats_a_remove_that_never_saw_it() -> None:
    first, second = ORSet(), ORSet()
    first.add(Tag(element="urgent", dot=dot(1)))
    second_state = first.copy()
    # One replica removes what it observed; the other adds a fresh tag at the same time.
    second_state.remove(TagRemoval(dot=dot(1)))
    second.add(Tag(element="urgent", dot=dot(7, B)))

    merged = second_state.join(second)
    assert merged.materialize() == ["urgent"]
    assert merged == second.join(second_state)


def test_re_adding_after_a_remove_works_because_the_tag_is_new() -> None:
    tags = ORSet()
    tags.add(Tag(element="review", dot=dot(1)))
    tags.remove(TagRemoval(dot=dot(1)))
    assert tags.materialize() == []
    tags.add(Tag(element="review", dot=dot(2)))
    assert tags.materialize() == ["review"]


def test_suggested_tags_are_invisible_until_the_suggestion_is_visible() -> None:
    tags = ORSet()
    tags.add(Tag(element="agent", dot=dot(1), suggestion_id=SUGGESTION))
    assert tags.materialize(main_only) == []
    assert tags.materialize(with_suggestion) == ["agent"]


def test_a_suggested_remove_only_hides_the_tag_where_the_suggestion_is_visible() -> None:
    tags = ORSet()
    tags.add(Tag(element="draft", dot=dot(1)))
    tags.remove(TagRemoval(dot=dot(1), suggestion_id=SUGGESTION))
    assert tags.materialize(main_only) == ["draft"]
    assert tags.materialize(with_suggestion) == []


def test_removing_one_tag_in_two_scopes_keeps_both_removals_in_either_order() -> None:
    """Regression: a tag removed in main *and* inside a suggestion is two independent facts.

    Keying removals by the observed dot alone let the second removal overwrite the first, so the
    materialized set depended on which delivery order a replica happened to see. Found by
    ``test_replicas_converge_under_reordering_duplication_and_delay``.
    """
    suggested = TagRemoval(dot=dot(1), suggestion_id=SUGGESTION)
    main = TagRemoval(dot=dot(1))

    states = []
    for removals in ((suggested, main), (main, suggested)):
        tags = ORSet()
        tags.add(Tag(element="draft", dot=dot(1)))
        for removal in removals:
            assert tags.remove(removal) is True
        # Both scopes hide the tag, and neither removal erased the other.
        assert tags.removal_count == 2
        assert tags.materialize(main_only) == []
        assert tags.materialize(with_suggestion) == []
        states.append(tags)

    # Order of application cannot change the state, its hashable form, or a re-merge of it.
    assert states[0] == states[1]
    assert states[0].state_items() == states[1].state_items()
    assert states[0].join(states[1]) == states[0]


def test_a_removal_in_one_scope_leaves_the_other_scope_untouched() -> None:
    tags = ORSet()
    tags.add(Tag(element="draft", dot=dot(1)))
    tags.remove(TagRemoval(dot=dot(1), suggestion_id=SUGGESTION))
    # Only the suggestion hides it; the main document still sees the tag.
    assert tags.materialize(main_only) == ["draft"]
    assert tags.materialize(with_suggestion) == []
    # The same removal replayed is still a no-op, per scope.
    assert tags.remove(TagRemoval(dot=dot(1), suggestion_id=SUGGESTION)) is False
    assert tags.removal_count == 1


def test_tags_for_reports_what_a_remove_would_observe() -> None:
    tags = ORSet()
    tags.add(Tag(element="draft", dot=dot(1)))
    tags.add(Tag(element="draft", dot=dot(2)))
    tags.add(Tag(element="other", dot=dot(3)))
    assert tags.tags_for("draft", main_only) == [dot(1), dot(2)]
    tags.remove(TagRemoval(dot=dot(1)))
    assert tags.tags_for("draft", main_only) == [dot(2)]


def test_repeated_adds_and_removes_of_the_same_dot_are_no_ops() -> None:
    tags = ORSet()
    assert tags.add(Tag(element="x", dot=dot(1))) is True
    assert tags.add(Tag(element="x", dot=dot(1))) is False
    assert tags.remove(TagRemoval(dot=dot(1))) is True
    assert tags.remove(TagRemoval(dot=dot(1))) is False
    assert len(tags) == 1
    assert tags.removal_count == 1
    assert tags != "not a set"
    assert repr(tags).startswith("ORSet(")


def test_state_items_round_trip_through_from_items() -> None:
    tags = ORSet()
    tags.add(Tag(element="a", dot=dot(1)))
    tags.add(Tag(element="b", dot=dot(2), suggestion_id=SUGGESTION))
    tags.remove(TagRemoval(dot=dot(1)))
    adds, removes = tags.state_items()
    assert ORSet.from_items(adds, removes) == tags
