"""The Fugue tree: placement, traversal order, scopes and operation identity."""

from __future__ import annotations

import pytest

from crdt_coedit_agents.crdt.fugue import (
    FugueNode,
    FugueTree,
    NodeConflictError,
    Side,
    UnknownAnchorError,
    sibling_key,
)
from crdt_coedit_agents.crdt.ids import Dot

pytestmark = pytest.mark.unit

A = "a" * 32
B = "b" * 32
SUGGESTION = "sg-" + "1" * 16


def dot(seq: int, replica_id: str = A) -> Dot:
    return Dot(replica_id=replica_id, seq=seq)


def main_only(scope: str | None) -> bool:
    return scope is None


def everything(scope: str | None) -> bool:
    del scope
    return True


def type_text(tree: FugueTree, text: str, *, replica_id: str = A, start: int = 1) -> list[Dot]:
    """Type ``text`` left to right at the end, the way a person does."""
    cursor: Dot | None = tree.traversal()[-1] if tree.traversal() else None
    created: list[Dot] = []
    for offset, character in enumerate(text):
        node = tree.insert_after(dot(start + offset, replica_id), character, after=cursor)
        cursor = node.dot
        created.append(node.dot)
    return created


def test_typing_at_the_end_builds_a_right_leaning_chain() -> None:
    tree = FugueTree()
    dots = type_text(tree, "abc")
    assert tree.materialize() == "abc"
    assert tree.node(dots[0]).parent is None
    assert tree.node(dots[1]).parent == dots[0]
    assert tree.node(dots[1]).side is Side.RIGHT
    assert tree.node(dots[2]).parent == dots[1]


def test_inserting_between_two_elements_becomes_a_left_child_of_the_successor() -> None:
    tree = FugueTree()
    dots = type_text(tree, "ac")
    inserted = tree.insert_after(dot(9), "b", after=dots[0])
    assert inserted.parent == dots[1]
    assert inserted.side is Side.LEFT
    assert tree.materialize() == "abc"


def test_inserting_at_the_start_of_a_non_empty_document() -> None:
    tree = FugueTree()
    dots = type_text(tree, "bc")
    first = tree.insert_after(dot(9), "a", after=None)
    assert first.parent == dots[0]
    assert first.side is Side.LEFT
    assert tree.materialize() == "abc"


def test_repeated_inserts_at_one_anchor_are_backward_typing() -> None:
    tree = FugueTree()
    type_text(tree, "z")
    anchor = tree.traversal()[0]
    for offset, character in enumerate("cba"):
        tree.insert_after(dot(10 + offset), character, after=anchor)
    assert tree.materialize() == "zabc"


def test_deleting_hides_an_element_without_moving_the_others() -> None:
    tree = FugueTree()
    dots = type_text(tree, "abcd")
    assert tree.tombstone(dots[1]) is True
    assert tree.tombstone(dots[1]) is False
    assert tree.materialize() == "acd"
    assert tree.tombstone_count == 1
    # The tombstoned element is still a valid anchor for structural purposes.
    tree.insert_after(dot(50), "X", after=dots[1])
    assert tree.materialize() == "aXcd"


def test_a_node_whose_parent_has_not_arrived_is_invisible_until_it_does() -> None:
    tree = FugueTree()
    orphan = FugueNode(dot=dot(2), parent=dot(1), side=Side.RIGHT, value="b")
    tree.add_node(orphan)
    assert tree.materialize() == ""
    assert tree.orphans() == [dot(2)]

    tree.add_node(FugueNode(dot=dot(1), parent=None, side=Side.RIGHT, value="a"))
    assert tree.materialize() == "ab"
    assert tree.orphans() == []


def test_two_nodes_with_the_same_dot_are_refused() -> None:
    tree = FugueTree()
    tree.add_node(FugueNode(dot=dot(1), parent=None, side=Side.RIGHT, value="a"))
    assert tree.add_node(FugueNode(dot=dot(1), parent=None, side=Side.RIGHT, value="a")) is False
    with pytest.raises(NodeConflictError, match="same"):
        tree.add_node(FugueNode(dot=dot(1), parent=None, side=Side.RIGHT, value="b"))


def test_right_siblings_are_ordered_ascending_and_left_siblings_descending() -> None:
    tree = FugueTree()
    root = tree.insert_after(dot(1), "R", after=None)
    # Two concurrent inserts immediately after R: the earlier dot is closer to R.
    tree.add_node(FugueNode(dot=dot(5), parent=root.dot, side=Side.RIGHT, value="x"))
    tree.add_node(FugueNode(dot=dot(3), parent=root.dot, side=Side.RIGHT, value="y"))
    assert tree.materialize() == "Ryx"

    mirror = FugueTree()
    target = mirror.insert_after(dot(1), "T", after=None)
    mirror.add_node(FugueNode(dot=dot(5), parent=target.dot, side=Side.LEFT, value="x"))
    mirror.add_node(FugueNode(dot=dot(3), parent=target.dot, side=Side.LEFT, value="y"))
    assert mirror.materialize() == "xyT"
    assert sibling_key(dot(3)) < sibling_key(dot(5))


def test_a_suggestion_is_invisible_in_the_main_view_and_visible_in_its_own() -> None:
    tree = FugueTree()
    dots = type_text(tree, "ab")
    tree.insert_after(dot(9), "S", after=dots[0], suggestion_id=SUGGESTION)
    assert tree.materialize(main_only) == "ab"
    assert tree.materialize(everything) == "aSb"


def test_main_order_does_not_depend_on_a_suggestion_being_present() -> None:
    """A main edit resolved in the main scope never becomes a child of a suggested node."""
    with_suggestion = FugueTree()
    dots = type_text(with_suggestion, "ab")
    with_suggestion.insert_after(dot(9), "S", after=dots[0], suggestion_id=SUGGESTION)
    with_suggestion.insert_after(dot(20), "H", after=dots[0], scope=main_only)

    without = FugueTree()
    plain = type_text(without, "ab")
    without.insert_after(dot(20), "H", after=plain[0], scope=main_only)

    assert with_suggestion.materialize(main_only) == without.materialize(main_only) == "aHb"


def test_anchors_outside_the_editing_scope_are_refused() -> None:
    tree = FugueTree()
    dots = type_text(tree, "a")
    suggested = tree.insert_after(dot(9), "S", after=dots[0], suggestion_id=SUGGESTION)
    with pytest.raises(UnknownAnchorError, match="outside the editing scope"):
        tree.insert_position(suggested.dot, main_only)
    with pytest.raises(UnknownAnchorError):
        tree.insert_position(dot(999), main_only)
    with pytest.raises(UnknownAnchorError):
        tree.node(dot(999))


def test_join_merges_node_sets_and_tombstone_scopes() -> None:
    left = FugueTree()
    dots = type_text(left, "ab")
    right = left.copy()
    left.tombstone(dots[0])
    right.tombstone(dots[0], SUGGESTION)
    merged = left.join(right)
    assert dict(merged.tombstones())[dots[0]] == frozenset({None, SUGGESTION})
    assert merged == right.join(left)


def test_copies_are_independent_and_compare_by_value() -> None:
    tree = FugueTree()
    type_text(tree, "ab")
    clone = tree.copy()
    assert clone == tree
    assert tree != "not a tree"
    type_text(clone, "c", start=50)
    assert clone != tree
    assert len(tree) == 2
    assert dot(1) in tree
    assert repr(tree).startswith("FugueTree(")
    assert [node.value for node in tree.nodes()] == ["a", "b"]
    assert tree.anchors() == [str(dot(1)), str(dot(2))]


def test_a_deep_chain_does_not_exhaust_the_recursion_limit() -> None:
    tree = FugueTree()
    cursor: Dot | None = None
    for seq in range(1, 6_001):
        cursor = tree.insert_after(dot(seq), "x", after=cursor).dot
    assert len(tree.traversal()) == 6_000
    assert len(tree.materialize()) == 6_000


def test_from_items_rebuilds_an_identical_tree() -> None:
    tree = FugueTree()
    dots = type_text(tree, "abc", replica_id=B)
    tree.tombstone(dots[1], SUGGESTION)
    rebuilt = FugueTree.from_items(tree.nodes(), tree.tombstones())
    assert rebuilt == tree
    assert rebuilt.traversal() == tree.traversal()
