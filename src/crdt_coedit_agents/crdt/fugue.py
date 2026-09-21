"""Fugue: a sequence CRDT with maximal non-interleaving.

Based on Weidner and Kleppmann, *The Art of the Fugue: Minimizing Interleaving in Collaborative
Text Editing* (arXiv:2305.00583, 2023). This is an independent implementation of the tree
formulation described there; the code is original and the tie-break rule below is ours.

Structure
---------
Every inserted element is a tree node with a parent, a side (``L`` or ``R``) and an immutable
value. The list order is the depth-first traversal

    traverse(n) = [traverse(c) for c in left children of n] + [n]
                + [traverse(c) for c in right children of n]

and a virtual ``ROOT`` sentinel (``parent is None``) holds the top-level nodes on its right side.

To insert immediately after an existing element ``left``:

* if ``left`` has no right children, the new node becomes a **right child of ``left``**;
* otherwise the new node becomes a **left child of ``left``'s successor**, which is the leftmost
  descendant of ``left``'s first right child and therefore has no left children.

Exactly one of the two cases applies, so the position is always well defined, and in both cases
the new node is traversed immediately after ``left``.

Why runs do not interleave: typing a run produces a chain, because each new character is inserted
after the previous one, which has no right children yet. Two concurrent runs at the same position
become two chains hanging off the same node on the same side, and a depth-first traversal emits one
whole subtree before the other. That holds for runs typed left-to-right and right-to-left alike,
which is Fugue's improvement over RGA.

Tie-break rule
--------------
Siblings on the same side are ordered by ``(seq, replica_id)`` of their dot: **right** children
ascending, **left** children descending. The two directions mirror each other, so the structure is
symmetric under list reversal: among concurrent inserts immediately after a node the earliest is
closest to it, and among concurrent inserts immediately before a node the earliest is closest to it
as well.

Scopes
------
Every node and every tombstone carries a ``suggestion_id`` (``None`` for the main document).

* **Position resolution** happens inside the scope the edit is made in (main, or main plus one
  suggestion). A main edit therefore never becomes a child of a suggested node, so the set of main
  nodes is closed under "parent" and their relative order does not depend on any suggestion.
* **Materialization** walks the whole tree and emits only the nodes a view accepts. Because scopes
  are parent-closed, filtering the full traversal is the same as traversing the filtered tree, so
  rejecting a suggestion leaves the main view byte-identical to a run that never contained it.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from crdt_coedit_agents.crdt.ids import Dot

__all__ = [
    "ROOT_ID",
    "FugueNode",
    "FugueTree",
    "NodeConflictError",
    "Side",
    "UnknownAnchorError",
    "sibling_key",
]

ROOT_ID = "ROOT"

ScopeFilter = Callable[[str | None], bool]


def _main_only(suggestion_id: str | None) -> bool:
    return suggestion_id is None


class Side(StrEnum):
    LEFT = "L"
    RIGHT = "R"


class NodeConflictError(ValueError):
    """Two different nodes claim the same dot; operation identity has been violated."""


class UnknownAnchorError(KeyError):
    """An edit referenced an anchor id that is not in the tree (or not in the editing scope)."""


@dataclass(frozen=True, slots=True)
class FugueNode:
    """One immutable element of the sequence."""

    dot: Dot
    parent: Dot | None
    side: Side
    value: str
    suggestion_id: str | None = None

    @property
    def anchor_id(self) -> str:
        """The stable identifier agents and the CLI use to address this element."""
        return str(self.dot)


def sibling_key(dot: Dot) -> tuple[int, str]:
    """Total order on sibling dots: ``(seq, replica_id)``."""
    return (dot.seq, dot.replica_id)


class FugueTree:
    """A Fugue sequence. The join is the union of the node set and of the tombstone map."""

    __slots__ = ("_children", "_nodes", "_order", "_tombstones")

    def __init__(
        self,
        nodes: Mapping[Dot, FugueNode] | None = None,
        tombstones: Mapping[Dot, frozenset[str | None]] | None = None,
    ) -> None:
        self._nodes: dict[Dot, FugueNode] = {}
        self._tombstones: dict[Dot, frozenset[str | None]] = dict(tombstones or {})
        # parent dot (None = ROOT) -> (left children, right children), both ascending by
        # sibling_key. Left children are traversed in reverse.
        self._children: dict[Dot | None, tuple[list[Dot], list[Dot]]] = {}
        self._order: list[Dot] | None = None
        for node in (nodes or {}).values():
            self._attach(node)

    # ----------------------------------------------------------------- state

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FugueTree):
            return NotImplemented
        return self._nodes == other._nodes and self._tombstones == other._tombstones

    __hash__ = None  # type: ignore[assignment]  # mutable state

    def __repr__(self) -> str:
        return f"FugueTree(nodes={len(self._nodes)}, tombstoned={len(self._tombstones)})"

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, dot: object) -> bool:
        return dot in self._nodes

    @property
    def tombstone_count(self) -> int:
        return len(self._tombstones)

    def node(self, dot: Dot) -> FugueNode:
        try:
            return self._nodes[dot]
        except KeyError as exc:
            raise UnknownAnchorError(str(dot)) from exc

    def nodes(self) -> Iterator[FugueNode]:
        """Every node, in dot order (not list order)."""
        yield from (self._nodes[dot] for dot in sorted(self._nodes))

    def tombstones(self) -> Iterator[tuple[Dot, frozenset[str | None]]]:
        yield from ((dot, self._tombstones[dot]) for dot in sorted(self._tombstones))

    def copy(self) -> FugueTree:
        """Deep enough copy to mutate independently, without re-sorting every child list."""
        clone = FugueTree.__new__(FugueTree)
        clone._nodes = dict(self._nodes)
        clone._tombstones = dict(self._tombstones)
        clone._children = {
            parent: (list(left), list(right)) for parent, (left, right) in self._children.items()
        }
        clone._order = list(self._order) if self._order is not None else None
        return clone

    # ------------------------------------------------------------ mutations

    def _attach(self, node: FugueNode) -> bool:
        existing = self._nodes.get(node.dot)
        if existing is not None:
            if existing != node:
                raise NodeConflictError(
                    f"dot {node.dot} is already held by a different node; "
                    "two operations claim the same (replica, seq)"
                )
            return False
        self._nodes[node.dot] = node
        left, right = self._children.setdefault(node.parent, ([], []))
        bisect.insort(right if node.side is Side.RIGHT else left, node.dot, key=sibling_key)
        self._order = None
        return True

    def add_node(self, node: FugueNode) -> bool:
        """Add a node built elsewhere (a remote operation). Returns True when it was new."""
        return self._attach(node)

    def tombstone(self, target: Dot, suggestion_id: str | None = None) -> bool:
        """Mark ``target`` deleted in one scope. Returns True when that scope was new."""
        scopes = self._tombstones.get(target, frozenset())
        if suggestion_id in scopes:
            return False
        self._tombstones[target] = scopes | {suggestion_id}
        return True

    def join(self, other: FugueTree) -> FugueTree:
        """Union of node sets and of per-node tombstone scopes (the semilattice join)."""
        merged = self.copy()
        for node in other._nodes.values():
            merged._attach(node)
        for target, scopes in other._tombstones.items():
            merged._tombstones[target] = merged._tombstones.get(target, frozenset()) | scopes
        return merged

    # -------------------------------------------------------------- indexing

    def _kids(self, parent: Dot | None) -> tuple[list[Dot], list[Dot]]:
        return self._children.get(parent, ([], []))

    def _in_scope(self, dot: Dot, scope: ScopeFilter) -> bool:
        return scope(self._nodes[dot].suggestion_id)

    def insert_position(
        self, after: Dot | None, scope: ScopeFilter = _main_only
    ) -> tuple[Dot | None, Side]:
        """Where a node inserted immediately after ``after`` belongs (``None`` = at the start).

        The walk stays inside ``scope``, so the parent is always a node the editor can see.
        """
        if after is not None and after not in self._nodes:
            raise UnknownAnchorError(str(after))
        if after is not None and not self._in_scope(after, scope):
            raise UnknownAnchorError(f"{after} is outside the editing scope")
        _, right = self._kids(after)
        first_right = next((dot for dot in right if self._in_scope(dot, scope)), None)
        if first_right is None:
            return after, Side.RIGHT
        current = first_right
        while True:
            left, _ = self._kids(current)
            candidates = [dot for dot in left if self._in_scope(dot, scope)]
            if not candidates:
                return current, Side.LEFT
            current = candidates[-1]

    def insert_after(
        self,
        dot: Dot,
        value: str,
        *,
        after: Dot | None,
        suggestion_id: str | None = None,
        scope: ScopeFilter = _main_only,
    ) -> FugueNode:
        """Create and attach a node immediately after ``after`` within ``scope``."""
        parent, side = self.insert_position(after, scope)
        node = FugueNode(
            dot=dot, parent=parent, side=side, value=value, suggestion_id=suggestion_id
        )
        self._attach(node)
        return node

    # --------------------------------------------------------- materializing

    def invalidate_order(self) -> None:
        """Drop the cached traversal. Benchmarks use it to measure a cold walk."""
        self._order = None

    def traversal(self) -> list[Dot]:
        """Every reachable node in list order. Cached until the node set changes."""
        if self._order is None:
            self._order = self._compute_order()
        return self._order

    def _compute_order(self) -> list[Dot]:
        order: list[Dot] = []
        # Iterative depth-first walk: a right-leaning chain of 100k nodes would blow the
        # recursion limit. Stage 0 = descend into left children, stage 1 = emit and descend right.
        stack: list[tuple[Dot | None, int]] = [(None, 0)]
        while stack:
            dot, stage = stack.pop()
            left, right = self._kids(dot)
            if stage == 0:
                stack.append((dot, 1))
                # Left children are traversed descending, so push ascending to pop descending.
                stack.extend((child, 0) for child in left)
                continue
            if dot is not None:
                order.append(dot)
            # Right children are traversed ascending, so push descending to pop ascending.
            stack.extend((child, 0) for child in reversed(right))
        return order

    def is_visible(self, dot: Dot, view: ScopeFilter) -> bool:
        node = self._nodes[dot]
        if not view(node.suggestion_id):
            return False
        return not any(view(scope) for scope in self._tombstones.get(dot, ()))

    def visible_nodes(self, view: ScopeFilter = _main_only) -> list[FugueNode]:
        """Nodes of the view in list order."""
        return [self._nodes[dot] for dot in self.traversal() if self.is_visible(dot, view)]

    def materialize(self, view: ScopeFilter = _main_only) -> str:
        """The text of the view."""
        return "".join(node.value for node in self.visible_nodes(view))

    def anchors(self, view: ScopeFilter = _main_only) -> list[str]:
        """Stable anchor ids of the view, in list order."""
        return [node.anchor_id for node in self.visible_nodes(view)]

    def orphans(self) -> list[Dot]:
        """Nodes whose parent has not arrived yet; they are invisible until it does."""
        reachable = set(self.traversal())
        return sorted(dot for dot in self._nodes if dot not in reachable)

    @classmethod
    def from_items(
        cls, nodes: Iterable[FugueNode], tombstones: Iterable[tuple[Dot, frozenset[str | None]]]
    ) -> FugueTree:
        return cls({node.dot: node for node in nodes}, dict(tombstones))
