"""Observed-remove set (Shapiro et al., "A comprehensive study of CRDTs", INRIA RR-7506, 2011).

Adding an element records a unique tag (the dot of the add operation). Removing an element records
the tags the remover had *observed* for it. The element is present when it has at least one tag
that has not been removed, so:

- add wins over a concurrent remove (the remover never saw the concurrent tag);
- re-adding after a remove works, because the new add carries a fresh tag;
- both halves of the state are grow-only sets, so the join is set union and is therefore
  commutative, associative and idempotent.

Tags are also partitioned by suggestion id: a suggested add is only visible in a view that accepts
its suggestion, and a suggested remove only hides tags for such views. The main view therefore sees
exactly the set it would see if the suggestion had never been made.

A removal is identified by the tag it observed, the scope it was made in *and* the operation that
made it, never by the tag alone. One tag can be removed independently in the main document and
inside a suggestion, and two replicas can remove the same tag in the same scope; those are separate
facts, and a state that could hold only one of them would depend on which arrived last. Keeping all
of them is what makes the removes half a set that only ever grows — and recording the author is
what lets an accepted suggestion seal in exactly the removals a reviewer saw.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from crdt_coedit_agents.crdt.ids import Dot
from crdt_coedit_agents.crdt.lww import VisibilityFilter, main_only

__all__ = ["ORSet", "RemovalKey", "RemovalMark", "Tag", "TagRemoval"]

# A removal is named by the tag it observed, the scope it was made in *and* its author. The same
# tag may be removed in the main document and inside a suggestion, and by two different replicas;
# those are distinct facts, and keying removals by the dot alone would let one overwrite another,
# which would make the state depend on delivery order.
type RemovalKey = tuple[Dot, str | None, Dot | None]

#: A removal as a view sees it: the scope it was made in and the operation that made it.
type RemovalMark = tuple[str | None, Dot | None]


def _removal_key(removal: TagRemoval) -> RemovalKey:
    return (removal.dot, removal.suggestion_id, removal.author)


def _removal_sort_key(key: RemovalKey) -> tuple[Dot, bool, str, bool, str, int]:
    """Total order over removal keys. ``None`` (the main scope) sorts before any suggestion."""
    dot, suggestion_id, author = key
    return (
        dot,
        suggestion_id is not None,
        suggestion_id or "",
        author is not None,
        author.replica_id if author is not None else "",
        author.seq if author is not None else 0,
    )


@dataclass(frozen=True, slots=True)
class Tag:
    """One add: which element, named by the dot of the add operation, in which scope."""

    element: str
    dot: Dot
    suggestion_id: str | None = None


@dataclass(frozen=True, slots=True)
class TagRemoval:
    """One remove: the tag observed, the scope it was made in, and the operation that made it."""

    dot: Dot
    suggestion_id: str | None = None
    author: Dot | None = None


class ORSet:
    """Observed-remove set of strings with per-suggestion visibility."""

    __slots__ = ("_adds", "_removed_scopes", "_removes")

    def __init__(
        self,
        adds: Mapping[Dot, Tag] | None = None,
        removes: Mapping[RemovalKey, TagRemoval] | None = None,
    ) -> None:
        self._adds: dict[Dot, Tag] = dict(adds or {})
        self._removes: dict[RemovalKey, TagRemoval] = dict(removes or {})
        # Derived index: which scopes each tag was removed in. Keeps the removal test O(1) instead
        # of a scan over every removal, and is rebuilt from ``_removes`` alone, so it can never
        # disagree with it.
        self._removed_scopes: dict[Dot, frozenset[RemovalMark]] = {}
        for dot, suggestion_id, author in self._removes:
            previous = self._removed_scopes.get(dot, frozenset())
            self._removed_scopes[dot] = previous | {(suggestion_id, author)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORSet):
            return NotImplemented
        return self._adds == other._adds and self._removes == other._removes

    __hash__ = None  # type: ignore[assignment]  # mutable state

    def __repr__(self) -> str:
        return f"ORSet(adds={len(self._adds)}, removes={len(self._removes)})"

    def __len__(self) -> int:
        return len(self._adds)

    @property
    def removal_count(self) -> int:
        return len(self._removes)

    def tags(self) -> Iterator[Tag]:
        yield from (self._adds[dot] for dot in sorted(self._adds))

    def tags_for(self, element: str, is_visible: VisibilityFilter) -> list[Dot]:
        """Visible, not-yet-removed tags of ``element`` — what a remove operation observes."""
        return sorted(
            tag.dot
            for tag in self._adds.values()
            if tag.element == element
            and is_visible(tag.suggestion_id, tag.dot)
            and not self._is_removed(tag.dot, is_visible)
        )

    def copy(self) -> ORSet:
        return ORSet(self._adds, self._removes)

    def add(self, tag: Tag) -> bool:
        """Record one add. Returns True when the tag was new."""
        if tag.dot in self._adds:
            return False
        self._adds[tag.dot] = tag
        return True

    def remove(self, removal: TagRemoval) -> bool:
        """Record one observed-remove. Returns True when the removal was new."""
        key = _removal_key(removal)
        if key in self._removes:
            return False
        self._removes[key] = removal
        previous = self._removed_scopes.get(removal.dot, frozenset())
        self._removed_scopes[removal.dot] = previous | {(removal.suggestion_id, removal.author)}
        return True

    def join(self, other: ORSet) -> ORSet:
        """Union of both grow-only halves (the semilattice join)."""
        return ORSet({**self._adds, **other._adds}, {**self._removes, **other._removes})

    def _is_removed(self, dot: Dot, is_visible: VisibilityFilter) -> bool:
        """True when the tag was removed by any operation this view accepts."""
        return any(is_visible(scope, author) for scope, author in self._removed_scopes.get(dot, ()))

    def materialize(self, is_visible: VisibilityFilter = main_only) -> list[str]:
        """Sorted elements that have at least one visible, not-removed tag."""
        present: set[str] = set()
        for tag in self._adds.values():
            if not is_visible(tag.suggestion_id, tag.dot) or self._is_removed(tag.dot, is_visible):
                continue
            present.add(tag.element)
        return sorted(present)

    def state_items(self) -> tuple[tuple[Tag, ...], tuple[TagRemoval, ...]]:
        """Deterministic snapshot of both halves, used by hashing and serialization."""
        adds = tuple(self._adds[dot] for dot in sorted(self._adds))
        removes = tuple(self._removes[key] for key in sorted(self._removes, key=_removal_sort_key))
        return adds, removes

    @classmethod
    def from_items(cls, adds: Iterable[Tag], removes: Iterable[TagRemoval]) -> ORSet:
        return cls(
            {tag.dot: tag for tag in adds},
            {_removal_key(item): item for item in removes},
        )
