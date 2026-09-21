"""Last-writer-wins register map keyed by string, with hybrid-logical-clock timestamps.

The state is one register per ``(key, suggestion_id)`` pair, where ``suggestion_id is None`` is the
main document. A register holds ``(value, timestamp)``; the join keeps the entry with the greater
timestamp. A timestamp already carries the writing replica id, so the comparison is a total order
and two different writes can never tie.

Materializing a view picks, for every key, the winning entry **among the visible scopes** — main
plus whichever suggestions the view accepts. Keeping suggested writes in their own register is what
makes rejection clean: hiding a suggestion's register simply lets the previous main value win
again, with no rewriting, re-signing or rebasing.

Deleting a key is writing ``None``: the tombstone is itself a timestamped entry, so a delete
concurrent with a write is resolved by timestamp instead of silently losing one of them.

Algebraically this is a product of independent per-register semilattices, so commutativity,
associativity and idempotence follow from the per-register maximum
(``tests/property/test_algebra.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from crdt_coedit_agents.crdt.hlc import Timestamp

__all__ = ["LWWEntry", "LWWMap", "RegisterKey", "VisibilityFilter"]

RegisterKey = tuple[str, str | None]
VisibilityFilter = Callable[[str | None], bool]


@dataclass(frozen=True, slots=True)
class LWWEntry:
    """One register reading: the value written and when."""

    value: JsonValue
    timestamp: Timestamp

    def wins_over(self, other: LWWEntry) -> bool:
        return self.timestamp > other.timestamp


class LWWMap:
    """A map of last-writer-wins registers, partitioned by suggestion scope."""

    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[RegisterKey, LWWEntry] | None = None) -> None:
        self._entries: dict[RegisterKey, LWWEntry] = dict(entries or {})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LWWMap):
            return NotImplemented
        return self._entries == other._entries

    __hash__ = None  # type: ignore[assignment]  # mutable state

    def __repr__(self) -> str:
        return f"LWWMap({dict(sorted(self._entries.items(), key=_sort_key))!r})"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, register: object) -> bool:
        return register in self._entries

    def items(self) -> Iterator[tuple[RegisterKey, LWWEntry]]:
        """Registers in a deterministic order (key, then main before suggestions)."""
        yield from sorted(self._entries.items(), key=_sort_key)

    def entry(self, key: str, suggestion_id: str | None = None) -> LWWEntry | None:
        return self._entries.get((key, suggestion_id))

    def keys(self) -> frozenset[str]:
        return frozenset(key for key, _ in self._entries)

    def copy(self) -> LWWMap:
        return LWWMap(self._entries)

    def set(self, key: str, entry: LWWEntry, suggestion_id: str | None = None) -> bool:
        """Apply one write. Returns True when it won the comparison and changed the state."""
        register: RegisterKey = (key, suggestion_id)
        current = self._entries.get(register)
        if current is not None and not entry.wins_over(current):
            return False
        self._entries[register] = entry
        return True

    def join(self, other: LWWMap) -> LWWMap:
        """Per-register maximum by timestamp (the semilattice join)."""
        merged = LWWMap(self._entries)
        for (key, suggestion_id), entry in other._entries.items():
            merged.set(key, entry, suggestion_id)
        return merged

    def winner(self, key: str, is_visible: VisibilityFilter) -> LWWEntry | None:
        """The entry that wins for ``key`` among the scopes ``is_visible`` accepts."""
        best: LWWEntry | None = None
        for (entry_key, suggestion_id), entry in self._entries.items():
            if entry_key != key or not is_visible(suggestion_id):
                continue
            if best is None or entry.wins_over(best):
                best = entry
        return best

    def materialize(
        self, is_visible: VisibilityFilter = lambda scope: scope is None
    ) -> dict[str, JsonValue]:
        """Materialize the visible map. Keys whose winning entry is ``None`` are absent."""
        result: dict[str, JsonValue] = {}
        for key in sorted(self.keys()):
            entry = self.winner(key, is_visible)
            if entry is not None and entry.value is not None:
                result[key] = entry.value
        return result


def _sort_key(item: tuple[RegisterKey, LWWEntry]) -> tuple[str, int, str]:
    (key, suggestion_id), _ = item
    return (key, 0 if suggestion_id is None else 1, suggestion_id or "")
