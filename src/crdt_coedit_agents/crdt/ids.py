"""Replica identifiers, operation dots and version vectors.

A *dot* is the pair ``(replica_id, seq)`` that names exactly one operation. Sequence numbers start
at 1 and are contiguous per replica, so the set of operations a replica has seen from every author
compresses into a *version vector*: ``{replica_id: highest contiguous seq}``.

Dots also name CRDT elements. A Fugue node and an OR-Set tag are identified by the dot of the
operation that created them, so element identity is global, stable and needs no coordination.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Self

__all__ = [
    "DOT_RANGE_PATTERN",
    "REPLICA_ID_PATTERN",
    "Dot",
    "VersionVector",
    "compress_dots",
    "format_dot",
    "is_replica_id",
    "parse_dot",
    "parse_dot_ranges",
]

# Replica ids are derived from a public key (see ops.signing), so they are fixed-width hex.
REPLICA_ID_PATTERN = r"^[0-9a-f]{32}$"
_REPLICA_ID = re.compile(REPLICA_ID_PATTERN)
_DOT = re.compile(r"^([0-9a-f]{32})\.([1-9][0-9]{0,17})$")

# A contiguous run of one replica's sequence numbers, inclusive at both ends. Used to name an exact
# *set* of operations compactly — see SuggestionDecide.sealed.
DOT_RANGE_PATTERN = r"^[0-9a-f]{32}\.[1-9][0-9]{0,17}-[1-9][0-9]{0,17}$"
_DOT_RANGE = re.compile(r"^([0-9a-f]{32})\.([1-9][0-9]{0,17})-([1-9][0-9]{0,17})$")


def is_replica_id(text: str) -> bool:
    """True when ``text`` really is a replica id, i.e. 32 lower-case hex characters."""
    return _REPLICA_ID.fullmatch(text) is not None


@dataclass(frozen=True, slots=True, order=True)
class Dot:
    """The globally unique name of one operation (and of the element it creates)."""

    replica_id: str
    seq: int

    def __post_init__(self) -> None:
        if not _REPLICA_ID.fullmatch(self.replica_id):
            raise ValueError(f"replica id must match {REPLICA_ID_PATTERN}")
        if self.seq < 1:
            raise ValueError("sequence numbers start at 1")

    def __str__(self) -> str:
        return f"{self.replica_id}.{self.seq}"


def format_dot(dot: Dot) -> str:
    """Wire form of a dot: ``<replica_id>.<seq>``."""
    return str(dot)


def parse_dot(text: str) -> Dot:
    """Parse the wire form of a dot. Raises ``ValueError`` on anything else."""
    match = _DOT.fullmatch(text)
    if match is None:
        raise ValueError(f"malformed dot: {text!r}")
    return Dot(replica_id=match.group(1), seq=int(match.group(2)))


def compress_dots(dots: Iterable[Dot]) -> tuple[str, ...]:
    """Name an exact set of dots as the shortest list of contiguous per-replica runs.

    A run ``"<replica_id>.<lo>-<hi>"`` is only emitted when *every* sequence number between ``lo``
    and ``hi`` is in the input, so expanding the runs gives back exactly the set that went in. That
    exactness is the point: the runs are what a decision seals, and a gap left by an operation the
    decider had not seen must stay a gap.
    """
    by_replica: dict[str, set[int]] = {}
    for dot in dots:
        by_replica.setdefault(dot.replica_id, set()).add(dot.seq)
    runs: list[str] = []
    for replica_id in sorted(by_replica):
        low: int | None = None
        previous: int | None = None
        for seq in sorted(by_replica[replica_id]):
            if low is None or previous is None:
                low = previous = seq
            elif seq == previous + 1:
                previous = seq
            else:
                runs.append(f"{replica_id}.{low}-{previous}")
                low = previous = seq
        if low is not None and previous is not None:
            runs.append(f"{replica_id}.{low}-{previous}")
    return tuple(runs)


def parse_dot_ranges(items: Iterable[str]) -> dict[str, tuple[tuple[int, int], ...]]:
    """Inverse of :func:`compress_dots`: ``replica_id -> ((low, high), ...)``."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        match = _DOT_RANGE.fullmatch(item)
        if match is None:
            raise ValueError(f"malformed dot range: {item!r}")
        low, high = int(match.group(2)), int(match.group(3))
        if low > high:
            raise ValueError(f"dot range runs backwards: {item!r}")
        ranges.setdefault(match.group(1), []).append((low, high))
    return {replica_id: tuple(items) for replica_id, items in ranges.items()}


class VersionVector:
    """Per-replica highest contiguous sequence number, i.e. a summary of "what I already have".

    The vector is a join-semilattice under element-wise maximum. It is only exact while each
    replica's operations arrive in sequence order; :meth:`observe` therefore advances an entry by
    one at a time and reports whether the operation closed a gap, so callers can buffer the rest.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Mapping[str, int] | None = None) -> None:
        self._entries: dict[str, int] = {}
        for replica_id, seq in (entries or {}).items():
            if not _REPLICA_ID.fullmatch(replica_id):
                raise ValueError(f"replica id must match {REPLICA_ID_PATTERN}")
            if seq < 0:
                raise ValueError("version vector entries must be non-negative")
            if seq:
                self._entries[replica_id] = seq

    @classmethod
    def from_dots(cls, dots: Iterator[Dot]) -> Self:
        """Build a vector from dots that are contiguous per replica."""
        vector = cls()
        for dot in sorted(dots):
            vector.observe(dot)
        return vector

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionVector):
            return NotImplemented
        return self._entries == other._entries

    def __hash__(self) -> int:
        return hash(frozenset(self._entries.items()))

    def __repr__(self) -> str:
        return f"VersionVector({dict(sorted(self._entries.items()))!r})"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, dot: Dot) -> bool:
        return dot.seq <= self._entries.get(dot.replica_id, 0)

    def get(self, replica_id: str) -> int:
        return self._entries.get(replica_id, 0)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self._entries.items()))

    def copy(self) -> VersionVector:
        return VersionVector(self._entries)

    def observe(self, dot: Dot) -> bool:
        """Record ``dot`` if it is the next one from its replica. Returns True when it advanced."""
        current = self._entries.get(dot.replica_id, 0)
        if dot.seq == current + 1:
            self._entries[dot.replica_id] = dot.seq
            return True
        return False

    def join(self, other: VersionVector) -> VersionVector:
        """Element-wise maximum (the semilattice join)."""
        merged = dict(self._entries)
        for replica_id, seq in other._entries.items():
            if seq > merged.get(replica_id, 0):
                merged[replica_id] = seq
        return VersionVector(merged)

    def dominates(self, other: VersionVector) -> bool:
        """True when this vector has seen everything ``other`` has."""
        return all(self.get(replica_id) >= seq for replica_id, seq in other._entries.items())

    def missing_ranges(self, other: VersionVector) -> dict[str, tuple[int, int]]:
        """Inclusive ``(low, high)`` sequence ranges ``other`` holds and this vector does not.

        This is the delta-state diff: the answer to "send me everything I am missing".
        """
        ranges: dict[str, tuple[int, int]] = {}
        for replica_id, seq in sorted(other._entries.items()):
            have = self.get(replica_id)
            if seq > have:
                ranges[replica_id] = (have + 1, seq)
        return ranges
