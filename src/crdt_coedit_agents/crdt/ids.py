"""Replica identifiers, operation dots and version vectors.

A *dot* is the pair ``(replica_id, seq)`` that names exactly one operation. Sequence numbers start
at 1 and are contiguous per replica, so the set of operations a replica has seen from every author
compresses into a *version vector*: ``{replica_id: highest contiguous seq}``.

Dots also name CRDT elements. A Fugue node and an OR-Set tag are identified by the dot of the
operation that created them, so element identity is global, stable and needs no coordination.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Self

__all__ = [
    "REPLICA_ID_PATTERN",
    "Dot",
    "VersionVector",
    "format_dot",
    "parse_dot",
]

# Replica ids are derived from a public key (see ops.signing), so they are fixed-width hex.
REPLICA_ID_PATTERN = r"^[0-9a-f]{32}$"
_REPLICA_ID = re.compile(REPLICA_ID_PATTERN)
_DOT = re.compile(r"^([0-9a-f]{32})\.([1-9][0-9]{0,17})$")


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
