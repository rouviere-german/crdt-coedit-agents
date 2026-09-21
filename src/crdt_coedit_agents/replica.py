"""A full replica: identity, clock, document state, operation history and delta synchronization.

Every participant — a human's CLI, an agent and the sync server itself — runs one of these. The
replica is the only place that *creates* operations: it resolves an anchor into a tree position
inside its own editing scope, signs the result and applies it locally, so a replica's own view is
always the one its operations describe.

Synchronization is delta-state: a peer sends its version vector, and the replica answers with the
operations that vector does not cover, chunked into bounded deltas. Incoming operations are
verified, replay-checked and applied in whatever order they arrive.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from crdt_coedit_agents.crdt.document import DocumentState, DocumentView
from crdt_coedit_agents.crdt.fugue import UnknownAnchorError
from crdt_coedit_agents.crdt.hlc import HybridLogicalClock, Timestamp
from crdt_coedit_agents.crdt.ids import Dot, VersionVector, parse_dot
from crdt_coedit_agents.ops.apply import OpRejectedError, apply_op
from crdt_coedit_agents.ops.model import (
    MAX_OPS_PER_DELTA,
    Delta,
    FieldSet,
    OpPayload,
    SignedOp,
    SuggestionDecide,
    TagAdd,
    TagRemove,
    TextDelete,
    TextInsert,
    UnsignedOp,
)
from crdt_coedit_agents.ops.signing import ReplicaIdentity, verify_op

__all__ = ["ReplayError", "Replica", "SyncReport", "new_suggestion_id"]


class ReplayError(ValueError):
    """A ``(replica_id, seq)`` pair was reused for a different operation."""


@dataclass(frozen=True, slots=True)
class SyncReport:
    """What a batch of incoming operations did."""

    received: int
    applied: int
    duplicates: int


def new_suggestion_id(rng: secrets.SystemRandom | None = None) -> str:
    """A fresh suggestion id. Unguessable so ids cannot be squatted by another replica."""
    source = rng if rng is not None else secrets.SystemRandom()
    return "sg-" + f"{source.getrandbits(64):016x}"


class Replica:
    """One replica of one document."""

    def __init__(
        self,
        identity: ReplicaIdentity,
        doc_id: str,
        *,
        state: DocumentState | None = None,
        clock: HybridLogicalClock | None = None,
    ) -> None:
        self._identity = identity
        self._doc_id = doc_id
        self._state = state if state is not None else DocumentState(doc_id=doc_id)
        self._clock = clock if clock is not None else HybridLogicalClock(identity.replica_id)
        self._ops: dict[Dot, SignedOp] = {}
        self._seen = VersionVector()
        self._gaps: dict[str, set[int]] = {}
        self._next_seq = 0

    # --------------------------------------------------------------- identity

    @property
    def replica_id(self) -> str:
        return self._identity.replica_id

    @property
    def doc_id(self) -> str:
        return self._doc_id

    @property
    def state(self) -> DocumentState:
        return self._state

    @property
    def version_vector(self) -> VersionVector:
        return self._seen.copy()

    @property
    def op_count(self) -> int:
        return len(self._ops)

    def ops(self) -> Iterator[SignedOp]:
        yield from (self._ops[dot] for dot in sorted(self._ops))

    def state_hash(self) -> str:
        return self._state.state_hash()

    def view(self, suggestion_id: str | None = None) -> DocumentView:
        """The main view, or the main view plus one suggestion."""
        scope = self._state.editing_scope(suggestion_id)
        return self._state.view(scope)

    # ------------------------------------------------------------ local edits

    def _next_dot(self) -> Dot:
        if self._next_seq == 0:
            self._next_seq = self._seen.get(self.replica_id)
        self._next_seq += 1
        return Dot(replica_id=self.replica_id, seq=self._next_seq)

    def _emit(self, payload: OpPayload, suggestion_id: str | None) -> SignedOp:
        dot = self._next_dot()
        timestamp: Timestamp = self._clock.tick()
        unsigned = UnsignedOp(
            doc_id=self._doc_id,
            replica_id=self.replica_id,
            seq=dot.seq,
            ts=timestamp.encode(),
            suggestion_id=suggestion_id,
            payload=payload,
        )
        signed = self._identity.sign(unsigned)
        self._store(signed)
        apply_op(self._state, signed)
        return signed

    def insert_text(
        self, text: str, *, after: str | None = None, suggestion_id: str | None = None
    ) -> list[SignedOp]:
        """Insert ``text`` one element at a time, starting immediately after ``after``.

        ``after=None`` inserts at the start of the document. Each element is anchored to the
        previous one, which produces the right-leaning chain that keeps concurrent runs from
        interleaving.
        """
        if not text:
            return []
        scope = self._state.editing_scope(suggestion_id)
        cursor: Dot | None = parse_dot(after) if after is not None else None
        emitted: list[SignedOp] = []
        for character in text:
            parent, side = self._state.text.insert_position(cursor, scope)
            payload = TextInsert(
                parent=str(parent) if parent is not None else None,
                side=side.value,
                value=character,
            )
            signed = self._emit(payload, suggestion_id)
            cursor = signed.dot
            emitted.append(signed)
        return emitted

    def append_text(self, text: str, *, suggestion_id: str | None = None) -> list[SignedOp]:
        """Insert ``text`` at the end of the current view."""
        anchors = self._state.text.anchors(self._state.editing_scope(suggestion_id))
        return self.insert_text(
            text, after=anchors[-1] if anchors else None, suggestion_id=suggestion_id
        )

    def delete_run(
        self, anchor: str, count: int = 1, *, suggestion_id: str | None = None
    ) -> list[SignedOp]:
        """Delete ``count`` visible elements starting at ``anchor`` (inclusive)."""
        if count < 1:
            raise ValueError("count must be at least 1")
        scope = self._state.editing_scope(suggestion_id)
        anchors = self._state.text.anchors(scope)
        try:
            start = anchors.index(anchor)
        except ValueError as exc:
            raise UnknownAnchorError(f"{anchor} is not visible in this scope") from exc
        targets = anchors[start : start + count]
        return [self._emit(TextDelete(target=item), suggestion_id) for item in targets]

    def set_field(
        self, key: str, value: JsonValue, *, suggestion_id: str | None = None
    ) -> SignedOp:
        return self._emit(FieldSet(key=key, value=value), suggestion_id)

    def add_tag(self, tag: str, *, suggestion_id: str | None = None) -> SignedOp:
        return self._emit(TagAdd(tag=tag), suggestion_id)

    def remove_tag(self, tag: str, *, suggestion_id: str | None = None) -> SignedOp:
        """Observed-remove: only the tags this replica can currently see are removed."""
        scope = self._state.editing_scope(suggestion_id)
        observed = self._state.tags.tags_for(tag, scope)
        payload = TagRemove(tag=tag, tags=tuple(str(dot) for dot in observed))
        return self._emit(payload, suggestion_id)

    def decide(self, suggestion_id: str, *, accept: bool, reason: str | None = None) -> SignedOp:
        """Accept or reject a suggestion. Always a main-scope operation."""
        return self._emit(
            SuggestionDecide(suggestion_id=suggestion_id, accept=accept, reason=reason), None
        )

    # ------------------------------------------------------------------ sync

    def _store(self, signed: SignedOp) -> bool:
        dot = signed.dot
        existing = self._ops.get(dot)
        if existing is not None:
            if existing != signed:
                raise ReplayError(
                    f"({dot.replica_id}, {dot.seq}) was already used by a different operation"
                )
            return False
        self._ops[dot] = signed
        self._observe(dot)
        return True

    def _observe(self, dot: Dot) -> None:
        if dot in self._seen:
            return
        gaps = self._gaps.setdefault(dot.replica_id, set())
        gaps.add(dot.seq)
        following = self._seen.get(dot.replica_id) + 1
        while following in gaps:
            gaps.discard(following)
            self._seen.observe(Dot(replica_id=dot.replica_id, seq=following))
            following += 1

    def has(self, dot: Dot) -> bool:
        return dot in self._ops

    def receive(self, ops: Iterable[SignedOp]) -> SyncReport:
        """Verify, replay-check and apply incoming operations. Order does not matter."""
        received = applied = duplicates = 0
        for signed in ops:
            received += 1
            if signed.op.doc_id != self._doc_id:
                raise OpRejectedError(
                    f"operation targets document {signed.op.doc_id!r}, not {self._doc_id!r}"
                )
            if self.has(signed.dot):
                if self._ops[signed.dot] != signed:
                    raise ReplayError(
                        f"({signed.dot.replica_id}, {signed.dot.seq}) was already used by a "
                        "different operation"
                    )
                duplicates += 1
                continue
            verify_op(signed)
            self._store(signed)
            if apply_op(self._state, signed):
                applied += 1
        return SyncReport(received=received, applied=applied, duplicates=duplicates)

    def receive_delta(self, delta: Delta) -> SyncReport:
        if delta.doc_id != self._doc_id:
            raise OpRejectedError(f"delta targets document {delta.doc_id!r}, not {self._doc_id!r}")
        return self.receive(delta.ops)

    def missing_for(self, peer: VersionVector) -> list[SignedOp]:
        """Operations this replica holds that ``peer``'s version vector does not cover."""
        return [self._ops[dot] for dot in sorted(self._ops) if dot not in peer]

    def deltas_for(self, peer: VersionVector, *, batch: int = MAX_OPS_PER_DELTA) -> list[Delta]:
        """The delta-state answer to "send me what I am missing", chunked to bounded batches."""
        if not 1 <= batch <= MAX_OPS_PER_DELTA:
            raise ValueError(f"batch must be within 1..{MAX_OPS_PER_DELTA}")
        missing = self.missing_for(peer)
        return [
            Delta(doc_id=self._doc_id, ops=tuple(missing[index : index + batch]))
            for index in range(0, len(missing), batch)
        ]

    def full_state_ops(self) -> Sequence[SignedOp]:
        """Every operation, i.e. what a full-state (non-delta) synchronization would send."""
        return list(self.ops())
