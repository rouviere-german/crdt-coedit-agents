"""Applying a signed operation to a document state.

Applying is a *total* function of the operation and is order-independent. An insert whose parent
has not arrived yet is still stored: the node is simply unreachable from the root, and therefore
invisible, until the parent shows up. The same holds for a delete of an element that has not
arrived and for an observed-remove of tags that have not arrived, because tombstones and removals
are keyed by the dot they refer to, not by a live object.

That is what lets the sync layer stay simple: deltas may arrive reordered, duplicated or in
several partial batches, and the state still converges. Applying an operation twice is a no-op,
which the return value reports so callers can count real changes.

Every element records the dot of the operation that created it — the node's own dot for an insert
or a tag add, and the operating dot for a delete, a field write or a tag removal. Visibility then
depends on *which operation* wrote something, which is what lets a decision seal in exactly the
operations a reviewer saw.
"""

from __future__ import annotations

from crdt_coedit_agents.crdt.document import DocumentState, decision_value
from crdt_coedit_agents.crdt.fugue import FugueNode, NodeConflictError, Side
from crdt_coedit_agents.crdt.lww import LWWEntry
from crdt_coedit_agents.crdt.orset import Tag, TagRemoval
from crdt_coedit_agents.ops.model import (
    FieldSet,
    SignedOp,
    SuggestionDecide,
    TagAdd,
    TagRemove,
    TextDelete,
    TextInsert,
)

__all__ = ["OpRejectedError", "apply_op", "can_apply"]


class OpRejectedError(ValueError):
    """An operation is well formed and signed but not applicable to this document."""


def can_apply(state: DocumentState, signed: SignedOp) -> None:
    """Raise :class:`OpRejectedError` if the operation does not belong to this document."""
    if signed.op.doc_id != state.doc_id:
        raise OpRejectedError(
            f"operation targets document {signed.op.doc_id!r}, not {state.doc_id!r}"
        )


def apply_op(state: DocumentState, signed: SignedOp) -> bool:
    """Apply ``signed`` to ``state`` in place. Returns True when the state changed.

    The caller is responsible for verifying the signature and for replay protection; this function
    assumes an authenticated operation and only enforces document-level semantics.
    """
    can_apply(state, signed)
    op = signed.op
    payload = op.payload
    suggestion = op.suggestion_id

    match payload:
        case TextInsert():
            node = FugueNode(
                dot=op.dot,
                parent=payload.parent_dot,
                side=Side(payload.side),
                value=payload.value,
                suggestion_id=suggestion,
            )
            try:
                return state.text.add_node(node)
            except NodeConflictError as exc:
                raise OpRejectedError(str(exc)) from exc
        case TextDelete():
            return state.text.tombstone(payload.target_dot, suggestion, op.dot)
        case FieldSet():
            entry = LWWEntry(value=payload.value, timestamp=op.timestamp, author=op.dot)
            return state.fields.set(payload.key, entry, suggestion)
        case TagAdd():
            return state.tags.add(Tag(element=payload.tag, dot=op.dot, suggestion_id=suggestion))
        case TagRemove():
            changed = False
            for dot in payload.observed:
                changed |= state.tags.remove(
                    TagRemoval(dot=dot, suggestion_id=suggestion, author=op.dot)
                )
            return changed
        case SuggestionDecide():
            entry = LWWEntry(
                value=decision_value(accept=payload.accept, seal=payload.sealed),
                timestamp=op.timestamp,
                author=op.dot,
            )
            return state.decisions.set(payload.suggestion_id, entry)

    raise OpRejectedError(f"unsupported operation kind {payload.kind!r}")  # pragma: no cover
