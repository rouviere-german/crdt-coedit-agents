"""Signed operations: the wire and storage form of every edit.

An operation is a small typed payload plus an envelope that names its author, its position in that
author's sequence, its hybrid-logical-clock timestamp and the suggestion it belongs to. The
envelope is canonically encoded and signed with the author's Ed25519 key, and the author's replica
id is derived from that key, so a forged signature, a spoofed replica id and a replayed
``(author, seq)`` pair are all detectable by any peer without asking a server.
"""

from crdt_coedit_agents.ops.apply import OpRejectedError, apply_op, can_apply
from crdt_coedit_agents.ops.model import (
    MAX_OPS_PER_DELTA,
    MAX_TEXT_VALUE_BYTES,
    Delta,
    FieldSet,
    OpKind,
    OpPayload,
    SignedOp,
    SuggestionDecide,
    TagAdd,
    TagRemove,
    TextDelete,
    TextInsert,
    UnsignedOp,
)
from crdt_coedit_agents.ops.signing import (
    InvalidSignatureError,
    ReplicaIdentity,
    SigningError,
    VerifyingKey,
    replica_id_for,
    verify_op,
)

__all__ = [
    "MAX_OPS_PER_DELTA",
    "MAX_TEXT_VALUE_BYTES",
    "Delta",
    "FieldSet",
    "InvalidSignatureError",
    "OpKind",
    "OpPayload",
    "OpRejectedError",
    "ReplicaIdentity",
    "SignedOp",
    "SigningError",
    "SuggestionDecide",
    "TagAdd",
    "TagRemove",
    "TextDelete",
    "TextInsert",
    "UnsignedOp",
    "VerifyingKey",
    "apply_op",
    "can_apply",
    "replica_id_for",
    "verify_op",
]
