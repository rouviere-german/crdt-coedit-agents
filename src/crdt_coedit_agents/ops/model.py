"""Typed operations and their signed envelope.

Six operation kinds cover the whole document:

============================ ======================================================
``text_insert``              add one character after an anchor (or at the start)
``text_delete``              tombstone one character
``field_set``                write one LWW register (``value: null`` deletes the key)
``tag_add`` / ``tag_remove`` OR-Set add and observed-remove
``suggestion_decide``        accept or reject a suggestion (main scope only)
============================ ======================================================

Every operation carries ``suggestion_id``. Operations with a suggestion id are invisible in the
main view until ``suggestion_decide`` sets that suggestion to ``true``; ``suggestion_decide``
itself may never carry one, because a suggestion that could accept itself would defeat the gate.

``suggestion_decide`` accepts a *set of operations*, not an id: its ``sealed`` field names exactly
the operations the decider could see. Operations added to the suggestion afterwards are outside the
seal and stay invisible for ever, so an approval cannot be widened after the fact.

Limits (payload size, ops per delta, field-value depth) are enforced here, in the model, so they
apply on every path: the server, a peer receiving a delta and a replica loading a snapshot.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import DOT_RANGE_PATTERN, REPLICA_ID_PATTERN, Dot, parse_dot

__all__ = [
    "MAX_DELTA_BYTES",
    "MAX_FIELD_DEPTH",
    "MAX_FIELD_VALUE_BYTES",
    "MAX_OPS_PER_DELTA",
    "MAX_SEAL_RUNS",
    "MAX_TAG_LENGTH",
    "MAX_TEXT_VALUE_BYTES",
    "Delta",
    "FieldSet",
    "OpKind",
    "OpPayload",
    "SignedOp",
    "SuggestionDecide",
    "TagAdd",
    "TagRemove",
    "TextDelete",
    "TextInsert",
    "UnsignedOp",
    "signing_payload",
]

MAX_TEXT_VALUE_BYTES = 8
MAX_FIELD_VALUE_BYTES = 4_096
MAX_FIELD_DEPTH = 8
MAX_TAG_LENGTH = 64
MAX_OPS_PER_DELTA = 512
MAX_DELTA_BYTES = 1_048_576
#: How many contiguous ``(replica, seq)`` runs one decision may seal. A suggestion written the
#: ordinary way needs one run per contributing replica; the bound is generous so that interleaved
#: authorship still fits, and refusing beyond it keeps a decision small enough to sign and relay.
MAX_SEAL_RUNS = 128

SUGGESTION_ID_PATTERN = r"^sg-[0-9a-f]{16}$"
FIELD_KEY_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
TAG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _/-]{0,63}$"
DOC_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
_ANCHOR = re.compile(r"^[0-9a-f]{32}\.[1-9][0-9]{0,17}$")
_DOT_RANGE = re.compile(DOT_RANGE_PATTERN)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _check_depth(value: JsonValue, depth: int = 0) -> None:
    if depth > MAX_FIELD_DEPTH:
        raise ValueError(f"field values may nest at most {MAX_FIELD_DEPTH} levels deep")
    if isinstance(value, dict):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


class TextInsert(_Frozen):
    """Add one element at a *resolved* tree position.

    The author turns "insert after anchor X" into ``(parent, side)`` against its own scope and puts
    the result on the wire. Receivers never re-resolve it, which is what makes the placement
    independent of what each replica happens to have seen. ``parent is None`` means the virtual
    root, so the element sits at the top level of the tree.
    """

    kind: Literal["text_insert"] = "text_insert"
    parent: str | None = None
    side: Literal["L", "R"] = "R"
    value: str = Field(min_length=1)

    @field_validator("parent")
    @classmethod
    def _valid_anchor(cls, value: str | None) -> str | None:
        if value is not None and not _ANCHOR.fullmatch(value):
            raise ValueError("parent must be an anchor id of the form <replica_id>.<seq>")
        return value

    @field_validator("value")
    @classmethod
    def _bounded_value(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_TEXT_VALUE_BYTES:
            raise ValueError(f"text values are limited to {MAX_TEXT_VALUE_BYTES} UTF-8 bytes")
        if "\x00" in value:
            raise ValueError("text values may not contain NUL")
        return value

    @property
    def parent_dot(self) -> Dot | None:
        return parse_dot(self.parent) if self.parent is not None else None


class TextDelete(_Frozen):
    """Tombstone the element named by ``target``."""

    kind: Literal["text_delete"] = "text_delete"
    target: str = Field(pattern=r"^[0-9a-f]{32}\.[1-9][0-9]{0,17}$")

    @property
    def target_dot(self) -> Dot:
        return parse_dot(self.target)


class FieldSet(_Frozen):
    """Write one LWW register. ``value: null`` removes the key from the view."""

    kind: Literal["field_set"] = "field_set"
    key: str = Field(pattern=FIELD_KEY_PATTERN)
    value: JsonValue = None

    @field_validator("value")
    @classmethod
    def _bounded_value(cls, value: JsonValue) -> JsonValue:
        _check_depth(value)
        from crdt_coedit_agents.core.canonical import canonical_json  # noqa: PLC0415

        if len(canonical_json(value).encode("utf-8")) > MAX_FIELD_VALUE_BYTES:
            raise ValueError(f"field values are limited to {MAX_FIELD_VALUE_BYTES} bytes")
        return value


class TagAdd(_Frozen):
    kind: Literal["tag_add"] = "tag_add"
    tag: str = Field(pattern=TAG_PATTERN, max_length=MAX_TAG_LENGTH)


class TagRemove(_Frozen):
    """Observed-remove: ``tags`` are the add-dots the remover could see."""

    kind: Literal["tag_remove"] = "tag_remove"
    tag: str = Field(pattern=TAG_PATTERN, max_length=MAX_TAG_LENGTH)
    tags: tuple[str, ...] = ()

    @field_validator("tags")
    @classmethod
    def _valid_dots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > MAX_OPS_PER_DELTA:
            raise ValueError("too many observed tags in one remove")
        for item in value:
            if not _ANCHOR.fullmatch(item):
                raise ValueError("observed tags must be dots of the form <replica_id>.<seq>")
        return value

    @property
    def observed(self) -> tuple[Dot, ...]:
        return tuple(parse_dot(item) for item in self.tags)


class SuggestionDecide(_Frozen):
    """Accept or reject a suggestion. Always a main-scope operation.

    ``sealed`` names the operations of the suggestion the decider had actually seen, as contiguous
    ``"<replica_id>.<low>-<high>"`` runs. Anything outside it is not part of the decision, so an
    empty seal accepts nothing at all — which is the right reading of a decision that does not say
    what it decided.
    """

    kind: Literal["suggestion_decide"] = "suggestion_decide"
    suggestion_id: str = Field(pattern=SUGGESTION_ID_PATTERN)
    accept: bool
    reason: str | None = Field(default=None, max_length=280)
    sealed: tuple[str, ...] = ()

    @field_validator("sealed")
    @classmethod
    def _valid_runs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > MAX_SEAL_RUNS:
            raise ValueError(f"a decision may seal at most {MAX_SEAL_RUNS} operation runs")
        for item in value:
            if not _DOT_RANGE.fullmatch(item):
                raise ValueError("sealed runs must look like <replica_id>.<low>-<high>")
            low, _, high = item.partition("-")
            if int(low.rsplit(".", 1)[1]) > int(high):
                raise ValueError("a sealed run may not run backwards")
        return value


OpPayload = Annotated[
    TextInsert | TextDelete | FieldSet | TagAdd | TagRemove | SuggestionDecide,
    Field(discriminator="kind"),
]
OpKind = Literal[
    "text_insert", "text_delete", "field_set", "tag_add", "tag_remove", "suggestion_decide"
]
#: Operation kinds a suggester may emit. ``suggestion_decide`` is deliberately absent.
SUGGESTABLE_KINDS: frozenset[str] = frozenset(
    {"text_insert", "text_delete", "field_set", "tag_add", "tag_remove"}
)


class UnsignedOp(_Frozen):
    """Everything the signature covers. Canonical JSON of this model is the signed message."""

    v: Literal[1] = 1
    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    replica_id: str = Field(pattern=REPLICA_ID_PATTERN)
    seq: int = Field(ge=1, le=10**18)
    ts: str
    suggestion_id: str | None = Field(default=None, pattern=SUGGESTION_ID_PATTERN)
    payload: OpPayload

    @field_validator("ts")
    @classmethod
    def _valid_timestamp(cls, value: str) -> str:
        Timestamp.decode(value)
        return value

    @model_validator(mode="after")
    def _consistent(self) -> UnsignedOp:
        if Timestamp.decode(self.ts).replica_id != self.replica_id:
            raise ValueError("the timestamp's replica id must match the operation's author")
        if self.suggestion_id is not None and self.payload.kind not in SUGGESTABLE_KINDS:
            raise ValueError(f"{self.payload.kind} operations cannot belong to a suggestion")
        return self

    @property
    def dot(self) -> Dot:
        return Dot(replica_id=self.replica_id, seq=self.seq)

    @property
    def timestamp(self) -> Timestamp:
        return Timestamp.decode(self.ts)

    @property
    def kind(self) -> str:
        return self.payload.kind


def signing_payload(op: UnsignedOp) -> bytes:
    """The exact bytes an Ed25519 signature covers."""
    from crdt_coedit_agents.core.canonical import canonical_json  # noqa: PLC0415

    return b"crdt-coedit-op-v1\x00" + canonical_json(op.model_dump(mode="json")).encode("utf-8")


class SignedOp(_Frozen):
    """An operation with its author's public key and signature."""

    op: UnsignedOp
    public_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(pattern=r"^[0-9a-f]{128}$")

    @property
    def dot(self) -> Dot:
        return self.op.dot

    @property
    def kind(self) -> str:
        return self.op.kind

    @property
    def suggestion_id(self) -> str | None:
        return self.op.suggestion_id


class Delta(_Frozen):
    """A bounded batch of signed operations, the unit of delta-state synchronization."""

    v: Literal[1] = 1
    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    ops: tuple[SignedOp, ...] = Field(max_length=MAX_OPS_PER_DELTA)

    @model_validator(mode="after")
    def _same_document(self) -> Delta:
        for signed in self.ops:
            if signed.op.doc_id != self.doc_id:
                raise ValueError("every operation in a delta must belong to the delta's document")
        return self

    def encoded_size(self) -> int:
        return len(self.model_dump_json().encode("utf-8"))
