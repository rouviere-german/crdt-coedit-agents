"""The WebSocket wire protocol: four client messages and five server messages.

Every frame is a JSON object with a ``type`` discriminator and a bounded size. Clients open with
``hello`` carrying their version vector; the server answers ``welcome`` and then streams the
operations the vector does not cover, so a replica that was offline or partitioned catches up
without asking for a full state transfer.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from crdt_coedit_agents.crdt.ids import REPLICA_ID_PATTERN, VersionVector
from crdt_coedit_agents.ops.model import DOC_ID_PATTERN, MAX_OPS_PER_DELTA, SignedOp

__all__ = [
    "MAX_FRAME_BYTES",
    "Ack",
    "ClientFrame",
    "ClientMessage",
    "Delta",
    "ErrorFrame",
    "Hello",
    "Ping",
    "Pong",
    "ProtocolError",
    "Pull",
    "Push",
    "ServerFrame",
    "ServerMessage",
    "Welcome",
    "decode_client_message",
    "decode_server_message",
    "encode",
]

MAX_FRAME_BYTES = 1_048_576
MAX_VECTOR_ENTRIES = 256


class ProtocolError(ValueError):
    """A frame was too large, malformed or of an unknown type."""


class _Frame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Hello(_Frame):
    """Opening frame: "this is what I already have"."""

    type: Literal["hello"] = "hello"
    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    version_vector: dict[str, int] = Field(default_factory=dict, max_length=MAX_VECTOR_ENTRIES)

    def vector(self) -> VersionVector:
        return VersionVector(self.version_vector)


class Push(_Frame):
    """A batch of locally created operations."""

    type: Literal["push"] = "push"
    ops: tuple[SignedOp, ...] = Field(max_length=MAX_OPS_PER_DELTA)


class Pull(_Frame):
    """Ask for everything a version vector does not cover (used after a partition heals)."""

    type: Literal["pull"] = "pull"
    version_vector: dict[str, int] = Field(default_factory=dict, max_length=MAX_VECTOR_ENTRIES)

    def vector(self) -> VersionVector:
        return VersionVector(self.version_vector)


class Ping(_Frame):
    type: Literal["ping"] = "ping"


ClientMessage = Annotated[Hello | Push | Pull | Ping, Field(discriminator="type")]


class Welcome(_Frame):
    type: Literal["welcome"] = "welcome"
    doc_id: str
    replica_id: str = Field(pattern=REPLICA_ID_PATTERN)
    role: str
    version_vector: dict[str, int]
    log_head: str


class Delta(_Frame):
    """Operations sent to a subscriber.

    ``reason`` tells a solicited answer (to ``hello`` or ``pull``) apart from one the server pushed
    because another replica wrote something, so a client can wait for its own answer without
    mistaking a relayed batch for it.
    """

    type: Literal["delta"] = "delta"
    ops: tuple[SignedOp, ...]
    reason: Literal["catch_up", "relay"] = "relay"


class Ack(_Frame):
    """Result of a push. ``rejected`` reports counts by reason, never operation content."""

    type: Literal["ack"] = "ack"
    accepted: int
    duplicates: int
    rejected: dict[str, int] = Field(default_factory=dict)
    log_head: str


class ErrorFrame(_Frame):
    type: Literal["error"] = "error"
    code: str
    message: str


class Pong(_Frame):
    type: Literal["pong"] = "pong"


ServerFrame = Welcome | Delta | Ack | ErrorFrame | Pong
ClientFrame = Hello | Push | Pull | Ping
ServerMessage = Annotated[ServerFrame, Field(discriminator="type")]


_CLIENT_ADAPTER: TypeAdapter[ClientFrame] = TypeAdapter(ClientMessage)
_SERVER_ADAPTER: TypeAdapter[ServerFrame] = TypeAdapter(ServerMessage)


def decode_client_message(raw: str) -> ClientFrame:
    """Parse one client frame. Raises :class:`ProtocolError` with no attacker-controlled text."""
    if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        return _CLIENT_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        locations = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
        raise ProtocolError(f"malformed frame at: {', '.join(locations[:8])}") from exc


def decode_server_message(raw: str) -> ServerFrame:
    """Parse one server frame. Unknown or malformed frames raise instead of being ignored."""
    if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        return _SERVER_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        locations = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
        raise ProtocolError(f"malformed frame at: {', '.join(locations[:8])}") from exc


def encode(message: ServerFrame | ClientFrame) -> str:
    return message.model_dump_json()
