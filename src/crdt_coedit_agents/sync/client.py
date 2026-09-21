"""The client side of a synchronization session: a full replica plus a channel.

``ReplicaSession`` owns a :class:`~crdt_coedit_agents.replica.Replica` and keeps it in step with a
server over any :class:`~crdt_coedit_agents.sync.session.Channel`. Every incoming operation is
verified locally before it is applied, so a compromised server can withhold operations or reorder
them — the two failure modes a CRDT tolerates — but cannot inject forged edits.

Being offline is not a special mode: local edits are applied immediately and queue up, and
:meth:`ReplicaSession.push_pending` sends whatever accumulated the next time a channel is open.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from crdt_coedit_agents.crdt.ids import VersionVector
from crdt_coedit_agents.ops.model import MAX_OPS_PER_DELTA, SignedOp
from crdt_coedit_agents.replica import Replica, SyncReport
from crdt_coedit_agents.sync.protocol import (
    Ack,
    Delta,
    ErrorFrame,
    Hello,
    Ping,
    Pong,
    Pull,
    Push,
    ServerFrame,
    Welcome,
    decode_server_message,
    encode,
)
from crdt_coedit_agents.sync.session import Channel

__all__ = ["ReplicaSession", "ServerRejectedError", "SessionState"]


class ServerRejectedError(RuntimeError):
    """The server refused a frame or sent something the client cannot use."""


@dataclass(slots=True)
class SessionState:
    """What the client learned from the server on this connection."""

    connected: bool = False
    role: str = ""
    log_head: str = ""
    server_vector: dict[str, int] = field(default_factory=dict)
    acks: list[Ack] = field(default_factory=list)


class ReplicaSession:
    """Drives one replica against one channel."""

    def __init__(self, replica: Replica, channel: Channel) -> None:
        self._replica = replica
        self._channel = channel
        self._outbox: list[SignedOp] = []
        self.relayed = 0
        self.state = SessionState()

    @property
    def replica(self) -> Replica:
        return self._replica

    @property
    def pending(self) -> int:
        return len(self._outbox)

    def queue(self, ops: Sequence[SignedOp]) -> None:
        """Queue locally created operations for the next push (works while offline)."""
        self._outbox.extend(ops)

    async def hello(self) -> Welcome:
        """Open the session and apply everything the server has that this replica does not."""
        await self._channel.send(
            encode(
                Hello(
                    doc_id=self._replica.doc_id,
                    version_vector=self._replica.version_vector.as_dict(),
                )
            )
        )
        welcome = await self._expect(Welcome)
        self.state.connected = True
        self.state.role = welcome.role
        self.state.log_head = welcome.log_head
        self.state.server_vector = dict(welcome.version_vector)
        await self.drain_once()
        return welcome

    async def push_pending(self) -> Ack:
        """Send everything queued, in bounded batches, and return the last acknowledgement."""
        if not self._outbox:
            return Ack(accepted=0, duplicates=0, rejected={}, log_head=self.state.log_head)
        last: Ack | None = None
        pending, self._outbox = self._outbox, []
        for index in range(0, len(pending), MAX_OPS_PER_DELTA):
            batch = tuple(pending[index : index + MAX_OPS_PER_DELTA])
            await self._channel.send(encode(Push(ops=batch)))
            last = await self._expect(Ack)
            self.state.log_head = last.log_head
            self.state.acks.append(last)
        if last is None:  # pragma: no cover - the loop always runs at least once
            raise ServerRejectedError("push produced no acknowledgement")
        return last

    async def pull(self) -> SyncReport:
        """Ask for everything this replica is missing and apply it."""
        await self._channel.send(
            encode(Pull(version_vector=self._replica.version_vector.as_dict()))
        )
        return await self.drain_once()

    async def drain_once(self) -> SyncReport:
        """Wait for the solicited catch-up delta and apply it.

        Deltas the server pushed in the meantime are applied on the way, so a relayed batch is
        never mistaken for the answer to this request.
        """
        delta = await self._expect(Delta)
        return self._replica.receive(delta.ops)

    async def ping(self) -> None:
        await self._channel.send(encode(Ping()))
        await self._expect(Pong)

    async def _expect[T: ServerFrame](self, kind: type[T]) -> T:
        """Read frames until the expected one arrives, applying relayed deltas on the way."""
        while True:
            message = decode_server_message(await self._channel.receive())
            if isinstance(message, ErrorFrame):
                raise ServerRejectedError(f"{message.code}: {message.message}")
            if isinstance(message, Delta) and message.reason == "relay":
                self.relayed += self._replica.receive(message.ops).applied
                continue
            if not isinstance(message, kind):
                raise ServerRejectedError(f"expected {kind.__name__}, got {type(message).__name__}")
            return message

    def missing_for_server(self) -> list[SignedOp]:
        """Operations the server's last reported vector does not cover."""
        return self._replica.missing_for(VersionVector(self.state.server_vector))
