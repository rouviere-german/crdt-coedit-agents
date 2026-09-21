"""A pair of in-memory channels, so a client and the server can talk without a socket.

The demo and most integration tests use this instead of a real WebSocket. The frames, the session
state machine and every server-side check are the real ones; only the transport is replaced, which
keeps the offline demo honest and deterministic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from crdt_coedit_agents.sync.session import ChannelClosedError

__all__ = ["InProcessChannel", "channel_pair"]


@dataclass(slots=True)
class _Link:
    """The shared state of one channel pair."""

    to_server: asyncio.Queue[str]
    to_client: asyncio.Queue[str]
    closed: bool = False
    close_code: int = 0
    close_reason: str = ""


class InProcessChannel:
    """One end of a channel pair. ``receive`` raises :class:`ChannelClosedError` after a close."""

    def __init__(self, link: _Link, *, server_side: bool) -> None:
        self._link = link
        self._server_side = server_side

    @property
    def close_code(self) -> int:
        return self._link.close_code

    @property
    def close_reason(self) -> str:
        return self._link.close_reason

    @property
    def is_closed(self) -> bool:
        return self._link.closed

    def _outbound(self) -> asyncio.Queue[str]:
        return self._link.to_client if self._server_side else self._link.to_server

    def _inbound(self) -> asyncio.Queue[str]:
        return self._link.to_server if self._server_side else self._link.to_client

    async def send(self, text: str) -> None:
        if self._link.closed:
            raise ChannelClosedError("channel is closed")
        await self._outbound().put(text)

    async def receive(self) -> str:
        inbound = self._inbound()
        while True:
            if not inbound.empty():
                return inbound.get_nowait()
            if self._link.closed:
                raise ChannelClosedError("channel is closed")
            getter = asyncio.ensure_future(inbound.get())
            try:
                return await getter
            except asyncio.CancelledError:  # pragma: no cover - cancellation path
                getter.cancel()
                raise

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if not self._link.closed:
            self._link.closed = True
            self._link.close_code = code
            self._link.close_reason = reason

    def pending_inbound(self) -> int:
        return self._inbound().qsize()


def channel_pair(*, capacity: int = 0) -> tuple[InProcessChannel, InProcessChannel]:
    """Return ``(client_channel, server_channel)`` sharing one link.

    ``capacity`` bounds the queue toward the client, which is how a socket's send buffer behaves:
    a client that stops reading eventually blocks the server's writes. Zero means unbounded.
    """
    link = _Link(to_server=asyncio.Queue(), to_client=asyncio.Queue(maxsize=capacity))
    return InProcessChannel(link, server_side=False), InProcessChannel(link, server_side=True)
