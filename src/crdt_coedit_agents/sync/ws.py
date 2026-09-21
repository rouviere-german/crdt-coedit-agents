"""A live WebSocket channel, for clients that talk to a running server.

This is the only place in the package that opens a socket. It is a thin adapter: the protocol,
the replica and every check live in :mod:`crdt_coedit_agents.sync.client` and
:mod:`crdt_coedit_agents.sync.session`, which are driven by in-process channels in the tests and
the demo, so the logic is exercised without a network.

The token travels in the ``Authorization`` header, never in the URL, so it cannot end up in a proxy
log or a browser history.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from crdt_coedit_agents.sync.session import ChannelClosedError

__all__ = ["WebSocketClientChannel", "open_channel", "sync_url"]

MAX_FRAME_BYTES = 1_048_576


def sync_url(base_url: str, doc_id: str) -> str:
    """Turn ``http(s)://host:port`` into the WebSocket URL of one document's sync endpoint."""
    stripped = base_url.rstrip("/")
    if stripped.startswith("https://"):
        scheme = "wss://" + stripped[len("https://") :]
    elif stripped.startswith("http://"):
        scheme = "ws://" + stripped[len("http://") :]
    elif stripped.startswith(("ws://", "wss://")):
        scheme = stripped
    else:
        raise ValueError("base_url must start with http://, https://, ws:// or wss://")
    return f"{scheme}/v1/docs/{doc_id}/sync"


class WebSocketClientChannel:
    """Adapts a ``websockets`` connection to the transport-independent channel protocol."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def send(self, text: str) -> None:
        import websockets  # noqa: PLC0415 - only needed when a socket is actually used

        try:
            await self._connection.send(text)
        except websockets.exceptions.WebSocketException as exc:
            raise ChannelClosedError("websocket closed") from exc

    async def receive(self) -> str:
        import websockets  # noqa: PLC0415

        try:
            frame = await self._connection.recv()
        except websockets.exceptions.WebSocketException as exc:
            raise ChannelClosedError("websocket closed") from exc
        if not isinstance(frame, str):
            raise ChannelClosedError("the server sent a binary frame; this protocol is text only")
        return frame

    async def close(self, code: int = 1000, reason: str = "") -> None:
        import websockets  # noqa: PLC0415

        with contextlib.suppress(websockets.exceptions.WebSocketException):
            await self._connection.close(code=code, reason=reason[:120])


@contextlib.asynccontextmanager
async def open_channel(
    base_url: str, doc_id: str, token: str, *, open_timeout: float = 10.0
) -> AsyncIterator[WebSocketClientChannel]:
    """Open an authenticated sync channel to a running server."""
    from websockets.asyncio.client import connect  # noqa: PLC0415

    async with connect(
        sync_url(base_url, doc_id),
        additional_headers={"Authorization": f"Bearer {token}"},
        max_size=MAX_FRAME_BYTES,
        open_timeout=open_timeout,
    ) as connection:
        yield WebSocketClientChannel(connection)
