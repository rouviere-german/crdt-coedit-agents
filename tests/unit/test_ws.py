"""The live WebSocket adapter: URL building, frame handling and how the token travels.

The protocol and every check run over in-process channels elsewhere. What is left here is the
adapter itself, driven through a stand-in for the ``websockets`` connection object so no socket is
opened.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import websockets.asyncio.client
from websockets.exceptions import WebSocketException

from crdt_coedit_agents.sync.session import ChannelClosedError
from crdt_coedit_agents.sync.ws import (
    MAX_FRAME_BYTES,
    WebSocketClientChannel,
    open_channel,
    sync_url,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://127.0.0.1:8000", "ws://127.0.0.1:8000/v1/docs/notes/sync"),
        ("https://sync.example.test/", "wss://sync.example.test/v1/docs/notes/sync"),
        ("ws://localhost:9000", "ws://localhost:9000/v1/docs/notes/sync"),
        ("wss://sync.example.test//", "wss://sync.example.test/v1/docs/notes/sync"),
    ],
)
def test_the_sync_url_maps_http_schemes_to_websocket_schemes(base_url: str, expected: str) -> None:
    assert sync_url(base_url, "notes") == expected


@pytest.mark.parametrize("base_url", ["ftp://example.test", "example.test:8000", ""])
def test_a_base_url_without_a_supported_scheme_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError, match="must start with"):
        sync_url(base_url, "notes")


def test_the_token_is_never_part_of_the_sync_url() -> None:
    assert "token" not in sync_url("https://sync.example.test", "notes")


class FakeConnection:
    """Records what the adapter sends and replays scripted frames or failures."""

    def __init__(self, frames: list[Any] | None = None, *, fail: bool = False) -> None:
        self.frames = list(frames or [])
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None
        self.fail = fail

    async def send(self, text: str) -> None:
        if self.fail:
            raise WebSocketException("connection lost")
        self.sent.append(text)

    async def recv(self) -> Any:
        if self.fail:
            raise WebSocketException("connection lost")
        return self.frames.pop(0)

    async def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)
        if self.fail:
            raise WebSocketException("already gone")


def test_the_channel_passes_text_frames_through_and_truncates_close_reasons() -> None:
    connection = FakeConnection(['{"type": "welcome"}'])
    channel = WebSocketClientChannel(connection)

    async def scenario() -> str:
        await channel.send('{"type": "hello"}')
        received = await channel.receive()
        await channel.close(4000, "x" * 500)
        return received

    assert asyncio.run(scenario()) == '{"type": "welcome"}'
    assert connection.sent == ['{"type": "hello"}']
    assert connection.closed == (4000, "x" * 120)


def test_a_binary_frame_closes_the_channel() -> None:
    channel = WebSocketClientChannel(FakeConnection([b"\x00\x01"]))
    with pytest.raises(ChannelClosedError, match="binary frame"):
        asyncio.run(channel.receive())


def test_transport_failures_surface_as_a_closed_channel() -> None:
    channel = WebSocketClientChannel(FakeConnection(fail=True))
    with pytest.raises(ChannelClosedError, match="websocket closed"):
        asyncio.run(channel.send("{}"))
    with pytest.raises(ChannelClosedError, match="websocket closed"):
        asyncio.run(channel.receive())
    asyncio.run(channel.close())  # closing an already-dead socket is not an error


def test_open_channel_sends_the_token_in_a_header_and_bounds_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    connection = FakeConnection()

    @asynccontextmanager
    async def fake_connect(uri: str, **kwargs: Any) -> AsyncIterator[FakeConnection]:
        seen["uri"] = uri
        seen.update(kwargs)
        yield connection

    monkeypatch.setattr(websockets.asyncio.client, "connect", fake_connect)

    async def scenario() -> WebSocketClientChannel:
        async with open_channel(
            "https://sync.example.test", "notes", "secret-token", open_timeout=3.0
        ) as channel:
            return channel

    channel = asyncio.run(scenario())
    assert isinstance(channel, WebSocketClientChannel)
    assert seen["uri"] == "wss://sync.example.test/v1/docs/notes/sync"
    assert "secret-token" not in seen["uri"]
    assert seen["additional_headers"] == {"Authorization": "Bearer secret-token"}
    assert seen["max_size"] == MAX_FRAME_BYTES
    assert seen["open_timeout"] == 3.0
