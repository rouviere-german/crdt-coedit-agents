"""The server side of one synchronization session, independent of the transport.

A *channel* is anything that can send and receive text frames and be closed. The WebSocket route
wraps a Starlette ``WebSocket`` in one; the offline demo wraps a pair of in-memory queues. Both run
exactly this state machine, so the demo exercises the real protocol and the real policy rather than
a simplified copy.

Session shape::

    client: hello(version_vector)
    server: welcome(...) then delta(...) with everything the vector did not cover
    client: push(ops) / pull(version_vector) / ping
    server: ack(...) / delta(...) / pong, plus delta(...) pushed from other replicas

Three bounds apply per connection: a token-bucket message rate limit, the maximum frame size
enforced while decoding, and the subscriber's outbound queue. Exceeding the queue closes the
connection instead of buffering without limit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from crdt_coedit_agents.core.logging import get_logger
from crdt_coedit_agents.core.ratelimit import RateLimiter
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.sync.auth import Grant
from crdt_coedit_agents.sync.hub import DocumentHub, TooManySubscribersError
from crdt_coedit_agents.sync.protocol import (
    Ack,
    Delta,
    ErrorFrame,
    Hello,
    Ping,
    Pong,
    ProtocolError,
    Pull,
    Push,
    ServerMessage,
    Welcome,
    decode_client_message,
    encode,
)

__all__ = [
    "CLOSE_POLICY",
    "CLOSE_RATE_LIMIT",
    "CLOSE_SLOW_CONSUMER",
    "Channel",
    "ChannelClosedError",
    "SessionLimits",
    "run_session",
]

# Application close codes (4000-4999 is the private range).
CLOSE_POLICY = 4403
CLOSE_RATE_LIMIT = 4429
CLOSE_SLOW_CONSUMER = 4430

_log = get_logger("crdt_coedit_agents.session")


class ChannelClosedError(Exception):
    """The peer went away."""


class Channel(Protocol):
    """A bidirectional text channel."""

    async def send(self, text: str) -> None: ...

    async def receive(self) -> str: ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


@dataclass(frozen=True, slots=True)
class SessionLimits:
    messages_per_second: float = 50.0
    message_burst: int = 200


async def _send(channel: Channel, message: ServerMessage) -> None:
    await channel.send(encode(message))


async def run_session(
    hub: DocumentHub,
    grant: Grant,
    channel: Channel,
    *,
    limits: SessionLimits | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Drive one authenticated session to completion. Returns when the channel closes."""
    bounds = limits if limits is not None else SessionLimits()
    limiter = (
        RateLimiter(bounds.messages_per_second, bounds.message_burst, max_keys=1, clock=clock)
        if clock is not None
        else RateLimiter(bounds.messages_per_second, bounds.message_burst, max_keys=1)
    )

    try:
        opening = decode_client_message(await channel.receive())
    except ProtocolError as exc:
        await _send(channel, ErrorFrame(code="malformed_frame", message=str(exc)))
        await channel.close(CLOSE_POLICY, "malformed hello")
        return
    except ChannelClosedError:
        return
    if not isinstance(opening, Hello) or opening.doc_id != grant.doc_id:
        await _send(
            channel,
            ErrorFrame(
                code="bad_hello", message="the first frame must be a hello for this document"
            ),
        )
        await channel.close(CLOSE_POLICY, "bad hello")
        return

    try:
        subscription = hub.subscribe(grant.doc_id, grant.replica_id)
    except TooManySubscribersError:
        await _send(channel, ErrorFrame(code="too_many_subscribers", message="document is busy"))
        await channel.close(CLOSE_POLICY, "too many subscribers")
        return

    try:
        await _send(
            channel,
            Welcome(
                doc_id=grant.doc_id,
                replica_id=grant.replica_id,
                role=grant.role.value,
                version_vector=hub.version_vector(grant.doc_id).as_dict(),
                log_head=hub.log_head(grant.doc_id),
            ),
        )
        await _send(
            channel,
            Delta(ops=tuple(hub.catch_up(grant.doc_id, opening.vector())), reason="catch_up"),
        )

        reader: asyncio.Future[object] = asyncio.ensure_future(
            _read_loop(hub, grant, channel, limiter)
        )
        pump: asyncio.Future[object] = asyncio.ensure_future(_pump(channel, subscription.queue))
        overflow: asyncio.Future[object] = asyncio.ensure_future(subscription.overflow.wait())
        workers: tuple[asyncio.Future[object], ...] = (reader, pump, overflow)
        try:
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Cancel every worker, including when this session itself is cancelled, so no task
            # outlives the connection it belongs to.
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        if overflow in done:
            _log.warning("slow_consumer", doc_id=grant.doc_id, principal=grant.principal)
            await channel.close(CLOSE_SLOW_CONSUMER, "outbound queue overflowed")
            return
        for task in done:
            if task.cancelled():  # pragma: no cover - only on an outer cancellation
                continue
            exception = task.exception()
            if isinstance(exception, _RateLimitedError):
                await channel.close(CLOSE_RATE_LIMIT, "message rate limit exceeded")
                return
            if exception is not None and not isinstance(exception, ChannelClosedError):
                raise exception
    finally:
        hub.unsubscribe(subscription)


class _RateLimitedError(Exception):
    pass


async def _pump(channel: Channel, queue: asyncio.Queue[tuple[SignedOp, ...]]) -> None:
    while True:
        ops = await queue.get()
        await _send(channel, Delta(ops=ops))


async def _read_loop(
    hub: DocumentHub, grant: Grant, channel: Channel, limiter: RateLimiter
) -> None:
    while True:
        raw = await channel.receive()
        if not limiter.check("connection").allowed:
            await _send(channel, ErrorFrame(code="rate_limited", message="too many frames"))
            raise _RateLimitedError
        try:
            message = decode_client_message(raw)
        except ProtocolError as exc:
            await _send(channel, ErrorFrame(code="malformed_frame", message=str(exc)))
            continue
        match message:
            case Push():
                result = hub.submit(grant, message.ops)
                await _send(
                    channel,
                    Ack(
                        accepted=result.accepted,
                        duplicates=result.duplicates,
                        rejected=dict(result.rejected),
                        log_head=result.log_head,
                    ),
                )
            case Pull():
                await _send(
                    channel,
                    Delta(
                        ops=tuple(hub.catch_up(grant.doc_id, message.vector())),
                        reason="catch_up",
                    ),
                )
            case Ping():
                await _send(channel, Pong())
            case Hello():
                await _send(
                    channel,
                    ErrorFrame(code="duplicate_hello", message="hello may only be sent once"),
                )
