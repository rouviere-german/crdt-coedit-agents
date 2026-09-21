"""A full synchronization session over in-process channels: the real protocol, no sockets."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import pytest

from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica, new_suggestion_id
from crdt_coedit_agents.sync.auth import Grant, Role
from crdt_coedit_agents.sync.client import ReplicaSession, ServerRejectedError
from crdt_coedit_agents.sync.hub import DocumentHub, HubLimits
from crdt_coedit_agents.sync.inproc import InProcessChannel, channel_pair
from crdt_coedit_agents.sync.protocol import (
    MAX_FRAME_BYTES,
    Ack,
    Delta,
    ErrorFrame,
    Hello,
    ProtocolError,
    Pull,
    ServerFrame,
    decode_client_message,
    decode_server_message,
    encode,
)
from crdt_coedit_agents.sync.session import (
    CLOSE_POLICY,
    CLOSE_RATE_LIMIT,
    CLOSE_SLOW_CONSUMER,
    SessionLimits,
    run_session,
)
from tests.conftest import make_replica

pytestmark = pytest.mark.integration

DOC = "test-doc"
T = TypeVar("T")


@dataclass
class Peer:
    session: ReplicaSession
    channel: InProcessChannel
    task: asyncio.Task[None]

    async def close(self) -> None:
        await self.channel.close()
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.task


def run[T](scenario: Callable[[], Coroutine[Any, Any, T]]) -> T:
    return asyncio.run(scenario())


def expect[T: ServerFrame](raw: str, kind: type[T]) -> T:
    """Decode one server frame and narrow it to the type the test is asserting about."""
    message = decode_server_message(raw)
    assert isinstance(message, kind), f"expected {kind.__name__}, got {type(message).__name__}"
    return message


def new_hub(**limits: int) -> DocumentHub:
    return DocumentHub(OpLog(), limits=HubLimits(**limits))


def new_hub_replica(index: int) -> Replica:
    return make_replica(index, start_ms=1_700_000_000_000 + index * 10_000)


async def connect(
    replica: Replica,
    role: Role,
    *,
    hub: DocumentHub,
    limits: SessionLimits | None = None,
    capacity: int = 0,
) -> Peer:
    grant = Grant(
        principal=f"user:{role.value}", doc_id=DOC, role=role, replica_id=replica.replica_id
    )
    client, server = channel_pair(capacity=capacity)
    task = asyncio.ensure_future(run_session(hub, grant, server, limits=limits))
    peer = Peer(session=ReplicaSession(replica, client), channel=client, task=task)
    await peer.session.hello()
    return peer


async def push(peer: Peer, ops: list[SignedOp]) -> Ack:
    peer.session.queue(ops)
    return await peer.session.push_pending()


def test_a_session_welcomes_catches_up_pushes_and_relays() -> None:
    async def scenario() -> tuple[str, str, int]:
        hub = new_hub()
        owner = await connect(new_hub_replica(1), Role.OWNER, hub=hub)
        welcome = owner.session.state
        assert welcome.role == "owner"

        ack = await push(owner, owner.session.replica.insert_text("hello"))
        assert ack.accepted == 5

        watcher = await connect(new_hub_replica(2), Role.EDITOR, hub=hub)
        assert watcher.session.replica.view().text == "hello"

        await push(owner, owner.session.replica.append_text("!"))
        # The relay arrives without being asked for; receive() waits for it.
        relayed = expect(await watcher.channel.receive(), Delta)
        assert relayed.reason == "relay"
        watcher.session.replica.receive(relayed.ops)
        text = watcher.session.replica.view().text

        await owner.close()
        await watcher.close()
        return text, hub.state(DOC).view().text, len(relayed.ops)

    watcher_text, server_text, relayed_ops = run(scenario)
    assert watcher_text == "hello!" == server_text
    assert relayed_ops == 1


def test_a_partitioned_replica_catches_up_when_it_reconnects() -> None:
    async def scenario() -> tuple[str, str, bool]:
        hub = new_hub()
        owner = await connect(new_hub_replica(1), Role.OWNER, hub=hub)
        away = new_hub_replica(2)
        first = await connect(away, Role.EDITOR, hub=hub)

        await push(owner, owner.session.replica.insert_text("shared"))
        await first.close()

        # Offline work: applied locally, queued for later.
        offline = away.append_text(" offline")
        await push(owner, owner.session.replica.append_text(" online"))
        diverged = away.state_hash() != hub.state(DOC).state_hash()

        rejoined = await connect(away, Role.EDITOR, hub=hub)
        await push(rejoined, offline)
        await owner.session.pull()
        await rejoined.session.pull()

        texts = (owner.session.replica.view().text, away.view().text)
        await owner.close()
        await rejoined.close()
        return texts[0], texts[1], diverged

    owner_text, away_text, diverged = run(scenario)
    assert diverged
    assert owner_text == away_text
    assert "offline" in owner_text
    assert "online" in owner_text


def test_a_suggester_session_is_refused_the_operations_it_may_not_write() -> None:
    async def scenario() -> tuple[dict[str, int], dict[str, int]]:
        hub = new_hub()
        agent = new_hub_replica(3)
        peer = await connect(agent, Role.SUGGESTER, hub=hub)
        untagged = await push(peer, agent.insert_text("main text"))
        decision = await push(peer, [agent.decide(new_suggestion_id(), accept=True)])
        await peer.close()
        return untagged.rejected, decision.rejected

    untagged, decision = run(scenario)
    assert untagged == {"authorization": 9}
    assert decision == {"authorization": 1}


def test_the_first_frame_must_be_a_hello_for_this_document() -> None:
    async def scenario() -> tuple[int, str]:
        hub = new_hub()
        grant = Grant(
            principal="u", doc_id=DOC, role=Role.OWNER, replica_id=make_replica(1).replica_id
        )
        client, server = channel_pair()
        task = asyncio.ensure_future(run_session(hub, grant, server))
        await client.send(encode(Pull()))
        frame = expect(await client.receive(), ErrorFrame)
        await task
        return client.close_code, frame.code

    code, error = run(scenario)
    assert code == CLOSE_POLICY
    assert error == "bad_hello"


def test_a_hello_for_another_document_is_refused() -> None:
    async def scenario() -> str:
        hub = new_hub()
        grant = Grant(
            principal="u", doc_id=DOC, role=Role.OWNER, replica_id=make_replica(1).replica_id
        )
        client, server = channel_pair()
        task = asyncio.ensure_future(run_session(hub, grant, server))
        await client.send(encode(Hello(doc_id="elsewhere")))
        frame = expect(await client.receive(), ErrorFrame)
        await task
        return frame.code

    assert run(scenario) == "bad_hello"


def test_a_malformed_opening_frame_closes_the_session() -> None:
    async def scenario() -> tuple[str, int]:
        hub = new_hub()
        grant = Grant(
            principal="u", doc_id=DOC, role=Role.OWNER, replica_id=make_replica(1).replica_id
        )
        client, server = channel_pair()
        task = asyncio.ensure_future(run_session(hub, grant, server))
        await client.send("{not json")
        frame = expect(await client.receive(), ErrorFrame)
        await task
        return frame.code, client.close_code

    code, close_code = run(scenario)
    assert code == "malformed_frame"
    assert close_code == CLOSE_POLICY


def test_a_malformed_frame_mid_session_is_reported_without_dropping_the_session() -> None:
    async def scenario() -> tuple[str, str]:
        hub = new_hub()
        owner = new_hub_replica(1)
        peer = await connect(owner, Role.OWNER, hub=hub)
        await peer.channel.send('{"type": "unknown"}')
        frame = expect(await peer.channel.receive(), ErrorFrame)
        # The session survives: a normal exchange still works.
        await push(peer, owner.insert_text("still here"))
        text = hub.state(DOC).view().text
        await peer.close()
        return frame.code, text

    code, text = run(scenario)
    assert code == "malformed_frame"
    assert text == "still here"


def test_a_second_hello_is_refused_without_resetting_the_session() -> None:
    async def scenario() -> str:
        hub = new_hub()
        peer = await connect(new_hub_replica(1), Role.OWNER, hub=hub)
        await peer.channel.send(encode(Hello(doc_id=DOC)))
        frame = expect(await peer.channel.receive(), ErrorFrame)
        await peer.close()
        return frame.code

    assert run(scenario) == "duplicate_hello"


def test_too_many_frames_closes_the_connection() -> None:
    async def scenario() -> int:
        hub = new_hub()
        owner = new_hub_replica(1)
        limits = SessionLimits(messages_per_second=0.001, message_burst=2)
        peer = await connect(owner, Role.OWNER, hub=hub, limits=limits)
        with contextlib.suppress(ServerRejectedError):
            for _ in range(5):
                await peer.session.ping()
        await peer.task
        code = peer.channel.close_code
        await peer.close()
        return code

    assert run(scenario) == CLOSE_RATE_LIMIT


def test_a_client_that_stops_reading_is_disconnected() -> None:
    async def scenario() -> int:
        hub = new_hub(subscriber_queue=1, max_ops_per_push=64)
        author = new_hub_replica(1)
        writer = await connect(author, Role.OWNER, hub=hub)
        # capacity=1: the client's inbound buffer fills, the pump blocks, the subscriber queue
        # overflows, and the session disconnects instead of buffering without limit.
        silent = await connect(new_hub_replica(2), Role.EDITOR, hub=hub, capacity=1)

        for _ in range(6):
            await push(writer, author.append_text("x"))
            await asyncio.sleep(0)
        await silent.task
        code = silent.channel.close_code
        await writer.close()
        await silent.close()
        return code

    assert run(scenario) == CLOSE_SLOW_CONSUMER


def test_frames_are_bounded_and_unknown_types_are_refused() -> None:
    with pytest.raises(ProtocolError, match="exceeds"):
        decode_client_message("x" * (MAX_FRAME_BYTES + 1))
    with pytest.raises(ProtocolError, match="malformed frame"):
        decode_client_message(json.dumps({"type": "nope"}))
    with pytest.raises(ProtocolError, match="malformed frame"):
        decode_server_message(json.dumps({"type": "nope"}))
    with pytest.raises(ProtocolError, match="exceeds"):
        decode_server_message("x" * (MAX_FRAME_BYTES + 1))
    assert isinstance(decode_client_message(encode(Hello(doc_id=DOC))), Hello)
