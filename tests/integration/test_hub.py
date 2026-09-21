"""The document hub: the checks every operation passes before it enters a document."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crdt_coedit_agents.agentic.audit import AuditLog, verify_file
from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import VersionVector
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.ops.model import FieldSet, SignedOp, TextInsert, UnsignedOp
from crdt_coedit_agents.replica import Replica, new_suggestion_id
from crdt_coedit_agents.sync.auth import Grant, Role
from crdt_coedit_agents.sync.hub import (
    DocumentHub,
    HubLimits,
    TooManySubscribersError,
)
from tests.conftest import make_replica, seeded_identity

pytestmark = pytest.mark.integration

DOC = "test-doc"


def grant(replica: Replica, role: Role, principal: str = "user:ana") -> Grant:
    return Grant(principal=principal, doc_id=DOC, role=role, replica_id=replica.replica_id)


def hub(tmp_path: Path | None = None, **limits: int) -> tuple[DocumentHub, AuditLog | None]:
    audit = AuditLog(tmp_path / "audit.jsonl") if tmp_path is not None else None
    return DocumentHub(OpLog(), limits=HubLimits(**limits), audit=audit), audit


def test_an_owner_can_write_and_the_server_applies_the_operations() -> None:
    document_hub, _ = hub()
    owner = make_replica(1)
    result = document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("hello"))
    assert result.accepted == 5
    assert result.ok
    assert document_hub.state(DOC).view().text == "hello"
    assert document_hub.version_vector(DOC) == owner.version_vector
    assert document_hub.verify_log(DOC).ok
    assert list(document_hub.documents()) == [DOC]


def test_a_forged_signature_is_refused_and_never_reaches_the_log(tmp_path: Path) -> None:
    document_hub, audit = hub(tmp_path)
    owner = make_replica(1)
    ops = owner.insert_text("a")
    forged = ops[0].model_copy(
        update={"op": ops[0].op.model_copy(update={"payload": TextInsert(value="z")})}
    )
    result = document_hub.submit(grant(owner, Role.OWNER), [forged])
    assert result.rejected == {"signature": 1}
    assert document_hub.op_log.count(DOC) == 0
    assert audit is not None
    assert any(
        json.loads(line)["action"] == "op.rejected"
        for line in audit.path.read_text(encoding="utf-8").splitlines()
    )


def test_a_suggester_cannot_write_main_operations_or_accept_flags(tmp_path: Path) -> None:
    document_hub, _ = hub(tmp_path)
    agent = make_replica(2)
    suggester = grant(agent, Role.SUGGESTER, principal="agent:one")

    untagged = document_hub.submit(suggester, agent.insert_text("untagged"))
    assert untagged.accepted == 0
    assert untagged.rejected == {"authorization": 8}

    suggestion_id = new_suggestion_id()
    tagged = document_hub.submit(
        suggester, agent.insert_text("tagged", suggestion_id=suggestion_id)
    )
    assert tagged.accepted == 6
    assert document_hub.state(DOC).view().text == ""

    decision = document_hub.submit(suggester, [agent.decide(suggestion_id, accept=True)])
    assert decision.rejected == {"authorization": 1}
    assert document_hub.state(DOC).view().text == ""


def test_an_editor_writes_the_document_but_cannot_decide() -> None:
    document_hub, _ = hub()
    editor = make_replica(3)
    assert document_hub.submit(grant(editor, Role.EDITOR), editor.insert_text("ok")).ok
    rejected = document_hub.submit(
        grant(editor, Role.EDITOR), [editor.decide(new_suggestion_id(), accept=True)]
    )
    assert rejected.rejected == {"authorization": 1}


def test_a_replay_of_a_spent_sequence_number_is_refused() -> None:
    document_hub, _ = hub()
    owner = make_replica(1)
    document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("a"))

    identity = seeded_identity(1)
    collision = identity.sign(
        UnsignedOp(
            doc_id=DOC,
            replica_id=identity.replica_id,
            seq=1,
            ts=Timestamp(9_999, 0, identity.replica_id).encode(),
            payload=FieldSet(key="title", value="forged"),
        )
    )
    result = document_hub.submit(grant(owner, Role.OWNER), [collision])
    assert result.rejected == {"replay": 1}
    assert document_hub.state(DOC).view().fields == {}


def test_resending_the_same_operations_counts_them_as_duplicates() -> None:
    document_hub, _ = hub()
    owner = make_replica(1)
    ops = owner.insert_text("hi")
    document_hub.submit(grant(owner, Role.OWNER), ops)
    again = document_hub.submit(grant(owner, Role.OWNER), ops)
    assert (again.accepted, again.duplicates) == (0, 2)
    assert document_hub.state(DOC).view().text == "hi"


def test_a_push_larger_than_the_limit_is_refused_whole() -> None:
    document_hub, _ = hub(max_ops_per_push=3)
    owner = make_replica(1)
    result = document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("abcd"))
    assert result.rejected == {"too_many_ops": 4}
    assert document_hub.op_log.count(DOC) == 0


def test_a_document_stops_growing_at_its_size_limit() -> None:
    document_hub, _ = hub(max_document_bytes=8, max_ops_per_push=64)
    owner = make_replica(1)
    assert document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("abcdefgh")).ok
    blocked = document_hub.submit(grant(owner, Role.OWNER), owner.append_text("more"))
    assert blocked.rejected == {"document_full": 4}


def test_catch_up_sends_exactly_what_a_version_vector_is_missing() -> None:
    document_hub, _ = hub()
    owner = make_replica(1)
    document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("abcde"))

    newcomer = make_replica(2, start_ms=1_700_000_500_000)
    missing = document_hub.catch_up(DOC, newcomer.version_vector)
    assert len(missing) == 5
    newcomer.receive(missing)
    assert newcomer.state_hash() == document_hub.state(DOC).state_hash()
    assert document_hub.catch_up(DOC, newcomer.version_vector) == []


def test_catch_up_is_bounded() -> None:
    document_hub, _ = hub(max_catch_up_ops=2, max_ops_per_push=64)
    owner = make_replica(1)
    document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("abcde"))
    assert len(document_hub.catch_up(DOC, VersionVector())) == 2


def test_accepted_operations_are_relayed_to_other_subscribers_only() -> None:
    async def scenario() -> tuple[int, int]:
        document_hub, _ = hub()
        owner, watcher = make_replica(1), make_replica(2, start_ms=1_700_000_500_000)
        author_feed = document_hub.subscribe(DOC, owner.replica_id)
        watcher_feed = document_hub.subscribe(DOC, watcher.replica_id)
        assert document_hub.subscriber_count(DOC) == 2

        document_hub.submit(grant(owner, Role.OWNER), owner.insert_text("hi"))
        relayed = watcher_feed.queue.get_nowait()
        watcher.receive(relayed)
        document_hub.unsubscribe(watcher_feed)
        document_hub.unsubscribe(watcher_feed)
        return author_feed.queue.qsize(), len(relayed)

    author_pending, relayed = asyncio.run(scenario())
    assert author_pending == 0
    assert relayed == 2


def test_a_subscriber_that_stops_reading_is_marked_for_disconnection() -> None:
    async def scenario() -> tuple[bool, int]:
        document_hub, _ = hub(subscriber_queue=1, max_ops_per_push=64)
        owner, slow = make_replica(1), make_replica(2, start_ms=1_700_000_500_000)
        feed = document_hub.subscribe(DOC, slow.replica_id)
        for _ in range(3):
            document_hub.submit(grant(owner, Role.OWNER), owner.append_text("x"))
        return feed.closed, feed.dropped

    closed, dropped = asyncio.run(scenario())
    assert closed is True
    assert dropped > 0


def test_a_document_refuses_more_subscribers_than_it_allows() -> None:
    async def scenario() -> None:
        document_hub, _ = hub(max_subscribers_per_doc=1)
        document_hub.subscribe(DOC, make_replica(1).replica_id)
        with pytest.raises(TooManySubscribersError, match="subscribers"):
            document_hub.subscribe(DOC, make_replica(2).replica_id)

    asyncio.run(scenario())


def test_the_hub_rebuilds_its_state_from_the_log_and_audits_decisions(tmp_path: Path) -> None:
    log = OpLog(tmp_path / "oplog.sqlite3")
    audit = AuditLog(tmp_path / "audit.jsonl")
    first = DocumentHub(log, audit=audit)

    owner = make_replica(1)
    agent = make_replica(2, start_ms=1_700_000_500_000)
    first.submit(grant(owner, Role.OWNER), owner.insert_text("hello"))
    agent.receive(first.catch_up(DOC, agent.version_vector))
    suggestion_id = new_suggestion_id()
    first.submit(
        grant(agent, Role.SUGGESTER, "agent:one"),
        agent.insert_text("!", after=agent.view().anchors[-1], suggestion_id=suggestion_id),
    )
    assert first.state(DOC).pending_suggestions() == (suggestion_id,)

    owner.receive(first.catch_up(DOC, owner.version_vector))
    first.submit(grant(owner, Role.OWNER), [owner.decide(suggestion_id, accept=True)])
    assert first.state(DOC).view().text == "hello!"

    reopened = DocumentHub(OpLog(tmp_path / "oplog.sqlite3"))
    assert reopened.state(DOC).state_hash() == first.state(DOC).state_hash()
    assert reopened.verify_log(DOC, expect_head=first.log_head(DOC)).ok

    entries = [json.loads(line) for line in audit.path.read_text(encoding="utf-8").splitlines()]
    assert any(entry["action"] == "suggestion.accepted" for entry in entries)
    assert verify_file(audit.path).ok


def test_operations_for_another_document_never_pass_authorization() -> None:
    document_hub, _ = hub()
    owner = make_replica(1, "other-doc")
    ops: list[SignedOp] = owner.insert_text("a")
    result = document_hub.submit(grant(owner, Role.OWNER), ops)
    assert result.rejected == {"authorization": 1}
