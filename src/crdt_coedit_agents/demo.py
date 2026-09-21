"""The offline end-to-end demo: `uv run python tasks.py demo`.

It runs the real sync service in process (the same hub, the same session state machine, the same
role policy) over in-memory channels, with three replicas:

* **owner** — a human, role ``owner``, the only one allowed to write accept flags;
* **drafter** — an agent, role ``suggester``, connected;
* **offline-agent** — a second agent that is partitioned away, keeps working, and reconnects.

The story: the owner writes a document; the drafter proposes an edit and a human approves exactly
what it saw; the network splits; both sides keep editing; the split heals; every replica converges
to the same state hash; the owner accepts one suggestion and rejects the other, and the rejected
one leaves the main text byte-identical to what it was before it existed.

No API key, no sockets, no network: the planner is a seeded fake, the transport is a pair of
queues, and everything it prints comes from the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue, SecretStr

from crdt_coedit_agents.agentic.approval import ApprovalStore
from crdt_coedit_agents.agentic.audit import AuditLog, verify_file
from crdt_coedit_agents.agentic.budget import BudgetLedger, BudgetLimits
from crdt_coedit_agents.agentic.providers.fake import FakeProvider
from crdt_coedit_agents.agents.planner import (
    AddTagStep,
    EditPlan,
    InsertAfterStep,
    SetFieldStep,
    SuggestionPlanner,
)
from crdt_coedit_agents.agents.replica import AgentReplica
from crdt_coedit_agents.agents.scripted import ScriptedPlanPolicy
from crdt_coedit_agents.crdt.document import DocumentState
from crdt_coedit_agents.crdt.ids import VersionVector
from crdt_coedit_agents.ops.apply import apply_op
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.ops.model import SuggestionDecide
from crdt_coedit_agents.ops.signing import ReplicaIdentity
from crdt_coedit_agents.replica import Replica
from crdt_coedit_agents.review import (
    approve_publish,
    decide_in_document,
    pending_reviews,
)
from crdt_coedit_agents.snapshot import load_snapshot, write_snapshot
from crdt_coedit_agents.sync.auth import Role, issue_token, verify_token
from crdt_coedit_agents.sync.client import ReplicaSession
from crdt_coedit_agents.sync.hub import DocumentHub, HubLimits
from crdt_coedit_agents.sync.inproc import InProcessChannel, channel_pair
from crdt_coedit_agents.sync.session import run_session

__all__ = ["DOC_ID", "run_demo"]

DOC_ID = "release-notes"

INITIAL_TEXT = "Release 2.4 ships faster sync and clearer errors."


@dataclass(slots=True)
class _Peer:
    """One connected replica and the server task serving it."""

    name: str
    session: ReplicaSession
    client_channel: InProcessChannel
    server_task: asyncio.Task[None]

    async def disconnect(self) -> None:
        await self.client_channel.close(1000, "demo partition")
        self.server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.server_task


def _drafter_plan(anchors: Sequence[str], instruction: str) -> EditPlan:
    """What the drafting agent proposes: a closing sentence, a status field and a tag."""
    del instruction
    last = anchors[-1] if len(anchors) > 1 else "START"
    return EditPlan(
        summary="Add an upgrade note, mark the document in review and tag it.",
        steps=(
            InsertAfterStep(anchor_id=last, text=" Upgrading is a drop-in replacement."),
            SetFieldStep(key="status", value="in-review"),
            AddTagStep(tag="agent-drafted"),
        ),
    )


def _offline_plan(anchors: Sequence[str], instruction: str) -> EditPlan:
    """What the partitioned agent proposes against its stale view."""
    del instruction
    return EditPlan(
        summary="Open the notes with a heading.",
        steps=(InsertAfterStep(anchor_id=anchors[0], text="Notes: "),),
    )


def _agent(
    name: str,
    identity: ReplicaIdentity,
    build: Any,
    workdir: Path,
    approvals: ApprovalStore,
    audit: AuditLog,
) -> tuple[AgentReplica, BudgetLedger]:
    ledger = BudgetLedger(
        BudgetLimits.of(steps=4, tokens=200_000, usd=0.25), workdir / f"budget-{name}.json"
    )
    planner = SuggestionPlanner(
        FakeProvider(ScriptedPlanPolicy(build), seed=17), budget=ledger, max_steps=8
    )
    agent = AgentReplica(
        Replica(identity, DOC_ID),
        planner,
        approvals=approvals,
        audit=audit,
        principal=f"agent:{name}",
    )
    return agent, ledger


async def _connect(
    hub: DocumentHub, secret: SecretStr, name: str, replica: Replica, role: Role
) -> _Peer:
    token = issue_token(
        secret,
        principal=f"demo:{name}",
        doc_id=DOC_ID,
        role=role,
        replica_id=replica.replica_id,
    )
    grant = verify_token(secret, token, doc_id=DOC_ID)
    client_channel, server_channel = channel_pair()
    task = asyncio.ensure_future(run_session(hub, grant, server_channel))
    session = ReplicaSession(replica, client_channel)
    await session.hello()
    return _Peer(name=name, session=session, client_channel=client_channel, server_task=task)


async def _run(workdir: Path) -> dict[str, JsonValue]:  # noqa: PLR0915 - one linear story
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in ("oplog.sqlite3", "audit.jsonl", "approvals.json", "snapshot.json"):
        (workdir / stale).unlink(missing_ok=True)
    for name in ("drafter", "offline"):
        (workdir / f"budget-{name}.json").unlink(missing_ok=True)

    secret = SecretStr(secrets.token_urlsafe(32))
    audit = AuditLog(workdir / "audit.jsonl")
    approvals = ApprovalStore(workdir / "approvals.json")
    hub = DocumentHub(
        OpLog(workdir / "oplog.sqlite3"),
        limits=HubLimits(max_ops_per_push=256),
        audit=audit,
    )

    owner = Replica(ReplicaIdentity.generate(), DOC_ID)
    drafter, drafter_budget = _agent(
        "drafter", ReplicaIdentity.generate(), _drafter_plan, workdir, approvals, audit
    )
    offline, offline_budget = _agent(
        "offline", ReplicaIdentity.generate(), _offline_plan, workdir, approvals, audit
    )

    owner_peer = await _connect(hub, secret, "owner", owner, Role.OWNER)
    owner_peer.session.queue(owner.insert_text(INITIAL_TEXT))
    owner_peer.session.queue([owner.set_field("title", "Release notes")])
    await owner_peer.session.push_pending()

    # --- a connected agent proposes, a human approves exactly what it saw -----------------
    drafter_peer = await _connect(hub, secret, "drafter", drafter.replica, Role.SUGGESTER)
    offline_peer = await _connect(hub, secret, "offline", offline.replica, Role.SUGGESTER)

    first = drafter.propose("Add a short upgrade note at the end.")
    first_review = approve_publish(
        approvals,
        audit,
        request_id=pending_reviews(approvals)[0].request_id,
        approver="human:maintainer",
        reason="the wording matches the release",
    )
    drafter_peer.session.queue(drafter.publish(first.suggestion_id))
    await drafter_peer.session.push_pending()
    await owner_peer.session.pull()
    text_before_decisions = owner.view().text

    # --- partition: the offline agent keeps working with no connection -------------------
    await offline_peer.disconnect()
    second = offline.propose("Open the notes with a heading.")
    second_review = approve_publish(
        approvals,
        audit,
        request_id=next(
            item.request_id
            for item in pending_reviews(approvals)
            if item.suggestion_id == second.suggestion_id
        ),
        approver="human:maintainer",
        reason="worth looking at, decide on the content later",
    )
    offline_published = offline.publish(second.suggestion_id)
    owner_peer.session.queue(owner.append_text(" Thanks to everyone who reported issues."))
    await owner_peer.session.push_pending()
    partition_divergence = owner.state_hash() != offline.replica.state_hash()

    # --- heal: reconnect, catch up, push what accumulated --------------------------------
    healed = await _connect(hub, secret, "offline", offline.replica, Role.SUGGESTER)
    healed.session.queue(offline_published)
    await healed.session.push_pending()
    await owner_peer.session.pull()
    await drafter_peer.session.pull()
    await healed.session.pull()

    hashes = {
        "owner": owner.state_hash(),
        "drafter": drafter.replica.state_hash(),
        "offline": offline.replica.state_hash(),
        "server": hub.state(DOC_ID).state_hash(),
    }

    # --- the owner decides what enters the document ---------------------------------------
    accept_op = decide_in_document(
        owner,
        audit,
        first_review,
        accept=True,
        approver="human:maintainer",
        reason="reads well and matches the release",
    )
    reject_op = decide_in_document(
        owner,
        audit,
        second_review,
        accept=False,
        approver="human:maintainer",
        reason="the heading is redundant",
    )
    owner_peer.session.queue([accept_op, reject_op])
    await owner_peer.session.push_pending()
    await drafter_peer.session.pull()
    await healed.session.pull()

    final_text = owner.view().text
    rejected_left_no_trace = _replay_without(owner, second.suggestion_id) == final_text

    # --- evidence -------------------------------------------------------------------------
    chain = hub.verify_log(DOC_ID)
    audit_result = verify_file(audit.path, expect_head=audit.head)
    snapshot_path = workdir / "snapshot.json"
    write_snapshot(snapshot_path, owner.state, owner.version_vector)
    restored, restored_vector = load_snapshot(snapshot_path)

    for peer in (owner_peer, drafter_peer, healed):
        await peer.disconnect()
    # Release the SQLite handle so the directory can be reused (and cleaned up on Windows).
    hub.op_log.close()

    return {
        "doc_id": DOC_ID,
        "initial_text": INITIAL_TEXT,
        "suggestions": {
            "accepted": {
                "id": first_review.suggestion_id,
                "summary": first_review.summary,
                "inserted_text": first_review.inserted_text,
                "digest": first_review.digest[:16],
            },
            "rejected": {
                "id": second_review.suggestion_id,
                "summary": second_review.summary,
                "inserted_text": second_review.inserted_text,
            },
        },
        "partition_caused_divergence": partition_divergence,
        "converged": len(set(hashes.values())) == 1,
        "state_hash": hashes["owner"],
        "replica_hashes": {name: value[:16] for name, value in sorted(hashes.items())},
        "text_before_decisions": text_before_decisions,
        "final_text": final_text,
        "rejected_suggestion_left_no_trace": rejected_left_no_trace,
        "main_view_fields": dict(owner.view().fields),
        "main_view_tags": list(owner.view().tags),
        "op_log": {
            "ok": chain.ok,
            "entries": chain.entries,
            "head": chain.head[:16],
        },
        "audit_log": {"ok": audit_result.ok, "entries": audit_result.entries},
        "snapshot_restored": restored.state_hash() == owner.state_hash(),
        "snapshot_version_vector": _as_json_vector(restored_vector),
        "agent_budgets": {
            "drafter": dict(drafter_budget.state.model_dump(mode="json")),
            "offline": dict(offline_budget.state.model_dump(mode="json")),
        },
    }


def _as_json_vector(vector: VersionVector) -> dict[str, JsonValue]:
    """A version vector as plain JSON values."""
    return dict(vector.as_dict().items())


def _replay_without(owner: Replica, suggestion_id: str) -> str:
    """Replay the whole history with one suggestion's operations removed, and materialize it.

    This is the honest form of "rejecting left no trace": it rebuilds the document from the
    operations that would have existed if the agent had never run, rather than just re-reading the
    filter that hid them.
    """
    state = DocumentState(doc_id=owner.doc_id)
    for signed in owner.ops():
        payload = signed.op.payload
        if signed.op.suggestion_id == suggestion_id:
            continue
        if isinstance(payload, SuggestionDecide) and payload.suggestion_id == suggestion_id:
            continue
        apply_op(state, signed)
    return state.view().text


def run_demo(workdir: Path) -> dict[str, JsonValue]:
    """Run the demo end to end and return its summary."""
    return asyncio.run(_run(workdir))
