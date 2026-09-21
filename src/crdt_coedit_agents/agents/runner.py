"""Running one agent against a live server: connect, propose, wait for a human, publish.

The agent is a suggest-only replica. It catches up over the sync protocol, asks its planner for one
edit plan, applies it under a fresh suggestion id and opens an approval request — then stops. With
``wait_seconds`` it polls the shared approval store until a person decides with ``review decide``;
it publishes only if the answer is yes, and only while the suggestion still hashes to what that
person looked at.

There is deliberately no automatic approver. The waiting loop reads a decision someone else made;
it cannot make one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from crdt_coedit_agents.agentic.approval import ApprovalStatus, ApprovalStore
from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.agentic.budget import BudgetLedger, BudgetLimits
from crdt_coedit_agents.agentic.errors import ApprovalRequiredError
from crdt_coedit_agents.agentic.providers.base import LLMProvider
from crdt_coedit_agents.agentic.providers.fake import FakeProvider
from crdt_coedit_agents.agents.planner import (
    AddTagStep,
    EditPlan,
    InsertAfterStep,
    SuggestionPlanner,
)
from crdt_coedit_agents.agents.replica import AgentReplica, SuggestionResult
from crdt_coedit_agents.agents.scripted import ScriptedPlanPolicy
from crdt_coedit_agents.core.config import Settings
from crdt_coedit_agents.ops.signing import ReplicaIdentity
from crdt_coedit_agents.replica import Replica
from crdt_coedit_agents.sync.client import ReplicaSession
from crdt_coedit_agents.sync.session import Channel

__all__ = ["AgentRunConfig", "build_agent", "choose_provider", "run_agent"]

POLL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    doc_id: str
    instruction: str
    note: str
    base_url: str
    token: str
    approvals: Path
    audit: Path
    budget: Path
    wait_seconds: int = 0


def offline_plan(note: str) -> object:
    """A scripted plan for running with no API key: one sentence at the end, plus a tag."""

    def build(anchors: Sequence[str], instruction: str) -> EditPlan:
        del instruction
        anchor = anchors[-1] if len(anchors) > 1 else "START"
        return EditPlan(
            summary=note[:200],
            steps=(
                InsertAfterStep(anchor_id=anchor, text=f" {note}"),
                AddTagStep(tag="agent-drafted"),
            ),
        )

    return ScriptedPlanPolicy(build)


def choose_provider(settings: Settings, note: str) -> tuple[LLMProvider, str]:
    """The live provider when a key is configured, the scripted one otherwise."""
    if settings.anthropic_api_key is not None and settings.anthropic_api_key.get_secret_value():
        from crdt_coedit_agents.agentic.providers.anthropic import (  # noqa: PLC0415
            AnthropicProvider,
        )

        return AnthropicProvider.from_settings(settings), "anthropic"
    return FakeProvider(offline_plan(note), seed=11), "fake"  # type: ignore[arg-type]


def build_agent(
    identity: ReplicaIdentity, config: AgentRunConfig, settings: Settings
) -> tuple[AgentReplica, str]:
    provider, provider_name = choose_provider(settings, config.note)
    ledger = BudgetLedger(
        BudgetLimits.of(
            steps=settings.budget_max_steps,
            tokens=settings.budget_max_tokens,
            usd=settings.budget_max_usd,
        ),
        config.budget,
    )
    planner = SuggestionPlanner(provider, budget=ledger, max_steps=settings.agent_max_plan_steps)
    replica = Replica(identity, config.doc_id)
    agent = AgentReplica(
        replica,
        planner,
        approvals=ApprovalStore(config.approvals),
        audit=AuditLog(config.audit),
        principal=f"agent:{identity.replica_id[:12]}",
    )
    return agent, provider_name


async def _await_decision(store: ApprovalStore, request_id: str, wait_seconds: int) -> str:
    """Poll until someone decides, or the deadline passes. Never decides anything itself."""
    deadline = time.monotonic() + wait_seconds
    while True:
        status = store.get(request_id).status
        if status is not ApprovalStatus.PENDING:
            return status.value
        if time.monotonic() >= deadline:
            return "timeout"
        await asyncio.sleep(POLL_SECONDS)


async def run_agent(
    identity: ReplicaIdentity,
    config: AgentRunConfig,
    settings: Settings,
    *,
    channel: Channel | None = None,
) -> dict[str, JsonValue]:
    """Run one agent to completion. ``channel`` lets tests drive it without a socket."""
    agent, provider_name = build_agent(identity, config, settings)
    if channel is not None:
        return await _run_on_channel(agent, provider_name, config, channel)

    from crdt_coedit_agents.sync.ws import open_channel  # noqa: PLC0415

    async with open_channel(config.base_url, config.doc_id, config.token) as live:
        return await _run_on_channel(agent, provider_name, config, live)


async def _run_on_channel(
    agent: AgentReplica, provider_name: str, config: AgentRunConfig, channel: Channel
) -> dict[str, JsonValue]:
    session = ReplicaSession(agent.replica, channel)
    welcome = await session.hello()
    result: SuggestionResult = agent.propose(config.instruction)
    summary: dict[str, JsonValue] = {
        "doc_id": config.doc_id,
        "provider": provider_name,
        "role": welcome.role,
        "replica_id": agent.replica.replica_id,
        **result.as_json(),
        "published": False,
        "decision": "pending",
    }
    if config.wait_seconds <= 0:
        return summary

    decision = await _await_decision(
        agent.approvals, result.approval_request_id, config.wait_seconds
    )
    summary["decision"] = decision
    if decision != ApprovalStatus.APPROVED.value:
        return summary
    try:
        session.queue(agent.publish(result.suggestion_id))
    except ApprovalRequiredError:
        summary["decision"] = "superseded"
        return summary
    ack = await session.push_pending()
    summary["published"] = True
    summary["accepted_ops"] = ack.accepted
    summary["rejected_ops"] = dict(ack.rejected)
    return summary
