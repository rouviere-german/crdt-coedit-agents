"""The agent process wiring end to end: catch up, propose, wait for a person, publish.

``run_agent`` is driven over an in-process channel into a real :class:`DocumentHub` running the
real session state machine with a ``suggester`` grant, so everything except the socket is the code
that runs in production. The human side of the review is the same ``approve_publish`` /
``deny_publish`` the CLI calls.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, SecretStr

from crdt_coedit_agents.agentic.approval import ApprovalStore
from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.agents import runner
from crdt_coedit_agents.agents.runner import AgentRunConfig, choose_provider, run_agent
from crdt_coedit_agents.core.config import Settings
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.review import approve_publish, deny_publish, pending_reviews
from crdt_coedit_agents.sync import ws
from crdt_coedit_agents.sync.auth import Grant, Role
from crdt_coedit_agents.sync.hub import DocumentHub
from crdt_coedit_agents.sync.inproc import InProcessChannel, channel_pair
from crdt_coedit_agents.sync.session import run_session
from tests.conftest import make_replica, seeded_identity

pytestmark = pytest.mark.integration

DOC = "test-doc"
BASE = "Release notes."
NOTE = "Upgrading is a drop-in replacement."
AGENT = seeded_identity(7)


def seeded_hub() -> DocumentHub:
    hub = DocumentHub(OpLog())
    owner = make_replica(1, DOC)
    grant = Grant(principal="user:owner", doc_id=DOC, role=Role.OWNER, replica_id=owner.replica_id)
    assert hub.submit(grant, owner.insert_text(BASE)).accepted == len(BASE)
    return hub


def config(tmp_path: Path, *, wait_seconds: int = 0) -> AgentRunConfig:
    return AgentRunConfig(
        doc_id=DOC,
        instruction="Add an upgrade note at the end.",
        note=NOTE,
        base_url="http://127.0.0.1:8000",
        token="unused-over-an-in-process-channel",
        approvals=tmp_path / "approvals.json",
        audit=tmp_path / "audit.jsonl",
        budget=tmp_path / "budget.json",
        wait_seconds=wait_seconds,
    )


def offline_settings() -> Settings:
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    return settings


def suggester_session(hub: DocumentHub) -> tuple[InProcessChannel, asyncio.Task[None]]:
    grant = Grant(
        principal="agent:runner", doc_id=DOC, role=Role.SUGGESTER, replica_id=AGENT.replica_id
    )
    client, server = channel_pair()
    return client, asyncio.ensure_future(run_session(hub, grant, server))


async def finish(channel: InProcessChannel, task: asyncio.Task[None]) -> None:
    await channel.close()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def decide_when_requested(tmp_path: Path, *, approve: bool) -> str:
    """Play the human: wait for the agent's request, then decide it through the review API."""
    store = ApprovalStore(tmp_path / "approvals.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    for _ in range(500):
        reviews = pending_reviews(store)
        if reviews:
            decide = approve_publish if approve else deny_publish
            decide(
                store,
                audit,
                request_id=reviews[0].request_id,
                approver="human:maintainer",
                reason="checked the wording",
            )
            return reviews[0].suggestion_id
        await asyncio.sleep(0.01)
    raise AssertionError("the agent never opened a review request")


def test_without_a_key_the_agent_proposes_offline_and_stops_at_the_review(
    tmp_path: Path,
) -> None:
    hub = seeded_hub()

    async def scenario() -> dict[str, JsonValue]:
        channel, task = suggester_session(hub)
        try:
            return await run_agent(AGENT, config(tmp_path), offline_settings(), channel=channel)
        finally:
            await finish(channel, task)

    summary = asyncio.run(scenario())

    assert summary["provider"] == "fake"
    assert summary["role"] == "suggester"
    assert summary["replica_id"] == AGENT.replica_id
    assert summary["decision"] == "pending"
    assert summary["published"] is False
    assert summary["ops"] == len(" " + NOTE) + 1  # the sentence, one element per character, a tag
    reviews = pending_reviews(ApprovalStore(tmp_path / "approvals.json"))
    assert [item.suggestion_id for item in reviews] == [summary["suggestion_id"]]
    assert reviews[0].inserted_text == " " + NOTE
    assert hub.state(DOC).view().text == BASE  # nothing reached the server


def test_an_approved_suggestion_is_published_but_stays_invisible_until_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    hub = seeded_hub()

    async def scenario() -> tuple[dict[str, JsonValue], str]:
        channel, task = suggester_session(hub)
        try:
            agent_run = asyncio.ensure_future(
                run_agent(
                    AGENT, config(tmp_path, wait_seconds=30), offline_settings(), channel=channel
                )
            )
            decided = await decide_when_requested(tmp_path, approve=True)
            return await agent_run, decided
        finally:
            await finish(channel, task)

    summary, decided = asyncio.run(scenario())

    assert summary["suggestion_id"] == decided
    assert summary["decision"] == "approved"
    assert summary["published"] is True
    assert summary["accepted_ops"] == summary["ops"]
    assert summary["rejected_ops"] == {}
    server = hub.state(DOC)
    assert server.pending_suggestions() == (decided,)
    assert server.view().text == BASE  # published is not accepted
    assert server.view(server.editing_scope(decided)).text == BASE + " " + NOTE


def test_a_denied_suggestion_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    hub = seeded_hub()

    async def scenario() -> dict[str, JsonValue]:
        channel, task = suggester_session(hub)
        try:
            agent_run = asyncio.ensure_future(
                run_agent(
                    AGENT, config(tmp_path, wait_seconds=30), offline_settings(), channel=channel
                )
            )
            await decide_when_requested(tmp_path, approve=False)
            return await agent_run
        finally:
            await finish(channel, task)

    summary = asyncio.run(scenario())

    assert summary["decision"] == "denied"
    assert summary["published"] is False
    assert hub.state(DOC).known_suggestions() == frozenset()


def test_waiting_for_a_decision_gives_up_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    hub = seeded_hub()

    async def scenario() -> dict[str, JsonValue]:
        channel, task = suggester_session(hub)
        try:
            return await run_agent(
                AGENT, config(tmp_path, wait_seconds=1), offline_settings(), channel=channel
            )
        finally:
            await finish(channel, task)

    summary = asyncio.run(scenario())

    assert summary["decision"] == "timeout"
    assert summary["published"] is False
    assert len(pending_reviews(ApprovalStore(tmp_path / "approvals.json"))) == 1


def test_without_a_channel_the_agent_connects_to_the_configured_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live path opens exactly one authenticated channel, to the URL in the config."""
    hub = seeded_hub()
    opened: list[tuple[str, str, str]] = []

    @contextlib.asynccontextmanager
    async def fake_open_channel(
        base_url: str, doc_id: str, token: str, **_: Any
    ) -> AsyncIterator[InProcessChannel]:
        opened.append((base_url, doc_id, token))
        channel, task = suggester_session(hub)
        try:
            yield channel
        finally:
            await finish(channel, task)

    monkeypatch.setattr(ws, "open_channel", fake_open_channel)
    run_config = config(tmp_path)
    summary = asyncio.run(run_agent(AGENT, run_config, offline_settings()))

    assert opened == [(run_config.base_url, DOC, run_config.token)]
    assert summary["role"] == "suggester"
    assert summary["decision"] == "pending"


def test_a_configured_key_selects_the_live_provider_without_calling_it() -> None:
    settings = Settings(_env_file=None, anthropic_api_key=SecretStr("sk-test-not-a-real-key"))
    provider, name = choose_provider(settings, NOTE)
    assert name == "anthropic"
    assert provider.name == "anthropic"
    provider, name = choose_provider(
        Settings(_env_file=None, anthropic_api_key=SecretStr("")), NOTE
    )
    assert name == "fake"
