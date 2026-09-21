"""The real WebSocket route: token handling, the document API and an end-to-end edit."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from crdt_coedit_agents.core.config import Settings
from crdt_coedit_agents.core.security import hash_api_key
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.replica import Replica, new_suggestion_id
from crdt_coedit_agents.sync.auth import Role, issue_token
from crdt_coedit_agents.sync.hub import DocumentHub
from crdt_coedit_agents.sync.protocol import Hello, Push, encode
from crdt_coedit_agents.sync.server import create_sync_app
from tests.conftest import make_replica

pytestmark = pytest.mark.integration

DOC = "test-doc"
API_KEY = "websocket-test-key-" + "x" * 24
AUTH = {"X-API-Key": API_KEY}
SECRET = SecretStr("a-long-websocket-test-signing-secret")


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "api_key_sha256": [hash_api_key(API_KEY)],
        "environment": "test",
        "sync_jwt_secret": SECRET,
        "rate_limit_burst": 200,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def hub() -> DocumentHub:
    return DocumentHub(OpLog())


@pytest.fixture
def client(hub: DocumentHub) -> Iterator[TestClient]:
    app = create_sync_app(settings(), hub=hub)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def token(replica: Replica, role: Role = Role.OWNER) -> str:
    return issue_token(
        SECRET,
        principal=f"user:{role.value}",
        doc_id=DOC,
        role=role,
        replica_id=replica.replica_id,
    )


def bearer(replica: Replica, role: Role = Role.OWNER) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(replica, role)}"}


def test_a_websocket_session_syncs_a_document_end_to_end(
    client: TestClient, hub: DocumentHub
) -> None:
    owner = make_replica(1)
    with client.websocket_connect(f"/v1/docs/{DOC}/sync", headers=bearer(owner)) as socket:
        socket.send_text(encode(Hello(doc_id=DOC)))
        welcome = json.loads(socket.receive_text())
        assert welcome["type"] == "welcome"
        assert welcome["role"] == "owner"
        assert welcome["replica_id"] == owner.replica_id

        catch_up = json.loads(socket.receive_text())
        assert catch_up == {"type": "delta", "ops": [], "reason": "catch_up"}

        socket.send_text(encode(Push(ops=tuple(owner.insert_text("hello")))))
        ack = json.loads(socket.receive_text())
        assert ack["type"] == "ack"
        assert ack["accepted"] == 5
        assert ack["rejected"] == {}

    assert hub.state(DOC).view().text == "hello"


def test_a_second_client_catches_up_and_receives_relayed_edits(
    client: TestClient, hub: DocumentHub
) -> None:
    owner, editor = make_replica(1), make_replica(2, start_ms=1_700_000_500_000)
    with client.websocket_connect(f"/v1/docs/{DOC}/sync", headers=bearer(owner)) as first:
        first.send_text(encode(Hello(doc_id=DOC)))
        first.receive_text()
        first.receive_text()
        first.send_text(encode(Push(ops=tuple(owner.insert_text("shared")))))
        first.receive_text()

        with client.websocket_connect(
            f"/v1/docs/{DOC}/sync", headers=bearer(editor, Role.EDITOR)
        ) as second:
            second.send_text(encode(Hello(doc_id=DOC)))
            assert json.loads(second.receive_text())["role"] == "editor"
            catch_up = json.loads(second.receive_text())
            assert len(catch_up["ops"]) == 6
            editor.receive(hub.op_log.since(DOC, editor.version_vector))
            assert editor.view().text == "shared"

            first.send_text(encode(Push(ops=tuple(owner.append_text("!")))))
            first.receive_text()
            relayed = json.loads(second.receive_text())
            assert relayed["reason"] == "relay"
            assert len(relayed["ops"]) == 1


def test_a_connection_without_a_usable_token_is_closed_before_it_is_accepted(
    client: TestClient,
) -> None:
    owner = make_replica(1)
    for headers in (
        {},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": token(owner)},  # no Bearer prefix
    ):
        with (
            pytest.raises(WebSocketDisconnect) as info,
            client.websocket_connect(f"/v1/docs/{DOC}/sync", headers=headers) as socket,
        ):
            socket.receive_text()
        assert info.value.code == 4401


def test_a_token_for_another_document_cannot_open_this_one(client: TestClient) -> None:
    owner = make_replica(1)
    other = issue_token(
        SECRET,
        principal="user:owner",
        doc_id="other-doc",
        role=Role.OWNER,
        replica_id=owner.replica_id,
    )
    with (
        pytest.raises(WebSocketDisconnect) as info,
        client.websocket_connect(
            f"/v1/docs/{DOC}/sync", headers={"Authorization": f"Bearer {other}"}
        ) as socket,
    ):
        socket.receive_text()
    assert info.value.code == 4401


def test_synchronization_is_refused_when_no_secret_is_configured(hub: DocumentHub) -> None:
    app = create_sync_app(settings(sync_jwt_secret=None), hub=hub)
    with TestClient(app) as unconfigured:
        owner = make_replica(1)
        with (
            pytest.raises(WebSocketDisconnect) as info,
            unconfigured.websocket_connect(f"/v1/docs/{DOC}/sync", headers=bearer(owner)) as socket,
        ):
            socket.receive_text()
        assert info.value.code == 4401


def test_the_document_endpoint_shows_the_main_view_and_requires_a_key(
    client: TestClient, hub: DocumentHub
) -> None:
    owner, agent = make_replica(1), make_replica(2, start_ms=1_700_000_500_000)
    with client.websocket_connect(f"/v1/docs/{DOC}/sync", headers=bearer(owner)) as socket:
        socket.send_text(encode(Hello(doc_id=DOC)))
        socket.receive_text()
        socket.receive_text()
        socket.send_text(encode(Push(ops=tuple(owner.insert_text("hi")))))
        socket.receive_text()

    agent.receive(hub.op_log.since(DOC, agent.version_vector))
    suggestion_id = new_suggestion_id()
    with client.websocket_connect(
        f"/v1/docs/{DOC}/sync", headers=bearer(agent, Role.SUGGESTER)
    ) as socket:
        socket.send_text(encode(Hello(doc_id=DOC)))
        socket.receive_text()
        socket.receive_text()
        ops = agent.insert_text("!", after=agent.view().anchors[-1], suggestion_id=suggestion_id)
        socket.send_text(encode(Push(ops=tuple(ops))))
        assert json.loads(socket.receive_text())["accepted"] == 1

    assert client.get(f"/v1/docs/{DOC}").status_code == 401
    body = client.get(f"/v1/docs/{DOC}", headers=AUTH).json()
    assert body["text"] == "hi"
    assert body["pending_suggestions"] == [suggestion_id]
    assert body["state_hash"] == hub.state(DOC).state_hash()

    verify = client.get(f"/v1/docs/{DOC}/verify", headers=AUTH).json()
    assert verify["ok"] is True
    assert verify["entries"] == 3


def test_document_ids_are_validated_on_the_http_surface(client: TestClient) -> None:
    assert client.get("/v1/docs/NOT_A_DOC", headers=AUTH).status_code == 422
    assert client.get(f"/v1/docs/{'x' * 80}", headers=AUTH).status_code == 422


def test_readiness_reports_the_operation_log(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body == {"status": "ready", "checks": {"op_log": "ok"}}


def test_metrics_are_exposed_and_require_a_key(client: TestClient, hub: DocumentHub) -> None:
    owner = make_replica(1)
    with client.websocket_connect(f"/v1/docs/{DOC}/sync", headers=bearer(owner)) as socket:
        socket.send_text(encode(Hello(doc_id=DOC)))
        socket.receive_text()
        socket.receive_text()
        socket.send_text(encode(Push(ops=tuple(owner.insert_text("m")))))
        socket.receive_text()

    assert client.get("/metrics").status_code == 401
    body = client.get("/metrics", headers=AUTH).text
    assert 'crdt_ops_applied_total{kind="text_insert",scope="main"}' in body
    assert "crdt_merge_latency_seconds_bucket" in body
    assert "crdt_ws_connections" in body
