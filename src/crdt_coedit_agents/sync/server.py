"""The FastAPI sync service: a WebSocket endpoint plus read-only document views.

The HTTP surface reuses the hardened application factory (request ids, security headers, rate
limiting, body limits, non-leaking errors, `/healthz`, `/readyz`, `/metrics`). The WebSocket route
adds its own checks, because the HTTP middlewares deliberately pass non-HTTP scopes through:

* the token arrives in the ``Authorization: Bearer`` header, never in the URL, so it cannot end up
  in access logs or browser history;
* the connection is rejected *before* it is accepted when the token is missing, expired or issued
  for another document;
* the token pins the connection to one replica id, which every submitted operation must match;
* a per-connection token bucket limits frames, and the subscriber queue bounds outbound buffering.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Path as PathParam
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, JsonValue

from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.core.app import create_app
from crdt_coedit_agents.core.config import Settings, get_settings
from crdt_coedit_agents.core.logging import get_logger
from crdt_coedit_agents.core.security import AuthenticatedPrincipal
from crdt_coedit_agents.metrics import create_sync_metrics
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.ops.model import DOC_ID_PATTERN
from crdt_coedit_agents.sync.auth import TokenError, verify_token
from crdt_coedit_agents.sync.hub import DocumentHub, HubLimits
from crdt_coedit_agents.sync.session import (
    CLOSE_POLICY,
    ChannelClosedError,
    SessionLimits,
    run_session,
)

__all__ = ["DocumentResponse", "WebSocketChannel", "build_hub", "create_sync_app"]

_log = get_logger("crdt_coedit_agents.server")
_WS_UNAUTHORIZED = 4401

# Module level: with postponed annotation evaluation, FastAPI resolves route annotations against
# module globals, so a path-parameter alias defined inside the factory would not be found.
DocId = Annotated[str, PathParam(pattern=DOC_ID_PATTERN, max_length=64)]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    scope: str
    text: str
    fields: dict[str, JsonValue]
    tags: list[str]
    state_hash: str
    log_head: str
    pending_suggestions: list[str]


class WebSocketChannel:
    """Adapts a Starlette ``WebSocket`` to the transport-independent channel protocol."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send(self, text: str) -> None:
        try:
            await self._websocket.send_text(text)
        except (WebSocketDisconnect, RuntimeError) as exc:
            raise ChannelClosedError("websocket closed") from exc

    async def receive(self) -> str:
        try:
            return await self._websocket.receive_text()
        except (WebSocketDisconnect, RuntimeError, KeyError) as exc:
            raise ChannelClosedError("websocket closed") from exc

    async def close(self, code: int = 1000, reason: str = "") -> None:
        # Closing an already-closed socket raises; the connection is gone either way.
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await self._websocket.close(code=code, reason=reason[:120])


def build_hub(settings: Settings, *, registry: CollectorRegistry | None = None) -> DocumentHub:
    """Open the operation log and the audit log, and wire them into a hub."""
    db_path = settings.sync_db_path
    audit_path = settings.sync_audit_path
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    limits = HubLimits(
        max_ops_per_push=settings.sync_max_ops_per_push,
        max_document_bytes=settings.sync_max_document_bytes,
        max_catch_up_ops=settings.sync_max_catch_up_ops,
        subscriber_queue=settings.sync_subscriber_queue,
        max_subscribers_per_doc=settings.sync_max_subscribers_per_doc,
    )
    return DocumentHub(
        OpLog(db_path),
        limits=limits,
        metrics=create_sync_metrics(registry),
        audit=AuditLog(Path(audit_path)),
    )


def create_sync_app(
    settings: Settings | None = None,
    *,
    hub: DocumentHub | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the sync service. Pass ``hub`` in tests to share an in-memory log."""
    cfg = settings if settings is not None else get_settings()
    document_hub = hub if hub is not None else build_hub(cfg, registry=registry)
    session_limits = SessionLimits(
        messages_per_second=cfg.sync_messages_per_second,
        message_burst=cfg.sync_message_burst,
    )

    def _log_reachable() -> bool:
        try:
            document_hub.op_log.documents()
        except Exception:  # noqa: BLE001 - readiness must report, not crash
            return False
        return True

    app = create_app(
        cfg,
        readiness_checks={"op_log": _log_reachable},
        registry=document_hub.metrics.registry,
    )
    app.state.hub = document_hub
    app.state.session_limits = session_limits

    @app.get("/v1/docs/{doc_id}", response_model=DocumentResponse)
    async def read_document(doc_id: DocId, principal: AuthenticatedPrincipal) -> DocumentResponse:
        """The main view of a document. Suggested edits are not part of it."""
        state = document_hub.state(doc_id)
        view = state.view()
        _log.info("document_read", doc_id=doc_id, key_id=principal.key_id, chars=len(view.text))
        return DocumentResponse(
            doc_id=doc_id,
            scope=view.scope,
            text=view.text,
            fields=view.fields,
            tags=view.tags,
            state_hash=state.state_hash(),
            log_head=document_hub.log_head(doc_id),
            pending_suggestions=list(state.pending_suggestions()),
        )

    @app.get("/v1/docs/{doc_id}/verify")
    async def verify_document_log(
        doc_id: DocId, principal: AuthenticatedPrincipal
    ) -> dict[str, Any]:
        """Re-verify the hash chain and every signature in the document's operation log."""
        del principal
        result = document_hub.verify_log(doc_id)
        return {
            "ok": result.ok,
            "entries": result.entries,
            "head": result.head,
            "error": result.error,
            "chain_seq": result.chain_seq,
        }

    @app.websocket("/v1/docs/{doc_id}/sync")
    async def sync(websocket: WebSocket, doc_id: str) -> None:
        secret = cfg.sync_jwt_secret
        header = websocket.headers.get("authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        if secret is None:
            await websocket.close(code=_WS_UNAUTHORIZED, reason="sync is not configured")
            return
        try:
            grant = verify_token(secret, token, doc_id=doc_id)
        except TokenError as exc:
            _log.info("ws_token_rejected", doc_id=doc_id, reason=str(exc))
            await websocket.close(code=_WS_UNAUTHORIZED, reason="invalid token")
            return
        await websocket.accept()
        channel = WebSocketChannel(websocket)
        try:
            await run_session(document_hub, grant, channel, limits=session_limits)
        except ChannelClosedError:
            pass
        finally:
            await channel.close(CLOSE_POLICY, "session ended")

    return app
