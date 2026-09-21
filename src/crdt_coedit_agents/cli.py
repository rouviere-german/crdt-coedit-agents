"""Command line client: ``crdt-coedit-agents`` or ``python -m crdt_coedit_agents``.

The commands operate on a local operation log, which is the same store the sync server uses. That
keeps the review workflow usable on one host without a running server, and every write still goes
through the hub, so the role policy, the replay guard and the hash chain apply exactly as they do
over the network.

Keys never appear on the command line: ``keygen`` writes a 0600 key file, and every command that
signs reads it from ``--key``. The JWT secret comes from the environment (``APP_SYNC_JWT_SECRET``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from crdt_coedit_agents import __version__

__all__ = ["app", "main"]

app = typer.Typer(
    name="crdt-coedit-agents",
    help="Humans and agents co-editing one CRDT document.",
    no_args_is_help=True,
    add_completion=False,
)
doc_app = typer.Typer(help="Inspect and verify documents in a local operation log.")
review_app = typer.Typer(help="Review agent suggestions waiting for a human.")
app.add_typer(doc_app, name="doc")
app.add_typer(review_app, name="review")

DbOption = Annotated[Path, typer.Option("--db", help="Path to the operation log database.")]
KeyOption = Annotated[Path, typer.Option("--key", help="Path to a replica key file.")]
DocArgument = Annotated[str, typer.Argument(help="Document id.")]

_DEFAULT_DB = Path("var/oplog.sqlite3")
_DEFAULT_KEY = Path("var/replica.key")
_DEFAULT_APPROVALS = Path("var/approvals.json")
_DEFAULT_AUDIT = Path("var/audit.jsonl")


def _echo(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _settings() -> Any:
    """Read settings fresh for each command: one invocation, one process, one environment."""
    from crdt_coedit_agents.core.config import Settings  # noqa: PLC0415

    return Settings()


def _hub(db: Path) -> Any:
    from crdt_coedit_agents.ops.log import OpLog  # noqa: PLC0415
    from crdt_coedit_agents.sync.hub import DocumentHub  # noqa: PLC0415

    return DocumentHub(OpLog(db))


@app.callback()
def _root() -> None:
    """Configure logging once for every command."""
    from crdt_coedit_agents.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)


@app.command()
def version() -> None:
    """Print the package version."""
    _echo({"version": __version__})


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port.")] = None,
) -> None:
    """Run the synchronization service."""
    import uvicorn  # noqa: PLC0415

    from crdt_coedit_agents.core.logging import configure_logging  # noqa: PLC0415
    from crdt_coedit_agents.core.tracing import install_global_tracer_provider  # noqa: PLC0415
    from crdt_coedit_agents.sync.server import create_sync_app  # noqa: PLC0415

    settings = _settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    if settings.sync_jwt_secret is None:
        _fail("APP_SYNC_JWT_SECRET must be set before serving synchronization.")
    application = create_sync_app(settings)
    install_global_tracer_provider(application.state.tracer_provider)
    uvicorn.run(
        application,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )


@app.command()
def keygen(
    key: KeyOption = _DEFAULT_KEY,
    if_missing: Annotated[
        bool, typer.Option("--if-missing", help="Succeed quietly when the key already exists.")
    ] = False,
) -> None:
    """Create a replica Ed25519 key and print the replica id it derives."""
    from crdt_coedit_agents.ops.signing import ReplicaIdentity  # noqa: PLC0415

    if key.exists():
        if not if_missing:
            _fail(f"{key} already exists; refusing to overwrite a replica key.")
        existing = ReplicaIdentity.load(key)
        _echo(
            {
                "key_file": str(key),
                "replica_id": existing.replica_id,
                "public_key": existing.public_key_hex,
                "created": False,
            }
        )
        return
    identity = ReplicaIdentity.generate()
    key.parent.mkdir(parents=True, exist_ok=True)
    identity.save(key)
    _echo(
        {
            "key_file": str(key),
            "replica_id": identity.replica_id,
            "public_key": identity.public_key_hex,
            "created": True,
        }
    )


@app.command("generate-api-key")
def generate_api_key() -> None:
    """Print a new HTTP API key and the SHA-256 digest to configure."""
    from crdt_coedit_agents.core.security import generate_api_key as make_key  # noqa: PLC0415

    raw, digest = make_key()
    _echo({"api_key": raw, "api_key_sha256": digest})
    typer.echo("Store only api_key_sha256 in APP_API_KEY_SHA256.", err=True)


@app.command()
def token(
    doc_id: DocArgument,
    role: Annotated[str, typer.Option(help="owner, editor or suggester.")] = "editor",
    principal: Annotated[str, typer.Option(help="Who the token is issued to.")] = "cli",
    key: KeyOption = _DEFAULT_KEY,
) -> None:
    """Mint a synchronization token for the replica in ``--key``."""
    from crdt_coedit_agents.ops.signing import ReplicaIdentity, SigningError  # noqa: PLC0415
    from crdt_coedit_agents.sync.auth import Role, issue_token  # noqa: PLC0415

    settings = _settings()
    if settings.sync_jwt_secret is None:
        _fail("APP_SYNC_JWT_SECRET must be set to mint tokens.")
        return
    try:
        identity = ReplicaIdentity.load(key)
        grant_role = Role(role)
    except (SigningError, ValueError) as exc:
        _fail(str(exc))
        return
    minted = issue_token(
        settings.sync_jwt_secret,
        principal=principal,
        doc_id=doc_id,
        role=grant_role,
        replica_id=identity.replica_id,
        ttl_seconds=settings.sync_token_ttl_seconds,
    )
    _echo(
        {
            "doc_id": doc_id,
            "role": grant_role.value,
            "replica_id": identity.replica_id,
            "token": minted,
        }
    )


@doc_app.command("show")
def doc_show(
    doc_id: DocArgument,
    db: DbOption = _DEFAULT_DB,
    suggestion: Annotated[
        str | None, typer.Option(help="Also show this suggestion's edits.")
    ] = None,
) -> None:
    """Materialize a document from the local operation log."""
    hub = _hub(db)
    state = hub.state(doc_id)
    view = state.view(state.editing_scope(suggestion))
    _echo(
        {
            **view.as_json(),
            "anchors": len(view.anchors),
            "state_hash": state.state_hash(),
            "pending_suggestions": list(state.pending_suggestions()),
            "log_head": hub.log_head(doc_id),
        }
    )


@doc_app.command("verify")
def doc_verify(
    doc_id: DocArgument,
    db: DbOption = _DEFAULT_DB,
    expect_head: Annotated[str | None, typer.Option(help="Head hash to require.")] = None,
) -> None:
    """Re-verify the hash chain and every signature in a document's operation log."""
    result = _hub(db).verify_log(doc_id, expect_head=expect_head)
    _echo(
        {
            "ok": result.ok,
            "entries": result.entries,
            "head": result.head,
            "error": result.error,
            "chain_seq": result.chain_seq,
        }
    )
    if not result.ok:
        raise typer.Exit(code=1)


@doc_app.command("snapshot")
def doc_snapshot(
    doc_id: DocArgument,
    out: Annotated[Path, typer.Option("--out", help="Snapshot file to write.")],
    db: DbOption = _DEFAULT_DB,
) -> None:
    """Write a verified snapshot of a document."""
    from crdt_coedit_agents.snapshot import write_snapshot  # noqa: PLC0415

    hub = _hub(db)
    snapshot = write_snapshot(out, hub.state(doc_id), hub.version_vector(doc_id))
    _echo({"path": str(out), "state_hash": snapshot.state_hash, "doc_id": doc_id})


@review_app.command("list")
def review_list(
    approvals: Annotated[Path, typer.Option("--approvals")] = _DEFAULT_APPROVALS,
) -> None:
    """Show suggestions waiting for a human decision."""
    from crdt_coedit_agents.agentic.approval import ApprovalStore  # noqa: PLC0415
    from crdt_coedit_agents.review import pending_reviews  # noqa: PLC0415

    items = pending_reviews(ApprovalStore(approvals))
    _echo([item.__dict__ for item in items])


@review_app.command("decide")
def review_decide(
    request_id: Annotated[str, typer.Argument(help="Approval request id.")],
    approve: Annotated[bool, typer.Option("--approve/--deny")] = True,
    approver: Annotated[str, typer.Option(help="Who is deciding.")] = "human:cli",
    reason: Annotated[str | None, typer.Option(help="Why.")] = None,
    approvals: Annotated[Path, typer.Option("--approvals")] = _DEFAULT_APPROVALS,
    audit: Annotated[Path, typer.Option("--audit")] = _DEFAULT_AUDIT,
) -> None:
    """Gate 1: let an agent publish a draft, or refuse it."""
    from crdt_coedit_agents.agentic.approval import ApprovalStore  # noqa: PLC0415
    from crdt_coedit_agents.agentic.audit import AuditLog  # noqa: PLC0415
    from crdt_coedit_agents.agentic.errors import ApprovalError  # noqa: PLC0415
    from crdt_coedit_agents.review import approve_publish, deny_publish  # noqa: PLC0415

    store = ApprovalStore(approvals)
    log = AuditLog(audit)
    action = approve_publish if approve else deny_publish
    try:
        item = action(store, log, request_id=request_id, approver=approver, reason=reason)
    except (ApprovalError, KeyError, ValueError) as exc:
        _fail(str(exc))
        return
    _echo({"decision": "approved" if approve else "denied", **item.__dict__})


@app.command()
def accept(
    doc_id: DocArgument,
    request_id: Annotated[str, typer.Argument(help="Approval request id.")],
    reject: Annotated[bool, typer.Option("--reject", help="Reject instead of accepting.")] = False,
    key: KeyOption = _DEFAULT_KEY,
    db: DbOption = _DEFAULT_DB,
    approvals: Annotated[Path, typer.Option("--approvals")] = _DEFAULT_APPROVALS,
    audit: Annotated[Path, typer.Option("--audit")] = _DEFAULT_AUDIT,
    approver: Annotated[str, typer.Option(help="Who is deciding.")] = "human:cli",
    reason: Annotated[str | None, typer.Option(help="Why.")] = None,
) -> None:
    """Gate 2: write the accept or reject flag for a reviewed suggestion."""
    from crdt_coedit_agents.agentic.approval import ApprovalStore  # noqa: PLC0415
    from crdt_coedit_agents.agentic.audit import AuditLog  # noqa: PLC0415
    from crdt_coedit_agents.agentic.errors import ApprovalError  # noqa: PLC0415
    from crdt_coedit_agents.ops.signing import ReplicaIdentity, SigningError  # noqa: PLC0415
    from crdt_coedit_agents.replica import Replica  # noqa: PLC0415
    from crdt_coedit_agents.review import (  # noqa: PLC0415
        DigestMismatchError,
        ReviewItem,
        decide_in_document,
    )
    from crdt_coedit_agents.sync.auth import Grant, Role  # noqa: PLC0415

    try:
        identity = ReplicaIdentity.load(key)
    except SigningError as exc:
        _fail(str(exc))
        return
    hub = _hub(db)
    owner = Replica(identity, doc_id, state=hub.state(doc_id).copy())
    owner.receive(list(hub.op_log.since(doc_id, owner.version_vector)))
    store = ApprovalStore(approvals)
    log = AuditLog(audit)
    try:
        item = ReviewItem.from_record(store.get(request_id))
        operation = decide_in_document(
            owner,
            log,
            item,
            accept=not reject,
            approver=approver,
            reason=reason,
        )
    except (ApprovalError, KeyError, ValueError, DigestMismatchError) as exc:
        _fail(str(exc))
        return
    grant = Grant(
        principal=approver, doc_id=doc_id, role=Role.OWNER, replica_id=identity.replica_id
    )
    result = hub.submit(grant, [operation])
    _echo(
        {
            "suggestion_id": item.suggestion_id,
            "decision": "rejected" if reject else "accepted",
            "accepted_ops": result.accepted,
            "rejected": result.rejected,
            "log_head": result.log_head,
            "text": hub.state(doc_id).view().text,
        }
    )
    if result.rejected:
        raise typer.Exit(code=1)


@app.command("audit-verify")
def audit_verify(
    path: Annotated[Path, typer.Argument(help="Audit log file.")] = _DEFAULT_AUDIT,
    expect_head: Annotated[str | None, typer.Option(help="Head hash to require.")] = None,
) -> None:
    """Verify the hash-chained review audit log."""
    from crdt_coedit_agents.agentic.audit import verify_file  # noqa: PLC0415

    result = verify_file(path, expect_head=expect_head)
    _echo(
        {
            "ok": result.ok,
            "entries": result.entries,
            "head": result.head,
            "error": result.error,
            "line": result.line,
        }
    )
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def agent(
    doc_id: DocArgument,
    instruction: Annotated[str, typer.Option(help="What the agent should propose.")],
    note: Annotated[
        str, typer.Option(help="Offline fallback text when no API key is configured.")
    ] = "Reviewed by the drafting agent.",
    url: Annotated[
        str, typer.Option(help="Base URL of the sync service.")
    ] = "http://127.0.0.1:8000",
    key: KeyOption = _DEFAULT_KEY,
    token_value: Annotated[
        str | None, typer.Option("--token", help="Sync token; minted from the secret if omitted.")
    ] = None,
    approvals: Annotated[Path, typer.Option("--approvals")] = _DEFAULT_APPROVALS,
    audit: Annotated[Path, typer.Option("--audit")] = _DEFAULT_AUDIT,
    budget: Annotated[Path, typer.Option("--budget")] = Path("var/agent-budget.json"),
    wait_seconds: Annotated[
        int, typer.Option(help="Seconds to wait for a human decision before giving up.")
    ] = 0,
) -> None:
    """Connect as a suggest-only replica, propose one edit, and wait for a human decision.

    The agent never publishes on its own: it opens an approval request and, with ``--wait-seconds``,
    polls until a person decides with ``review decide``. Without an ``ANTHROPIC_API_KEY`` it uses a
    scripted offline planner, so the command works with no network and no credentials.
    """
    import asyncio  # noqa: PLC0415

    from crdt_coedit_agents.agents.runner import AgentRunConfig, run_agent  # noqa: PLC0415
    from crdt_coedit_agents.ops.signing import ReplicaIdentity, SigningError  # noqa: PLC0415
    from crdt_coedit_agents.sync.auth import Role, issue_token  # noqa: PLC0415

    settings = _settings()
    try:
        identity = ReplicaIdentity.load(key)
    except SigningError as exc:
        _fail(str(exc))
        return
    minted = token_value
    if minted is None:
        if settings.sync_jwt_secret is None:
            _fail("Pass --token or set APP_SYNC_JWT_SECRET so the agent can mint one.")
            return
        minted = issue_token(
            settings.sync_jwt_secret,
            principal=f"agent:{identity.replica_id[:12]}",
            doc_id=doc_id,
            role=Role.SUGGESTER,
            replica_id=identity.replica_id,
            ttl_seconds=settings.sync_token_ttl_seconds,
        )
    config = AgentRunConfig(
        doc_id=doc_id,
        instruction=instruction,
        note=note,
        base_url=url,
        token=minted,
        approvals=approvals,
        audit=audit,
        budget=budget,
        wait_seconds=wait_seconds,
    )
    _echo(asyncio.run(run_agent(identity, config, settings)))


@app.command()
def demo(
    workdir: Annotated[Path, typer.Option(help="Where the demo writes its files.")] = Path(
        "var/demo"
    ),
) -> None:
    """Run the offline end-to-end demo."""
    from crdt_coedit_agents.demo import run_demo  # noqa: PLC0415

    _echo(run_demo(workdir))


def main() -> int:
    """Console entry point. Typer exits the process itself."""
    app()
    return 0
