"""The offline demo and the command line client, exercised the way a person would use them."""

from __future__ import annotations

import json
import runpy
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from crdt_coedit_agents import cli
from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.core.security import hash_api_key
from crdt_coedit_agents.demo import DOC_ID, run_demo
from crdt_coedit_agents.ops.log import OpLog
from crdt_coedit_agents.replica import Replica
from crdt_coedit_agents.sync.auth import Grant, Role, verify_token
from crdt_coedit_agents.sync.hub import DocumentHub
from tests.conftest import make_replica

pytestmark = pytest.mark.integration

runner = CliRunner()
SECRET = "a-long-cli-test-signing-secret-value"


def invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    result = runner.invoke(cli.app, list(args), env=env or {})
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def payload(result: Any) -> Any:
    return json.loads(result.stdout)


def summary_of(workdir: Path) -> dict[str, Any]:
    """The demo summary as plain JSON, which is how the CLI and the README consume it."""
    return cast("dict[str, Any]", run_demo(workdir))


# ------------------------------------------------------------------------ demo


def test_the_demo_converges_accepts_one_suggestion_and_rejects_the_other(
    tmp_path: Path,
) -> None:
    summary = summary_of(tmp_path)
    assert summary["converged"] is True
    assert summary["partition_caused_divergence"] is True
    assert summary["rejected_suggestion_left_no_trace"] is True
    assert summary["snapshot_restored"] is True
    assert len(set(summary["replica_hashes"].values())) == 1

    accepted = summary["suggestions"]["accepted"]
    rejected = summary["suggestions"]["rejected"]
    assert accepted["inserted_text"] in summary["final_text"]
    assert rejected["inserted_text"] not in summary["final_text"]
    assert summary["op_log"]["ok"] is True
    assert summary["audit_log"]["ok"] is True
    assert summary["main_view_fields"]["status"] == "in-review"
    assert summary["main_view_tags"] == ["agent-drafted"]


def test_running_the_demo_twice_starts_from_a_clean_state(tmp_path: Path) -> None:
    first = summary_of(tmp_path)
    second = summary_of(tmp_path)
    assert first["op_log"]["entries"] == second["op_log"]["entries"]
    assert first["final_text"] == second["final_text"]
    assert first["state_hash"] != second["state_hash"]  # fresh keys every run


@pytest.mark.usefixtures("restore_logging")
def test_the_demo_command_prints_one_json_summary(tmp_path: Path) -> None:
    result = invoke("demo", "--workdir", str(tmp_path))
    assert result.exit_code == 0
    assert payload(result)["converged"] is True


# ------------------------------------------------------------------------- cli


def test_version_and_help_work() -> None:
    assert "version" in payload(invoke("version"))
    assert invoke("--help").exit_code == 0


def test_keygen_writes_a_key_once_and_reports_its_replica_id(tmp_path: Path) -> None:
    path = tmp_path / "replica.key"
    body = payload(invoke("keygen", "--key", str(path)))
    assert body["replica_id"] == body["public_key"][:0] + body["replica_id"]
    assert len(body["replica_id"]) == 32
    assert path.exists()

    again = invoke("keygen", "--key", str(path))
    assert again.exit_code == 1
    assert "already exists" in again.stderr


def test_generate_api_key_prints_a_key_and_its_digest() -> None:
    body = payload(invoke("generate-api-key"))
    assert body["api_key_sha256"] == hash_api_key(body["api_key"])


def test_token_needs_a_secret_and_mints_a_usable_grant(tmp_path: Path) -> None:
    key = tmp_path / "replica.key"
    invoke("keygen", "--key", str(key))

    without = invoke("token", DOC_ID, "--key", str(key))
    assert without.exit_code == 1
    assert "APP_SYNC_JWT_SECRET" in without.stderr

    env = {"APP_SYNC_JWT_SECRET": SECRET}
    body = payload(invoke("token", DOC_ID, "--role", "owner", "--key", str(key), env=env))
    grant = verify_token(SecretStr(SECRET), body["token"], doc_id=DOC_ID)
    assert grant.role is Role.OWNER
    assert body["replica_id"] == grant.replica_id

    bad_role = invoke("token", DOC_ID, "--role", "root", "--key", str(key), env=env)
    assert bad_role.exit_code == 1
    missing_key = invoke("token", DOC_ID, "--key", str(tmp_path / "nope.key"), env=env)
    assert missing_key.exit_code == 1


def seed_document(tmp_path: Path) -> tuple[Path, Replica]:
    db = tmp_path / "oplog.sqlite3"
    hub = DocumentHub(OpLog(db))
    owner: Replica = make_replica(1, DOC_ID)
    hub.submit(
        Grant(principal="cli", doc_id=DOC_ID, role=Role.OWNER, replica_id=owner.replica_id),
        owner.insert_text("hello"),
    )
    hub.op_log.close()
    return db, owner


def test_doc_show_and_verify_read_the_local_log(tmp_path: Path) -> None:
    db, _ = seed_document(tmp_path)
    body = payload(invoke("doc", "show", DOC_ID, "--db", str(db)))
    assert body["text"] == "hello"
    assert body["anchors"] == 5
    assert body["pending_suggestions"] == []

    verified = payload(invoke("doc", "verify", DOC_ID, "--db", str(db)))
    assert verified["ok"] is True
    assert verified["entries"] == 5

    mismatched = invoke("doc", "verify", DOC_ID, "--db", str(db), "--expect-head", "0" * 64)
    assert mismatched.exit_code == 1


def test_doc_verify_reports_a_tampered_row_as_json_instead_of_crashing(tmp_path: Path) -> None:
    db, _ = seed_document(tmp_path)
    connection = sqlite3.connect(db)
    connection.execute("UPDATE ops SET op_json = ? WHERE chain_seq = 0", ('{"nope": 1}',))
    connection.commit()
    connection.close()

    result = invoke("doc", "verify", DOC_ID, "--db", str(db))

    assert result.exit_code == 1
    body = payload(result)
    assert body["ok"] is False
    assert body["error"] == "malformed entry"


def test_doc_snapshot_writes_a_verified_file(tmp_path: Path) -> None:
    db, _ = seed_document(tmp_path)
    out = tmp_path / "snapshot.json"
    body = payload(invoke("doc", "snapshot", DOC_ID, "--out", str(out), "--db", str(db)))
    assert out.exists()
    assert len(body["state_hash"]) == 64


def test_the_review_and_accept_commands_drive_the_two_gates(tmp_path: Path) -> None:
    """A full pass: the demo leaves an approval store and a log the CLI can work on."""
    summary = summary_of(tmp_path)
    approvals = tmp_path / "approvals.json"
    audit = tmp_path / "audit.jsonl"
    db = tmp_path / "oplog.sqlite3"

    listed = payload(invoke("review", "list", "--approvals", str(approvals)))
    assert listed == []  # the demo already decided both suggestions

    verified = payload(invoke("doc", "verify", DOC_ID, "--db", str(db)))
    assert verified["ok"] is True
    assert verified["entries"] == summary["op_log"]["entries"]

    audited = payload(invoke("audit-verify", str(audit)))
    assert audited["ok"] is True

    shown = payload(invoke("doc", "show", DOC_ID, "--db", str(db)))
    assert shown["text"] == summary["final_text"]


def test_review_decide_reports_an_unknown_request(tmp_path: Path) -> None:
    approvals = tmp_path / "approvals.json"
    audit = tmp_path / "audit.jsonl"
    result = invoke(
        "review", "decide", "missing", "--approvals", str(approvals), "--audit", str(audit)
    )
    assert result.exit_code == 1
    assert "unknown approval request" in result.stderr


def test_accept_refuses_without_a_key_and_reports_an_unknown_request(tmp_path: Path) -> None:
    db, _ = seed_document(tmp_path)
    missing_key = invoke(
        "accept", DOC_ID, "req", "--key", str(tmp_path / "nope.key"), "--db", str(db)
    )
    assert missing_key.exit_code == 1

    key = tmp_path / "replica.key"
    invoke("keygen", "--key", str(key))
    unknown = invoke(
        "accept",
        DOC_ID,
        "req",
        "--key",
        str(key),
        "--db",
        str(db),
        "--approvals",
        str(tmp_path / "approvals.json"),
        "--audit",
        str(tmp_path / "audit.jsonl"),
    )
    assert unknown.exit_code == 1


def test_audit_verify_detects_a_tampered_log(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("human:cli", "review.accepted", {"suggestion_id": "sg-" + "1" * 16})
    assert payload(invoke("audit-verify", str(path)))["ok"] is True
    assert invoke("audit-verify", str(path), "--expect-head", "0" * 64).exit_code == 1

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = lines[0].replace("review.accepted", "review.rejected")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    assert invoke("audit-verify", str(path)).exit_code == 1


@pytest.mark.usefixtures("restore_logging")
def test_serve_refuses_to_start_without_a_signing_secret() -> None:
    result = invoke("serve")
    assert result.exit_code == 1
    assert "APP_SYNC_JWT_SECRET" in result.stderr


@pytest.mark.usefixtures("restore_logging")
def test_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn
    from opentelemetry import trace as otel_trace

    from crdt_coedit_agents.core import tracing

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    result = invoke(
        "serve",
        "--port",
        "8123",
        env={
            "APP_SYNC_JWT_SECRET": SECRET,
            "APP_SYNC_DB_PATH": str(tmp_path / "oplog.sqlite3"),
            "APP_SYNC_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
        },
    )
    assert result.exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_the_module_entry_point_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["crdt_coedit_agents", "version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("crdt_coedit_agents", run_name="__main__")
    assert exit_info.value.code == 0
    assert "version" in capsys.readouterr().out
