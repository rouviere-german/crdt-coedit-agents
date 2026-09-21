"""The hash-chained SQLite operation log: replay protection, chain integrity, catch-up."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import Dot, VersionVector
from crdt_coedit_agents.ops.log import GENESIS_HASH, OpLog, OpLogError, entry_hash
from crdt_coedit_agents.ops.model import FieldSet, SignedOp, UnsignedOp
from crdt_coedit_agents.replica import Replica
from tests.conftest import make_replica, seeded_identity

pytestmark = pytest.mark.unit

DOC = "test-doc"


def sample_ops(count: int = 4) -> tuple[Replica, list[SignedOp]]:
    replica = make_replica(1)
    replica.insert_text("abc"[: count - 1] or "a")
    replica.set_field("title", "Notes")
    return replica, list(replica.ops())


def test_appending_builds_a_chain_from_the_genesis_hash() -> None:
    _, ops = sample_ops()
    with OpLog() as log:
        assert log.head(DOC) == (0, GENESIS_HASH)
        entries = log.append_many(ops)
        assert [entry.chain_seq for entry in entries] == list(range(len(ops)))
        assert entries[0].prev_hash == GENESIS_HASH
        assert entries[1].prev_hash == entries[0].entry_hash
        assert log.count(DOC) == len(ops)
        assert log.documents() == [DOC]
        assert log.verify(DOC).ok


def test_a_re_sent_operation_is_a_duplicate_and_a_different_one_is_a_replay() -> None:
    _, ops = sample_ops()
    with OpLog() as log:
        log.append(ops[0])
        assert log.append(ops[0]) is None

        identity = seeded_identity(1)
        collision = identity.sign(
            UnsignedOp(
                doc_id=DOC,
                replica_id=identity.replica_id,
                seq=ops[0].op.seq,
                ts=Timestamp(9_999, 0, identity.replica_id).encode(),
                payload=FieldSet(key="title", value="forged"),
            )
        )
        with pytest.raises(OpLogError, match="replay"):
            log.append(collision)


def test_verification_detects_an_edited_row(tmp_path: Path) -> None:
    _, ops = sample_ops()
    path = tmp_path / "oplog.sqlite3"
    with OpLog(path) as log:
        log.append_many(ops)
        assert log.verify(DOC).ok

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE ops SET op_json = replace(op_json, 'Notes', 'Nope') WHERE chain_seq = ?",
        (len(ops) - 1,),
    )
    connection.commit()
    connection.close()

    with OpLog(path) as reopened:
        result = reopened.verify(DOC)
    assert not result.ok
    assert result.error in {"entry content modified", "invalid operation signature"}
    assert result.chain_seq == len(ops) - 1


def test_verification_detects_a_rebuilt_chain_with_a_forged_signature(tmp_path: Path) -> None:
    """Whoever can write the database can rebuild the chain — but not forge a signature."""
    _, ops = sample_ops()
    path = tmp_path / "oplog.sqlite3"
    with OpLog(path) as log:
        log.append_many(ops)

    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT chain_seq, op_json, prev_hash FROM ops WHERE op_json LIKE '%Notes%'"
    ).fetchone()
    chain_seq, op_json, prev_hash = int(row[0]), str(row[1]), str(row[2])
    tampered = op_json.replace('"Notes"', '"Nope!"')
    assert tampered != op_json
    # Recompute the row hash and drop everything after it, exactly as a careful attacker would.
    connection.execute(
        "UPDATE ops SET op_json = ?, entry_hash = ? WHERE chain_seq = ?",
        (tampered, entry_hash(chain_seq, prev_hash, tampered), chain_seq),
    )
    connection.execute("DELETE FROM ops WHERE chain_seq > ?", (chain_seq,))
    connection.commit()
    connection.close()

    with OpLog(path) as reopened:
        result = reopened.verify(DOC)
    assert not result.ok
    assert result.error == "invalid operation signature"
    assert result.chain_seq == chain_seq


def test_truncating_the_tail_needs_an_anchored_head(tmp_path: Path) -> None:
    _, ops = sample_ops()
    path = tmp_path / "oplog.sqlite3"
    with OpLog(path) as log:
        log.append_many(ops)
        head = log.head(DOC)[1]

    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM ops WHERE chain_seq >= ?", (len(ops) - 1,))
    connection.commit()
    connection.close()

    with OpLog(path) as reopened:
        assert reopened.verify(DOC).ok
        anchored = reopened.verify(DOC, expect_head=head)
    assert not anchored.ok
    assert anchored.error is not None
    assert "truncated" in anchored.error


def test_catch_up_returns_exactly_what_a_version_vector_is_missing() -> None:
    replica, ops = sample_ops()
    with OpLog() as log:
        log.append_many(ops)
        assert log.version_vector(DOC) == replica.version_vector
        assert len(log.since(DOC, VersionVector())) == len(ops)
        assert log.since(DOC, replica.version_vector) == []

        partial = VersionVector({replica.replica_id: 1})
        assert len(log.since(DOC, partial)) == len(ops) - 1
        assert len(log.since(DOC, VersionVector(), limit=2)) == 2


def test_membership_and_empty_documents_behave() -> None:
    replica, ops = sample_ops()
    with OpLog() as log:
        log.append_many(ops)
        assert log.contains(DOC, ops[0].dot)
        assert not log.contains(DOC, Dot(replica_id=replica.replica_id, seq=999))
        assert log.count("other") == 0
        assert log.verify("other").ok
        assert log.version_vector("other") == VersionVector()


def test_the_log_survives_being_closed_and_reopened(tmp_path: Path) -> None:
    _, ops = sample_ops()
    path = tmp_path / "nested" / "oplog.sqlite3"
    with OpLog(path) as log:
        log.append_many(ops)
        head = log.head(DOC)
    with OpLog(path) as reopened:
        assert reopened.head(DOC) == head
        assert reopened.verify(DOC, expect_head=head[1]).ok


def test_verification_reports_a_row_that_is_not_an_operation_at_all(tmp_path: Path) -> None:
    """Tamper-evidence has to be *reported*, not raised: `doc verify` prints JSON and exits 1."""
    _, ops = sample_ops()
    path = tmp_path / "oplog.sqlite3"
    with OpLog(path) as log:
        log.append_many(ops)

    connection = sqlite3.connect(path)
    connection.execute("UPDATE ops SET op_json = ? WHERE chain_seq = 0", ('{"nope": 1}',))
    connection.commit()
    connection.close()

    with OpLog(path) as reopened:
        result = reopened.verify(DOC)
    assert result.ok is False
    assert result.error == "malformed entry"
    assert result.chain_seq == 0
    assert result.entries == 0
