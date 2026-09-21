"""Hash-chained, append-only operation log backed by SQLite.

One chain per document. Row ``n`` stores ``entry_hash = sha256(canonical_json({"seq": n,
"prev_hash": …, "op": …}))``, so editing, deleting, reordering or re-encoding any row breaks every
hash after it. ``verify`` walks the chain, recomputes each hash and re-checks every Ed25519
signature, which makes the log tamper-evident even against someone with write access to the
database file — they can rebuild the chain, but not forge the signatures inside it.

The unique constraint on ``(doc_id, replica_id, author_seq)`` is the replay guard: a re-sent
operation is recognised as one already stored, while a *different* operation that reuses a spent
sequence number is refused instead of quietly overwriting the first.

Durability settings are chosen for a log: ``journal_mode=WAL`` and ``synchronous=FULL``, so a
committed append survives a process or host crash.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from crdt_coedit_agents.core.canonical import canonical_json, sha256_hex
from crdt_coedit_agents.crdt.ids import Dot, VersionVector
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.ops.signing import InvalidSignatureError, verify_op

__all__ = ["GENESIS_HASH", "ChainVerification", "LogEntry", "OpLog", "OpLogError"]

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ops (
    doc_id      TEXT    NOT NULL,
    chain_seq   INTEGER NOT NULL,
    replica_id  TEXT    NOT NULL,
    author_seq  INTEGER NOT NULL,
    kind        TEXT    NOT NULL,
    suggestion_id TEXT,
    op_json     TEXT    NOT NULL,
    prev_hash   TEXT    NOT NULL,
    entry_hash  TEXT    NOT NULL,
    PRIMARY KEY (doc_id, chain_seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS ops_author_seq
    ON ops (doc_id, replica_id, author_seq);
CREATE INDEX IF NOT EXISTS ops_by_replica
    ON ops (doc_id, replica_id, author_seq);
"""


class OpLogError(RuntimeError):
    """The log refused an append or failed verification."""


@dataclass(frozen=True, slots=True)
class LogEntry:
    chain_seq: int
    signed: SignedOp
    prev_hash: str
    entry_hash: str


@dataclass(frozen=True, slots=True)
class ChainVerification:
    ok: bool
    entries: int
    head: str
    error: str | None = None
    chain_seq: int | None = None


def entry_hash(chain_seq: int, prev_hash: str, op_json: str) -> str:
    """The hash stored on a row. ``op_json`` must already be canonical."""
    return sha256_hex(canonical_json({"seq": chain_seq, "prev_hash": prev_hash, "op": op_json}))


def _canonical_op(signed: SignedOp) -> str:
    return canonical_json(signed.model_dump(mode="json"))


class OpLog:
    """Append-only operation log. Safe to share between threads; one connection, one lock."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # ---------------------------------------------------------------- reading

    def head(self, doc_id: str) -> tuple[int, str]:
        """``(next chain sequence number, head hash)`` for a document."""
        with self._lock:
            row = self._connection.execute(
                "SELECT chain_seq, entry_hash FROM ops WHERE doc_id = ? "
                "ORDER BY chain_seq DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
        if row is None:
            return 0, GENESIS_HASH
        return int(row["chain_seq"]) + 1, str(row["entry_hash"])

    def count(self, doc_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM ops WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return int(row["n"])

    def documents(self) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT doc_id FROM ops ORDER BY doc_id"
            ).fetchall()
        return [str(row["doc_id"]) for row in rows]

    def entries(self, doc_id: str) -> Iterator[LogEntry]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT chain_seq, op_json, prev_hash, entry_hash FROM ops "
                "WHERE doc_id = ? ORDER BY chain_seq",
                (doc_id,),
            ).fetchall()
        for row in rows:
            yield LogEntry(
                chain_seq=int(row["chain_seq"]),
                signed=SignedOp.model_validate_json(str(row["op_json"])),
                prev_hash=str(row["prev_hash"]),
                entry_hash=str(row["entry_hash"]),
            )

    def version_vector(self, doc_id: str) -> VersionVector:
        """The highest *contiguous* author sequence number per replica."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT replica_id, author_seq FROM ops WHERE doc_id = ? "
                "ORDER BY replica_id, author_seq",
                (doc_id,),
            ).fetchall()
        vector = VersionVector()
        for row in rows:
            vector.observe(Dot(replica_id=str(row["replica_id"]), seq=int(row["author_seq"])))
        return vector

    def since(self, doc_id: str, peer: VersionVector, *, limit: int = 4096) -> list[SignedOp]:
        """Operations the peer's version vector does not cover, in chain order."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT replica_id, author_seq, op_json FROM ops WHERE doc_id = ? "
                "ORDER BY chain_seq",
                (doc_id,),
            ).fetchall()
        result: list[SignedOp] = []
        for row in rows:
            dot = Dot(replica_id=str(row["replica_id"]), seq=int(row["author_seq"]))
            if dot in peer:
                continue
            result.append(SignedOp.model_validate_json(str(row["op_json"])))
            if len(result) >= limit:
                break
        return result

    def contains(self, doc_id: str, dot: Dot) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM ops WHERE doc_id = ? AND replica_id = ? AND author_seq = ?",
                (doc_id, dot.replica_id, dot.seq),
            ).fetchone()
        return row is not None

    # ---------------------------------------------------------------- writing

    def append(self, signed: SignedOp) -> LogEntry | None:
        """Append one operation. Returns ``None`` when it was already stored (a duplicate).

        Raises :class:`OpLogError` when a different operation reuses a spent
        ``(replica_id, author_seq)`` pair.
        """
        doc_id = signed.op.doc_id
        op_json = _canonical_op(signed)
        with self._lock:
            stored = self._connection.execute(
                "SELECT op_json FROM ops WHERE doc_id = ? AND replica_id = ? AND author_seq = ?",
                (doc_id, signed.op.replica_id, signed.op.seq),
            ).fetchone()
            if stored is not None:
                if str(stored["op_json"]) != op_json:
                    raise OpLogError(
                        f"replay: ({signed.op.replica_id}, {signed.op.seq}) is already stored "
                        "with different content"
                    )
                return None
            chain_seq, prev = self.head(doc_id)
            digest = entry_hash(chain_seq, prev, op_json)
            try:
                self._connection.execute(
                    "INSERT INTO ops (doc_id, chain_seq, replica_id, author_seq, kind, "
                    "suggestion_id, op_json, prev_hash, entry_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        chain_seq,
                        signed.op.replica_id,
                        signed.op.seq,
                        signed.op.kind,
                        signed.op.suggestion_id,
                        op_json,
                        prev,
                        digest,
                    ),
                )
            except sqlite3.IntegrityError as exc:  # pragma: no cover - guarded above
                raise OpLogError(f"could not append operation: {exc}") from exc
            return LogEntry(chain_seq=chain_seq, signed=signed, prev_hash=prev, entry_hash=digest)

    def append_many(self, ops: Iterable[SignedOp]) -> list[LogEntry]:
        appended: list[LogEntry] = []
        with self._lock:
            for signed in ops:
                entry = self.append(signed)
                if entry is not None:
                    appended.append(entry)
        return appended

    # ------------------------------------------------------------ verification

    def verify(self, doc_id: str, *, expect_head: str | None = None) -> ChainVerification:
        """Recompute the chain and re-check every signature. Reports the first failure."""
        prev = GENESIS_HASH
        count = 0
        for entry in self.entries(doc_id):
            if entry.chain_seq != count:
                return ChainVerification(
                    False, count, prev, "sequence gap or reordering", entry.chain_seq
                )
            if entry.prev_hash != prev:
                return ChainVerification(False, count, prev, "broken hash chain", entry.chain_seq)
            op_json = _canonical_op(entry.signed)
            if entry_hash(entry.chain_seq, entry.prev_hash, op_json) != entry.entry_hash:
                return ChainVerification(
                    False, count, prev, "entry content modified", entry.chain_seq
                )
            try:
                verify_op(entry.signed)
            except InvalidSignatureError:
                return ChainVerification(
                    False, count, prev, "invalid operation signature", entry.chain_seq
                )
            prev = entry.entry_hash
            count += 1
        if expect_head is not None and expect_head != prev:
            return ChainVerification(
                False, count, prev, "head mismatch (truncated or rolled back)", None
            )
        return ChainVerification(True, count, prev)
