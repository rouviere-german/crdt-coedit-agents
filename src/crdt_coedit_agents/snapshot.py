"""Document snapshots: write-then-rename, hash-verified on load.

A snapshot is a compaction, not a source of truth: it stores the materialized CRDT state and the
version vector that produced it, so a replica can start from it instead of replaying the whole
operation log. It is written to a temporary file in the same directory and renamed over the target,
so a crash mid-write leaves either the old snapshot or the new one, never a half-written file.

The stored ``state_hash`` is recomputed on load. That catches truncation and accidental corruption.
It is *not* a signature: anyone who can write the file can write a consistent one, which is why a
snapshot is only ever trusted as far as the filesystem it came from, and why the operation log —
whose entries are individually signed — stays the authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crdt_coedit_agents.core.fileio import atomic_write_text
from crdt_coedit_agents.crdt.document import DocumentState
from crdt_coedit_agents.crdt.ids import VersionVector

__all__ = ["Snapshot", "SnapshotError", "load_snapshot", "write_snapshot"]

SNAPSHOT_VERSION = 1


class SnapshotError(ValueError):
    """A snapshot file is missing, malformed or does not match its recorded hash."""


class Snapshot(BaseModel):
    """The on-disk form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    v: int = Field(default=SNAPSHOT_VERSION, ge=1, le=SNAPSHOT_VERSION)
    doc_id: str
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_vector: dict[str, int]
    state: dict[str, Any]

    def restore(self) -> tuple[DocumentState, VersionVector]:
        """Rebuild the state and verify it hashes to what the file claims."""
        try:
            state = DocumentState.from_json(self.state)
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError(f"snapshot for {self.doc_id} is malformed") from exc
        if state.state_hash() != self.state_hash:
            raise SnapshotError(
                f"snapshot for {self.doc_id} does not match its recorded state hash"
            )
        if state.doc_id != self.doc_id:
            raise SnapshotError("snapshot document id does not match its contents")
        return state, VersionVector(self.version_vector)


def write_snapshot(path: Path, state: DocumentState, version_vector: VersionVector) -> Snapshot:
    """Write a snapshot atomically and return what was written."""
    snapshot = Snapshot(
        doc_id=state.doc_id,
        state_hash=state.state_hash(),
        version_vector=version_vector.as_dict(),
        state=state.to_json(),
    )
    atomic_write_text(path, snapshot.model_dump_json(indent=2) + "\n")
    return snapshot


def load_snapshot(path: Path) -> tuple[DocumentState, VersionVector]:
    """Load and verify a snapshot. Raises :class:`SnapshotError` on anything unexpected."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotError(f"could not read snapshot {path.name}") from exc
    try:
        snapshot = Snapshot.model_validate_json(raw)
    except ValueError as exc:
        raise SnapshotError(f"snapshot {path.name} is not a valid snapshot file") from exc
    return snapshot.restore()


def snapshot_summary(path: Path) -> dict[str, Any]:
    """Header fields of a snapshot without rebuilding the state (for the CLI)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"could not read snapshot {path.name}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(f"snapshot {path.name} is not an object")
    return {
        "doc_id": payload.get("doc_id"),
        "state_hash": payload.get("state_hash"),
        "version_vector": payload.get("version_vector"),
    }
