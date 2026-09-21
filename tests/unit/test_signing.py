"""Signed operations: forgery, spoofed replica ids, replay and key handling."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.ops.model import FieldSet, SignedOp, UnsignedOp, signing_payload
from crdt_coedit_agents.ops.signing import (
    InvalidSignatureError,
    ReplicaIdentity,
    SigningError,
    VerifyingKey,
    replica_id_for,
    verify_op,
)
from tests.conftest import seeded_identity

pytestmark = pytest.mark.unit


def make_op(identity: ReplicaIdentity, *, seq: int = 1, value: str = "Notes") -> SignedOp:
    unsigned = UnsignedOp(
        doc_id="doc",
        replica_id=identity.replica_id,
        seq=seq,
        ts=Timestamp(1_000, 0, identity.replica_id).encode(),
        payload=FieldSet(key="title", value=value),
    )
    return identity.sign(unsigned)


def test_a_replica_id_is_derived_from_the_public_key() -> None:
    identity = seeded_identity(1)
    raw = bytes.fromhex(identity.public_key_hex)
    assert identity.replica_id == replica_id_for(raw)
    assert len(identity.replica_id) == 32
    assert VerifyingKey.from_hex(identity.public_key_hex).replica_id == identity.replica_id


def test_a_valid_operation_verifies_and_returns_its_key() -> None:
    identity = seeded_identity(1)
    key = verify_op(make_op(identity))
    assert key.replica_id == identity.replica_id


def test_a_tampered_payload_fails_verification() -> None:
    identity = seeded_identity(1)
    signed = make_op(identity)
    forged = signed.model_copy(
        update={"op": signed.op.model_copy(update={"payload": FieldSet(key="title", value="X")})}
    )
    with pytest.raises(InvalidSignatureError, match="does not verify"):
        verify_op(forged)


def test_a_spoofed_replica_id_is_detected_even_with_a_valid_signature() -> None:
    attacker = seeded_identity(2)
    victim = seeded_identity(3)
    unsigned = UnsignedOp(
        doc_id="doc",
        replica_id=victim.replica_id,
        seq=1,
        ts=Timestamp(1_000, 0, victim.replica_id).encode(),
        payload=FieldSet(key="title", value="stolen"),
    )
    # The attacker cannot even sign it as the victim...
    with pytest.raises(SigningError, match="another replica"):
        attacker.sign(unsigned)
    # ...and presenting its own key next to the victim's id is caught.
    forged = SignedOp(
        op=unsigned,
        public_key=attacker.public_key_hex,
        signature=attacker._private.sign(signing_payload(unsigned)).hex(),
    )
    with pytest.raises(InvalidSignatureError, match="does not match the public key"):
        verify_op(forged)


def test_a_timestamp_attributed_to_another_replica_is_refused() -> None:
    identity = seeded_identity(1)
    other = seeded_identity(4)
    with pytest.raises(ValueError, match="timestamp's replica id"):
        UnsignedOp(
            doc_id="doc",
            replica_id=identity.replica_id,
            seq=1,
            ts=Timestamp(1_000, 0, other.replica_id).encode(),
            payload=FieldSet(key="title", value="x"),
        )


def test_signatures_cover_the_document_id_so_operations_do_not_move_between_documents() -> None:
    identity = seeded_identity(1)
    signed = make_op(identity)
    moved = signed.model_copy(update={"op": signed.op.model_copy(update={"doc_id": "other"})})
    with pytest.raises(InvalidSignatureError):
        verify_op(moved)


def test_signatures_cover_the_sequence_number_so_operations_cannot_be_renumbered() -> None:
    identity = seeded_identity(1)
    signed = make_op(identity)
    renumbered = signed.model_copy(update={"op": signed.op.model_copy(update={"seq": 2})})
    with pytest.raises(InvalidSignatureError):
        verify_op(renumbered)


def test_malformed_keys_and_signatures_are_refused_without_crashing() -> None:
    with pytest.raises(SigningError, match="64 hex characters"):
        VerifyingKey.from_hex("zz")
    with pytest.raises(SigningError, match="32 bytes"):
        replica_id_for(b"short")
    identity = seeded_identity(1)
    key = identity.verifying_key()
    with pytest.raises(InvalidSignatureError):
        key.verify(b"message", "00" * 64)


def test_a_key_file_round_trips_and_is_owner_readable_only(tmp_path: Path) -> None:
    identity = ReplicaIdentity.generate()
    path = tmp_path / "replica.key"
    identity.save(path)
    loaded = ReplicaIdentity.load(path)
    assert loaded.replica_id == identity.replica_id
    assert loaded.public_key_hex == identity.public_key_hex
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_loading_a_broken_key_file_reports_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.key"
    path.write_text("not hex\n", encoding="utf-8")
    with pytest.raises(SigningError, match="could not read a replica key"):
        ReplicaIdentity.load(path)
    with pytest.raises(SigningError, match="32 bytes"):
        ReplicaIdentity.from_seed(b"too short")
    with pytest.raises(SigningError, match="could not read a replica key"):
        ReplicaIdentity.load(tmp_path / "missing.key")


def test_generated_identities_differ() -> None:
    assert ReplicaIdentity.generate().replica_id != ReplicaIdentity.generate().replica_id


def test_the_signing_payload_is_domain_separated() -> None:
    identity = seeded_identity(1)
    payload = signing_payload(make_op(identity).op)
    assert payload.startswith(b"crdt-coedit-op-v1\x00")
