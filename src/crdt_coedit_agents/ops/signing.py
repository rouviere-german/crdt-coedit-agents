"""Ed25519 signing and verification, with replica ids derived from public keys.

``replica_id = sha256(public_key_bytes)[:16].hex()``. Because the id is a function of the key, a
replica cannot claim an id it does not hold the key for: verification recomputes the id from the
presented key and compares it with the one inside the signed envelope. There is no key
distribution problem for *authenticity* — only for *authorization*, which the sync server solves
with per-document JWT roles that pin a connection to one replica id.

Replay protection is the ``(author, seq)`` pair: sequence numbers are strictly increasing per
author, so a re-sent operation is recognised as one already applied, and a *different* operation
reusing a spent sequence number is rejected outright rather than silently replacing the first one.

Private keys never leave this module in encoded form except through
:meth:`ReplicaIdentity.private_bytes`, which the CLI uses to persist a key file with 0600
permissions. Keys are generated with ``cryptography``'s OS-backed randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from crdt_coedit_agents.core.fileio import atomic_write_bytes
from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.ops.model import SignedOp, UnsignedOp, signing_payload

__all__ = [
    "InvalidSignatureError",
    "ReplicaIdentity",
    "SigningError",
    "VerifyingKey",
    "replica_id_for",
    "verify_op",
]


class SigningError(ValueError):
    """A key could not be loaded, or an operation could not be signed."""


class InvalidSignatureError(ValueError):
    """The signature, the public key or the derived replica id did not check out."""


def replica_id_for(public_key_bytes: bytes) -> str:
    """Derive the replica id from a raw Ed25519 public key."""
    if len(public_key_bytes) != 32:
        raise SigningError("an Ed25519 public key is 32 bytes")
    digest = hashes.Hash(hashes.SHA256())
    digest.update(public_key_bytes)
    return digest.finalize()[:16].hex()


@dataclass(frozen=True, slots=True)
class VerifyingKey:
    """A peer's public key together with the replica id it implies."""

    replica_id: str
    public_key_hex: str

    @classmethod
    def from_hex(cls, public_key_hex: str) -> VerifyingKey:
        try:
            raw = bytes.fromhex(public_key_hex)
        except ValueError as exc:
            raise SigningError("public keys are 64 hex characters") from exc
        return cls(replica_id=replica_id_for(raw), public_key_hex=public_key_hex)

    def _key(self) -> Ed25519PublicKey:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.public_key_hex))

    def verify(self, message: bytes, signature_hex: str) -> None:
        try:
            self._key().verify(bytes.fromhex(signature_hex), message)
        except (InvalidSignature, ValueError) as exc:
            raise InvalidSignatureError("signature does not verify") from exc


class ReplicaIdentity:
    """An Ed25519 key pair and the replica id derived from it."""

    __slots__ = ("_private", "_public_hex", "_replica_id")

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private = private_key
        raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        self._public_hex = raw.hex()
        self._replica_id = replica_id_for(raw)

    @classmethod
    def generate(cls) -> ReplicaIdentity:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> ReplicaIdentity:
        """Deterministic identity from 32 seed bytes. For tests, demos and benchmarks only."""
        if len(seed) != 32:
            raise SigningError("an Ed25519 private key seed is 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    @classmethod
    def load(cls, path: Path) -> ReplicaIdentity:
        try:
            raw = bytes.fromhex(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise SigningError(f"could not read a replica key from {path.name}") from exc
        return cls.from_seed(raw)

    def save(self, path: Path) -> None:
        """Persist the private key seed, readable only by the owner."""
        atomic_write_bytes(path, self.private_bytes().hex().encode("ascii") + b"\n")
        path.chmod(0o600)

    @property
    def replica_id(self) -> str:
        return self._replica_id

    @property
    def public_key_hex(self) -> str:
        return self._public_hex

    def verifying_key(self) -> VerifyingKey:
        return VerifyingKey(replica_id=self._replica_id, public_key_hex=self._public_hex)

    def private_bytes(self) -> bytes:
        return self._private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def sign(self, op: UnsignedOp) -> SignedOp:
        if op.replica_id != self._replica_id:
            raise SigningError("cannot sign an operation attributed to another replica")
        signature = self._private.sign(signing_payload(op))
        return SignedOp(op=op, public_key=self._public_hex, signature=signature.hex())


def verify_op(signed: SignedOp) -> VerifyingKey:
    """Check the signature and that the replica id really belongs to the presented key.

    Returns the verifying key so callers can bind it to a connection or store it. Raises
    :class:`InvalidSignatureError` on a forged signature, a spoofed replica id, a key of the wrong
    length or a timestamp that does not name its author.
    """
    key = VerifyingKey.from_hex(signed.public_key)
    if key.replica_id != signed.op.replica_id:
        raise InvalidSignatureError(
            "replica id does not match the public key it claims to be derived from"
        )
    if Timestamp.decode(signed.op.ts).replica_id != signed.op.replica_id:
        raise InvalidSignatureError("timestamp replica id does not match the operation author")
    key.verify(signing_payload(signed.op), signed.signature)
    return key
