"""Per-document JWT roles and the server-side operation policy.

A token grants one *principal* one *role* on one *document*, and pins it to one *replica id*. The
replica id in the token must match the author of every operation the connection submits, so a
compromised or curious client cannot publish under someone else's identity even though it holds a
valid token — it would also need that replica's Ed25519 private key.

Roles form a ladder:

``owner``
    every operation kind, including ``suggestion_decide``. Accepting an agent suggestion is an
    owner action, so exactly one role is accountable for the accept gate.
``editor``
    every operation kind **except** ``suggestion_decide``: editors write the document, owners
    decide what agent suggestions become part of it.
``suggester``
    only content operations that carry a ``suggestion_id``. Agents get this role, so an agent
    physically cannot write an untagged operation or an accept flag, whatever it is persuaded to
    try.

Tokens are HS256 with a required ``exp``, ``iat``, ``iss`` and ``aud``. The algorithm list is
pinned, so the ``alg: none`` and "HMAC with the RSA public key" confusions do not apply.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jwt
from pydantic import SecretStr

from crdt_coedit_agents.crdt.ids import REPLICA_ID_PATTERN
from crdt_coedit_agents.ops.model import SUGGESTABLE_KINDS, SignedOp

__all__ = [
    "AUDIENCE",
    "ISSUER",
    "AuthorizationError",
    "Grant",
    "Role",
    "TokenError",
    "authorize_op",
    "issue_token",
    "verify_token",
]

ISSUER = "crdt-coedit-agents"
AUDIENCE = "crdt-coedit-sync"
_ALGORITHM = "HS256"
_MAX_TOKEN_BYTES = 4_096


class Role(StrEnum):
    OWNER = "owner"
    EDITOR = "editor"
    SUGGESTER = "suggester"


class TokenError(ValueError):
    """A token is missing, malformed, expired or not valid for this document."""


class AuthorizationError(PermissionError):
    """A syntactically valid operation is not permitted for this grant."""


@dataclass(frozen=True, slots=True)
class Grant:
    """What a verified token allows."""

    principal: str
    doc_id: str
    role: Role
    replica_id: str

    @property
    def may_decide(self) -> bool:
        return self.role is Role.OWNER

    @property
    def may_write_main(self) -> bool:
        return self.role in (Role.OWNER, Role.EDITOR)


def issue_token(
    secret: SecretStr,
    *,
    principal: str,
    doc_id: str,
    role: Role,
    replica_id: str,
    ttl_seconds: int = 3600,
    now: Callable[[], float] = time.time,
) -> str:
    """Mint a token. The caller is responsible for authenticating ``principal`` first."""
    if ttl_seconds < 1 or ttl_seconds > 86_400:
        raise TokenError("token lifetimes must be between 1 second and 24 hours")
    issued = int(now())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": principal,
        "doc": doc_id,
        "role": role.value,
        "rid": replica_id,
        "iat": issued,
        "nbf": issued,
        "exp": issued + ttl_seconds,
    }
    return jwt.encode(claims, secret.get_secret_value(), algorithm=_ALGORITHM)


def verify_token(secret: SecretStr, token: str, *, doc_id: str) -> Grant:
    """Verify a token for ``doc_id`` and return the grant it carries."""
    if not token or len(token.encode("utf-8")) > _MAX_TOKEN_BYTES:
        raise TokenError("token is missing or too large")
    try:
        claims = jwt.decode(
            token,
            secret.get_secret_value(),
            algorithms=[_ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except jwt.PyJWTError as exc:
        # The message is a class name, never the token or the signature.
        raise TokenError(f"token rejected ({type(exc).__name__})") from exc
    principal = claims.get("sub")
    document = claims.get("doc")
    replica_id = claims.get("rid")
    raw_role = claims.get("role")
    if not isinstance(principal, str) or not isinstance(document, str):
        raise TokenError("token is missing the subject or the document claim")
    if not isinstance(replica_id, str) or len(replica_id) != 32:
        raise TokenError(f"token replica id must match {REPLICA_ID_PATTERN}")
    if document != doc_id:
        raise TokenError("token was issued for a different document")
    if not isinstance(raw_role, str):
        raise TokenError("token is missing the role claim")
    try:
        role = Role(raw_role)
    except ValueError as exc:
        raise TokenError("token carries an unknown role") from exc
    return Grant(principal=principal, doc_id=doc_id, role=role, replica_id=replica_id)


def authorize_op(grant: Grant, signed: SignedOp) -> None:
    """Enforce the role policy for one operation. Raises :class:`AuthorizationError`."""
    op = signed.op
    if op.doc_id != grant.doc_id:
        raise AuthorizationError("operation targets a different document than the token")
    if op.replica_id != grant.replica_id:
        raise AuthorizationError("operation author does not match the replica the token pins")
    if op.kind == "suggestion_decide":
        if not grant.may_decide:
            raise AuthorizationError(f"role {grant.role.value} may not decide suggestions")
        return
    if op.suggestion_id is None:
        if not grant.may_write_main:
            raise AuthorizationError(
                f"role {grant.role.value} may only write operations tagged with a suggestion id"
            )
        return
    if op.kind not in SUGGESTABLE_KINDS:  # pragma: no cover - the model already refuses this
        raise AuthorizationError(f"{op.kind} operations cannot belong to a suggestion")
