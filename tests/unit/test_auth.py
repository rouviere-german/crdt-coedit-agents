"""Per-document JWT roles and the operation policy they enforce."""

from __future__ import annotations

import time

import jwt
import pytest
from pydantic import SecretStr

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.ops.model import FieldSet, SignedOp, SuggestionDecide, UnsignedOp
from crdt_coedit_agents.sync.auth import (
    AUDIENCE,
    ISSUER,
    AuthorizationError,
    Grant,
    Role,
    TokenError,
    authorize_op,
    issue_token,
    verify_token,
)
from tests.conftest import seeded_identity

pytestmark = pytest.mark.unit

SECRET = SecretStr("a-long-test-signing-secret-value")
OTHER_SECRET = SecretStr("a-different-long-test-secret-val")
DOC = "doc"
SUGGESTION = "sg-" + "1" * 16
IDENTITY = seeded_identity(1)


def mint(role: Role = Role.EDITOR, **overrides: object) -> str:
    values: dict[str, object] = {
        "principal": "user:ana",
        "doc_id": DOC,
        "role": role,
        "replica_id": IDENTITY.replica_id,
    }
    values.update(overrides)
    return issue_token(SECRET, **values)  # type: ignore[arg-type]


def grant(role: Role, replica_id: str | None = None) -> Grant:
    return Grant(
        principal="user:ana",
        doc_id=DOC,
        role=role,
        replica_id=replica_id or IDENTITY.replica_id,
    )


def op(
    *,
    suggestion_id: str | None = None,
    decide: bool = False,
    doc_id: str = DOC,
    identity: object = IDENTITY,
) -> SignedOp:
    signer = identity if isinstance(identity, type(IDENTITY)) else IDENTITY
    payload = (
        SuggestionDecide(suggestion_id=SUGGESTION, accept=True)
        if decide
        else FieldSet(key="title", value="x")
    )
    unsigned = UnsignedOp(
        doc_id=doc_id,
        replica_id=signer.replica_id,
        seq=1,
        ts=Timestamp(1_000, 0, signer.replica_id).encode(),
        suggestion_id=suggestion_id,
        payload=payload,
    )
    return signer.sign(unsigned)


def test_a_valid_token_yields_the_grant_it_carries() -> None:
    verified = verify_token(SECRET, mint(Role.OWNER), doc_id=DOC)
    assert verified == grant(Role.OWNER)
    assert verified.may_decide
    assert verified.may_write_main


def test_a_token_for_another_document_or_secret_is_refused() -> None:
    with pytest.raises(TokenError, match="different document"):
        verify_token(SECRET, mint(), doc_id="other-doc")
    with pytest.raises(TokenError, match="token rejected"):
        verify_token(OTHER_SECRET, mint(), doc_id=DOC)


def test_expired_and_future_tokens_are_refused() -> None:
    past = mint(now=lambda: time.time() - 7_200, ttl_seconds=60)
    with pytest.raises(TokenError, match="ExpiredSignatureError"):
        verify_token(SECRET, past, doc_id=DOC)
    ahead = mint(now=lambda: time.time() + 7_200)
    with pytest.raises(TokenError, match="ImmatureSignatureError"):
        verify_token(SECRET, ahead, doc_id=DOC)


def test_unsigned_tokens_are_refused_because_the_algorithm_is_pinned() -> None:
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "attacker",
        "doc": DOC,
        "role": "owner",
        "rid": IDENTITY.replica_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
    }
    none_token = jwt.encode(claims, key="", algorithm="none")
    with pytest.raises(TokenError, match="token rejected"):
        verify_token(SECRET, none_token, doc_id=DOC)


def test_tokens_missing_or_misusing_claims_are_refused() -> None:
    with pytest.raises(TokenError, match="missing or too large"):
        verify_token(SECRET, "", doc_id=DOC)
    with pytest.raises(TokenError, match="missing or too large"):
        verify_token(SECRET, "x" * 5_000, doc_id=DOC)

    def forge(**claims: object) -> str:
        base: dict[str, object] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "u",
            "doc": DOC,
            "role": "editor",
            "rid": IDENTITY.replica_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        }
        base.update(claims)
        return jwt.encode(base, SECRET.get_secret_value(), algorithm="HS256")

    with pytest.raises(TokenError, match="subject or the document"):
        verify_token(SECRET, forge(doc=7), doc_id=DOC)
    with pytest.raises(TokenError, match="replica id"):
        verify_token(SECRET, forge(rid="short"), doc_id=DOC)
    with pytest.raises(TokenError, match="missing the role"):
        verify_token(SECRET, forge(role=None), doc_id=DOC)
    with pytest.raises(TokenError, match="unknown role"):
        verify_token(SECRET, forge(role="root"), doc_id=DOC)


def test_token_lifetimes_are_bounded() -> None:
    with pytest.raises(TokenError, match="lifetimes"):
        mint(ttl_seconds=0)
    with pytest.raises(TokenError, match="lifetimes"):
        mint(ttl_seconds=90_000)


def test_a_suggester_may_only_write_suggestion_scoped_content() -> None:
    suggester = grant(Role.SUGGESTER)
    authorize_op(suggester, op(suggestion_id=SUGGESTION))
    with pytest.raises(AuthorizationError, match="only write operations tagged"):
        authorize_op(suggester, op())
    with pytest.raises(AuthorizationError, match="may not decide"):
        authorize_op(suggester, op(decide=True))


def test_an_editor_writes_the_document_but_does_not_decide_suggestions() -> None:
    editor = grant(Role.EDITOR)
    authorize_op(editor, op())
    authorize_op(editor, op(suggestion_id=SUGGESTION))
    with pytest.raises(AuthorizationError, match="may not decide"):
        authorize_op(editor, op(decide=True))


def test_an_owner_may_do_everything_including_deciding() -> None:
    owner = grant(Role.OWNER)
    authorize_op(owner, op())
    authorize_op(owner, op(suggestion_id=SUGGESTION))
    authorize_op(owner, op(decide=True))


def test_a_connection_may_only_publish_as_the_replica_its_token_pins() -> None:
    other = seeded_identity(2)
    with pytest.raises(AuthorizationError, match="does not match the replica"):
        authorize_op(grant(Role.OWNER), op(identity=other))
    with pytest.raises(AuthorizationError, match="different document"):
        authorize_op(grant(Role.OWNER, IDENTITY.replica_id), op(doc_id="elsewhere"))
