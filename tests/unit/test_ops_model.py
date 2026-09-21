"""Operation validation: every limit in the spec is enforced by the model itself."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.ops.model import (
    MAX_FIELD_DEPTH,
    MAX_FIELD_VALUE_BYTES,
    MAX_OPS_PER_DELTA,
    MAX_SEAL_RUNS,
    MAX_TEXT_VALUE_BYTES,
    Delta,
    FieldSet,
    SuggestionDecide,
    TagAdd,
    TagRemove,
    TextDelete,
    TextInsert,
    UnsignedOp,
)
from tests.conftest import seeded_identity

pytestmark = pytest.mark.unit

IDENTITY = seeded_identity(1)
ANCHOR = f"{IDENTITY.replica_id}.1"


def envelope(**overrides: object) -> UnsignedOp:
    values: dict[str, object] = {
        "doc_id": "doc",
        "replica_id": IDENTITY.replica_id,
        "seq": 1,
        "ts": Timestamp(1_000, 0, IDENTITY.replica_id).encode(),
        "payload": FieldSet(key="title", value="x"),
    }
    values.update(overrides)
    return UnsignedOp(**values)


def test_text_values_are_bounded_and_may_not_contain_nul() -> None:
    assert TextInsert(value="a").value == "a"
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        TextInsert(value="x" * (MAX_TEXT_VALUE_BYTES + 1))
    with pytest.raises(ValidationError, match="NUL"):
        TextInsert(value="a\x00b")
    with pytest.raises(ValidationError, match="at least 1"):
        TextInsert(value="")


def test_anchors_must_look_like_dots() -> None:
    assert TextInsert(parent=ANCHOR, value="a").parent_dot is not None
    assert TextInsert(value="a").parent_dot is None
    with pytest.raises(ValidationError, match="anchor id"):
        TextInsert(parent="not-a-dot", value="a")
    with pytest.raises(ValidationError):
        TextDelete(target="nope")
    assert TextDelete(target=ANCHOR).target_dot.seq == 1


def test_field_values_are_bounded_in_size_and_depth() -> None:
    assert FieldSet(key="title", value={"a": [1, 2]}).value == {"a": [1, 2]}
    with pytest.raises(ValidationError, match="limited to"):
        FieldSet(key="title", value="x" * (MAX_FIELD_VALUE_BYTES + 1))

    deep: object = "leaf"
    for _ in range(MAX_FIELD_DEPTH + 2):
        deep = {"n": deep}
    with pytest.raises(ValidationError, match="nest at most"):
        FieldSet(key="title", value=deep)
    with pytest.raises(ValidationError):
        FieldSet(key="Not A Key", value=1)


def test_tags_are_bounded_and_observed_removes_carry_dots() -> None:
    assert TagAdd(tag="draft").tag == "draft"
    with pytest.raises(ValidationError):
        TagAdd(tag="tag with <html>")
    removal = TagRemove(tag="draft", tags=(ANCHOR,))
    assert removal.observed[0].seq == 1
    with pytest.raises(ValidationError, match="must be dots"):
        TagRemove(tag="draft", tags=("nope",))
    with pytest.raises(ValidationError, match="too many observed tags"):
        TagRemove(tag="draft", tags=tuple([ANCHOR] * (MAX_OPS_PER_DELTA + 1)))


def test_a_suggestion_cannot_decide_itself() -> None:
    decide = SuggestionDecide(suggestion_id="sg-" + "1" * 16, accept=True)
    assert envelope(payload=decide).kind == "suggestion_decide"
    with pytest.raises(ValidationError, match="cannot belong to a suggestion"):
        envelope(payload=decide, suggestion_id="sg-" + "2" * 16)
    with pytest.raises(ValidationError):
        SuggestionDecide(suggestion_id="not-a-suggestion", accept=True)


def test_the_envelope_binds_the_timestamp_to_its_author() -> None:
    other = seeded_identity(2)
    with pytest.raises(ValidationError, match="timestamp's replica id"):
        envelope(ts=Timestamp(1_000, 0, other.replica_id).encode())
    with pytest.raises(ValidationError, match="malformed timestamp"):
        envelope(ts="not-a-timestamp")


def test_envelopes_reject_unknown_fields_and_bad_identifiers() -> None:
    with pytest.raises(ValidationError):
        envelope(extra="surprise")
    with pytest.raises(ValidationError):
        envelope(doc_id="Not A Doc")
    with pytest.raises(ValidationError):
        envelope(seq=0)


def test_a_delta_is_bounded_and_single_document() -> None:
    signed = IDENTITY.sign(envelope())
    delta = Delta(doc_id="doc", ops=(signed,))
    assert delta.encoded_size() > 0
    with pytest.raises(ValidationError, match="must belong to the delta's document"):
        Delta(doc_id="other", ops=(signed,))
    with pytest.raises(ValidationError):
        Delta(doc_id="doc", ops=tuple([signed] * (MAX_OPS_PER_DELTA + 1)))


def test_signed_operations_expose_their_identity() -> None:
    signed = IDENTITY.sign(envelope(suggestion_id="sg-" + "3" * 16))
    assert signed.dot.seq == 1
    assert signed.kind == "field_set"
    assert signed.suggestion_id == "sg-" + "3" * 16
    with pytest.raises(ValidationError):
        signed.model_copy(update={"signature": "zz"}).model_validate(
            {"op": signed.op, "public_key": signed.public_key, "signature": "zz"}
        )


def test_a_decision_seals_a_bounded_list_of_well_formed_runs() -> None:
    replica_id = "a" * 32
    decide = SuggestionDecide(
        suggestion_id="sg-" + "1" * 16, accept=True, sealed=(f"{replica_id}.3-7",)
    )
    assert decide.sealed == (f"{replica_id}.3-7",)
    assert SuggestionDecide(suggestion_id="sg-" + "1" * 16, accept=False).sealed == ()

    with pytest.raises(ValidationError, match="sealed runs must look like"):
        SuggestionDecide(suggestion_id="sg-" + "1" * 16, accept=True, sealed=(f"{replica_id}.3",))
    with pytest.raises(ValidationError, match="may not run backwards"):
        SuggestionDecide(suggestion_id="sg-" + "1" * 16, accept=True, sealed=(f"{replica_id}.9-3",))
    with pytest.raises(ValidationError, match="at most"):
        SuggestionDecide(
            suggestion_id="sg-" + "1" * 16,
            accept=True,
            sealed=tuple(f"{replica_id}.{n}-{n}" for n in range(1, MAX_SEAL_RUNS + 2)),
        )
