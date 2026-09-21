"""The replica: local edits, delta synchronization, replay protection and offline work."""

from __future__ import annotations

import pytest

from crdt_coedit_agents.crdt.fugue import UnknownAnchorError
from crdt_coedit_agents.crdt.hlc import MAX_DRIFT_MS, Timestamp
from crdt_coedit_agents.crdt.ids import Dot, VersionVector
from crdt_coedit_agents.ops.apply import OpRejectedError, apply_op
from crdt_coedit_agents.ops.model import Delta, FieldSet, SignedOp, TextInsert, UnsignedOp
from crdt_coedit_agents.replica import ReplayError, new_suggestion_id
from tests.conftest import make_replica, seeded_identity

pytestmark = pytest.mark.unit

SUGGESTION = "sg-" + "1" * 16


def test_local_edits_are_applied_immediately_and_numbered_contiguously() -> None:
    replica = make_replica(1)
    ops = replica.insert_text("hello")
    assert [op.op.seq for op in ops] == [1, 2, 3, 4, 5]
    assert replica.view().text == "hello"
    assert replica.version_vector.get(replica.replica_id) == 5
    assert replica.op_count == 5
    assert replica.doc_id == "test-doc"
    assert replica.insert_text("") == []


def test_append_and_insert_target_the_right_positions() -> None:
    replica = make_replica(1)
    replica.insert_text("bcd")
    replica.insert_text("a", after=None)
    replica.append_text("e")
    assert replica.view().text == "abcde"
    anchors = replica.view().anchors
    replica.insert_text("X", after=anchors[2])
    assert replica.view().text == "abcXde"


def test_deleting_a_run_removes_consecutive_visible_elements() -> None:
    replica = make_replica(1)
    replica.insert_text("abcdef")
    anchors = replica.view().anchors
    replica.delete_run(anchors[1], 3)
    assert replica.view().text == "aef"
    with pytest.raises(UnknownAnchorError, match="not visible"):
        replica.delete_run(anchors[1])
    with pytest.raises(ValueError, match="count must be"):
        replica.delete_run(replica.view().anchors[0], 0)


def test_tags_use_observed_remove_semantics() -> None:
    first = make_replica(1)
    second = make_replica(2, start_ms=1_700_000_010_000)
    first.add_tag("draft")
    second.receive(list(first.ops()))

    removals = second.remove_tag("draft")
    concurrent = first.add_tag("draft")
    first.receive([removals])
    second.receive([concurrent])

    assert first.view().tags == ["draft"]
    assert first.state_hash() == second.state_hash()


def test_field_writes_are_last_writer_wins_across_replicas() -> None:
    first = make_replica(1)
    second = make_replica(2, start_ms=1_700_000_010_000)
    first.set_field("title", "from-first")
    second.set_field("title", "from-second")
    first.receive(list(second.ops()))
    second.receive(list(first.ops()))
    assert first.view().fields == second.view().fields
    assert first.state_hash() == second.state_hash()


def test_a_suggestion_is_invisible_until_it_is_accepted() -> None:
    replica = make_replica(1)
    replica.insert_text("hi")
    suggestion_id = new_suggestion_id()
    replica.insert_text("!", after=replica.view().anchors[-1], suggestion_id=suggestion_id)
    assert replica.view().text == "hi"
    assert replica.view(suggestion_id).text == "hi!"
    replica.decide(suggestion_id, accept=True, reason="looks good")
    assert replica.view().text == "hi!"


def test_suggestion_ids_are_unguessable_and_unique() -> None:
    ids = {new_suggestion_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(item.startswith("sg-") and len(item) == 19 for item in ids)


def test_receiving_the_same_operation_twice_is_a_duplicate_not_an_error() -> None:
    source = make_replica(1)
    source.insert_text("ab")
    target = make_replica(2, start_ms=1_700_000_010_000)
    first = target.receive(list(source.ops()))
    second = target.receive(list(source.ops()))
    assert (first.received, first.applied, first.duplicates) == (2, 2, 0)
    assert (second.received, second.applied, second.duplicates) == (2, 0, 2)
    assert target.view().text == "ab"


def test_a_reused_sequence_number_with_different_content_is_a_replay() -> None:
    source = make_replica(1)
    source.insert_text("a")
    identity = seeded_identity(1)
    forged = identity.sign(
        UnsignedOp(
            doc_id=source.doc_id,
            replica_id=identity.replica_id,
            seq=1,
            ts=Timestamp(9_000, 0, identity.replica_id).encode(),
            payload=TextInsert(value="z"),
        )
    )
    target = make_replica(2, start_ms=1_700_000_010_000)
    target.receive(list(source.ops()))
    with pytest.raises(ReplayError, match="already used"):
        target.receive([forged])


def test_operations_for_another_document_are_refused() -> None:
    source = make_replica(1, "other-doc")
    source.insert_text("a")
    target = make_replica(2)
    with pytest.raises(OpRejectedError, match="targets document"):
        target.receive(list(source.ops()))
    with pytest.raises(OpRejectedError, match="targets document"):
        target.receive_delta(Delta(doc_id="other-doc", ops=tuple(source.ops())))


def test_a_forged_signature_never_reaches_the_state() -> None:
    source = make_replica(1)
    source.insert_text("a")
    signed = next(source.ops())
    forged = signed.model_copy(
        update={"op": signed.op.model_copy(update={"payload": TextInsert(value="z")})}
    )
    target = make_replica(2, start_ms=1_700_000_010_000)
    with pytest.raises(ValueError, match="does not verify"):
        target.receive([forged])
    assert target.view().text == ""


def test_deltas_carry_only_what_a_peer_is_missing_and_stay_bounded() -> None:
    source = make_replica(1)
    source.insert_text("a" * 40)
    empty = VersionVector()
    assert len(source.missing_for(empty)) == 40
    assert source.missing_for(source.version_vector) == []

    deltas = source.deltas_for(empty, batch=16)
    assert [len(delta.ops) for delta in deltas] == [16, 16, 8]
    assert all(delta.doc_id == source.doc_id for delta in deltas)
    assert len(source.full_state_ops()) == 40
    with pytest.raises(ValueError, match="batch must be"):
        source.deltas_for(empty, batch=0)


def test_operations_can_arrive_in_any_order_and_still_converge() -> None:
    source = make_replica(1)
    source.insert_text("hello world")
    ops = list(source.ops())

    forward = make_replica(2, start_ms=1_700_000_010_000)
    backward = make_replica(3, start_ms=1_700_000_020_000)
    forward.receive(ops)
    backward.receive(list(reversed(ops)))
    assert forward.state_hash() == backward.state_hash() == source.state_hash()
    assert backward.view().text == "hello world"


def test_a_replica_resuming_from_a_state_continues_its_own_sequence() -> None:
    first = make_replica(1)
    first.insert_text("abc")
    resumed = make_replica(1)
    resumed.receive(list(first.ops()))
    new_ops = resumed.append_text("d")
    assert new_ops[0].op.seq == 4
    assert resumed.view().text == "abcd"


def test_applying_an_operation_twice_reports_no_change() -> None:
    replica = make_replica(1)
    signed = replica.set_field("title", "x")
    assert apply_op(replica.state, signed) is False
    assert replica.has(signed.dot)
    assert not replica.has(Dot(replica_id=replica.replica_id, seq=99))


def test_an_unsupported_document_for_apply_is_reported() -> None:
    replica = make_replica(1)
    identity = seeded_identity(1)
    signed = identity.sign(
        UnsignedOp(
            doc_id="elsewhere",
            replica_id=identity.replica_id,
            seq=1,
            ts=Timestamp(1_000, 0, identity.replica_id).encode(),
            payload=FieldSet(key="title", value="x"),
        )
    )
    with pytest.raises(OpRejectedError, match="not 'test-doc'"):
        apply_op(replica.state, signed)


def far_future_op(index: int, doc_id: str, *, wall_ms: int, seq: int = 1) -> SignedOp:
    """A properly signed operation whose hybrid-logical timestamp is dated ``wall_ms``."""
    identity = seeded_identity(index)
    return identity.sign(
        UnsignedOp(
            doc_id=doc_id,
            replica_id=identity.replica_id,
            seq=seq,
            ts=Timestamp(wall_ms, 0, identity.replica_id).encode(),
            payload=FieldSet(key="title", value="PWNED"),
        )
    )


def test_an_operation_dated_far_in_the_future_is_refused_on_receive() -> None:
    target = make_replica(1)
    evil = far_future_op(2, target.doc_id, wall_ms=999_999_999_999_999)

    report = target.receive([evil])

    assert (report.received, report.applied, report.rejected) == (1, 0, 1)
    assert target.view().fields == {}
    assert target.op_count == 0
    # And the honest writer still wins afterwards, which is the point of the bound.
    target.set_field("title", "honest")
    assert target.view().fields == {"title": "honest"}


def test_an_operation_just_inside_the_drift_bound_is_still_accepted() -> None:
    target = make_replica(1)
    inside = far_future_op(2, target.doc_id, wall_ms=1_700_000_000_001 + MAX_DRIFT_MS)

    report = target.receive([inside])

    assert (report.applied, report.rejected) == (1, 0)
    assert target.view().fields == {"title": "PWNED"}


def test_receiving_a_remote_timestamp_pulls_the_local_clock_past_it() -> None:
    """The clock is hybrid *logical*: a write that follows a remote one also sorts after it."""
    ahead = make_replica(2, start_ms=1_700_000_030_000)
    behind = make_replica(1)
    remote = ahead.set_field("status", "from-ahead")

    behind.receive([remote])
    local = behind.set_field("status", "from-behind")

    assert local.op.timestamp > remote.op.timestamp
    assert behind.view().fields == {"status": "from-behind"}
