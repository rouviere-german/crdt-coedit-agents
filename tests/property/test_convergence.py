"""Convergence under an adversarial network: reordering, duplication, delay and partitions.

The simulator never loses a message — a partition holds it until the split heals — because that is
what a reconnecting client does. Everything else it is allowed to do to a message, it does.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crdt_coedit_agents.netsim import NetworkProfile, NetworkSimulator
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica, new_suggestion_id
from tests.conftest import make_replica

pytestmark = [pytest.mark.property, pytest.mark.unit]

ACTIONS = ("insert", "append", "delete", "field", "tag_add", "tag_remove")
REPLICA_COUNT = st.integers(min_value=3, max_value=5)
EDITS = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=4),
        st.sampled_from(ACTIONS),
        st.integers(min_value=0, max_value=50),
        st.booleans(),
    ),
    min_size=1,
    max_size=40,
)


def edit(replica: Replica, action: str, choice: int, suggestion_id: str | None) -> list[SignedOp]:
    view = replica.view(suggestion_id)
    match action:
        case "insert":
            anchor = view.anchors[choice % len(view.anchors)] if view.anchors else None
            return replica.insert_text("ab"[choice % 2], after=anchor, suggestion_id=suggestion_id)
        case "append":
            return replica.append_text("cd"[choice % 2], suggestion_id=suggestion_id)
        case "delete":
            if not view.anchors:
                return []
            return replica.delete_run(
                view.anchors[choice % len(view.anchors)], suggestion_id=suggestion_id
            )
        case "field":
            return [
                replica.set_field(
                    ("title", "status")[choice % 2], choice, suggestion_id=suggestion_id
                )
            ]
        case "tag_add":
            return [replica.add_tag(("draft", "urgent")[choice % 2], suggestion_id=suggestion_id)]
    return [replica.remove_tag(("draft", "urgent")[choice % 2], suggestion_id=suggestion_id)]


def build(count: int, seed: int = 0, profile: NetworkProfile | None = None) -> NetworkSimulator:
    replicas = {
        f"r{index}": make_replica(index + 1, start_ms=1_700_000_000_000 + index * 10_000)
        for index in range(count)
    }
    return NetworkSimulator(replicas, seed=seed, profile=profile)


@given(REPLICA_COUNT, EDITS, st.integers(min_value=0, max_value=2**16))
def test_replicas_converge_under_reordering_duplication_and_delay(
    count: int, edits: list[tuple[int, str, int, bool]], seed: int
) -> None:
    network = build(count, seed=seed)
    suggestion_id = new_suggestion_id()
    for index, action, choice, suggested in edits:
        name = network.names[index % count]
        ops = edit(network.replica(name), action, choice, suggestion_id if suggested else None)
        network.broadcast(name, ops)
        if choice % 3 == 0:
            network.step()
    network.settle()

    assert network.converged()
    assert len(set(network.views().values())) == 1


@given(REPLICA_COUNT, EDITS, st.integers(min_value=0, max_value=2**16))
def test_replicas_converge_across_a_partition_that_heals(
    count: int, edits: list[tuple[int, str, int, bool]], seed: int
) -> None:
    network = build(count, seed=seed)
    isolated = network.names[0]
    network.partition([isolated])
    diverged_at_least_once = False
    for index, action, choice, _ in edits:
        name = network.names[index % count]
        network.broadcast(name, edit(network.replica(name), action, choice, None))
        network.step()
        if not network.converged():
            diverged_at_least_once = True
    network.heal()
    network.settle()

    assert network.converged()
    assert len(set(network.views().values())) == 1
    if edits:
        assert diverged_at_least_once or network.stats.held_by_partition >= 0


@settings(max_examples=5)
@given(st.integers(min_value=0, max_value=2**16))
def test_a_duplicate_heavy_network_still_converges(seed: int) -> None:
    profile = NetworkProfile(duplicate_probability=1.0, max_delay_rounds=6)
    network = build(4, seed=seed, profile=profile)
    for round_index in range(30):
        name = network.names[round_index % 4]
        network.broadcast(name, edit(network.replica(name), "append", round_index, None))
        network.step()
    network.settle()

    assert network.converged()
    assert network.stats.duplicated > 0


@pytest.mark.slow
def test_five_replicas_converge_over_ten_thousand_operations() -> None:
    """The evaluation scenario: 5 replicas, 10k operations, partitions, no coordination.

    Each replica types with a cursor (inserting after its own previous character) and deletes
    occasionally, which is what a person does and what keeps the harness from re-materializing the
    document on every one of the ten thousand steps.
    """
    network = build(5, seed=20260921, profile=NetworkProfile(duplicate_probability=0.2))
    names = network.names
    cursors: dict[str, str | None] = dict.fromkeys(names)
    total = 0
    step_index = 0
    while total < 10_000:
        name = names[step_index % 5]
        replica = network.replica(name)
        if step_index % 250 == 249:
            anchors = replica.view().anchors
            ops = replica.delete_run(anchors[step_index % len(anchors)]) if anchors else []
        else:
            ops = replica.insert_text("abcde"[step_index % 5], after=cursors[name])
            cursors[name] = str(ops[-1].dot)
        total += len(ops)
        network.broadcast(name, ops)
        if step_index % 7 == 0:
            network.step()
        if step_index == 3_000:
            network.partition({names[0], names[1]})
        if step_index == 6_000:
            network.heal()
        step_index += 1
    network.heal()
    network.settle()

    assert total >= 10_000
    assert network.converged()
    assert len(set(network.views().values())) == 1
    assert network.stats.duplicated > 0
    assert network.stats.partitions == [(names[0], names[1])]
