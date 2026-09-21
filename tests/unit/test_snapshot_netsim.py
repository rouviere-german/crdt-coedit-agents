"""Snapshots (write-then-rename, hash-verified) and the network simulator itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crdt_coedit_agents.netsim import NetworkProfile, NetworkSimulator
from crdt_coedit_agents.replica import Replica
from crdt_coedit_agents.snapshot import (
    Snapshot,
    SnapshotError,
    load_snapshot,
    snapshot_summary,
    write_snapshot,
)
from tests.conftest import make_replica

pytestmark = pytest.mark.unit


def populated() -> Replica:
    replica = make_replica(1)
    replica.insert_text("hello")
    replica.set_field("title", "Notes")
    replica.add_tag("draft")
    return replica


# ------------------------------------------------------------------ snapshots


def test_a_snapshot_round_trips_and_carries_the_version_vector(tmp_path: Path) -> None:
    replica = populated()
    path = tmp_path / "snap.json"
    written = write_snapshot(path, replica.state, replica.version_vector)
    assert written.state_hash == replica.state_hash()

    state, vector = load_snapshot(path)
    assert state.state_hash() == replica.state_hash()
    assert vector == replica.version_vector
    assert state.view().text == "hello"


def test_a_snapshot_restores_into_a_working_replica(tmp_path: Path) -> None:
    replica = populated()
    path = tmp_path / "snap.json"
    write_snapshot(path, replica.state, replica.version_vector)
    state, _ = load_snapshot(path)

    resumed = make_replica(2, start_ms=1_700_000_500_000)
    resumed.state.text = state.text
    resumed.state.fields = state.fields
    resumed.state.tags = state.tags
    resumed.append_text("!")
    assert resumed.view().text == "hello!"


def test_a_corrupted_snapshot_is_refused(tmp_path: Path) -> None:
    replica = populated()
    path = tmp_path / "snap.json"
    write_snapshot(path, replica.state, replica.version_vector)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["text_nodes"][0]["value"] = "Z"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotError, match="does not match its recorded state hash"):
        load_snapshot(path)


def test_a_snapshot_whose_body_is_malformed_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "snap.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not a valid snapshot file"):
        load_snapshot(path)
    with pytest.raises(SnapshotError, match="could not read"):
        load_snapshot(tmp_path / "missing.json")
    with pytest.raises(SnapshotError, match="could not read"):
        snapshot_summary(tmp_path / "missing.json")

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not an object"):
        snapshot_summary(path)

    broken = Snapshot(
        doc_id="test-doc",
        state_hash="0" * 64,
        version_vector={},
        state={"v": 1, "doc_id": "test-doc"},
    )
    with pytest.raises(SnapshotError, match="malformed"):
        broken.restore()


def test_the_summary_reads_the_header_without_rebuilding_the_state(tmp_path: Path) -> None:
    replica = populated()
    path = tmp_path / "snap.json"
    written = write_snapshot(path, replica.state, replica.version_vector)
    summary = snapshot_summary(path)
    assert summary["doc_id"] == "test-doc"
    assert summary["state_hash"] == written.state_hash


def test_writing_a_snapshot_never_leaves_a_partial_file(tmp_path: Path) -> None:
    replica = populated()
    path = tmp_path / "snap.json"
    write_snapshot(path, replica.state, replica.version_vector)
    first = path.read_bytes()
    replica.append_text("!")
    write_snapshot(path, replica.state, replica.version_vector)
    assert path.read_bytes() != first
    assert list(tmp_path.glob("*.tmp")) == []
    assert load_snapshot(path)[0].view().text == "hello!"


# ------------------------------------------------------------------- network


def build(
    count: int = 3, *, seed: int = 0, profile: NetworkProfile | None = None
) -> NetworkSimulator:
    replicas = {
        f"r{index}": make_replica(index + 1, start_ms=1_700_000_000_000 + index * 1_000_000)
        for index in range(count)
    }
    return NetworkSimulator(replicas, seed=seed, profile=profile)


def test_the_simulator_delivers_to_every_other_replica() -> None:
    network = build(profile=NetworkProfile(duplicate_probability=0.0, max_delay_rounds=0))
    ops = network.replica("r0").insert_text("hi")
    network.broadcast("r0", ops)
    assert network.inflight == 2  # one message per peer, carrying the whole batch
    network.settle()
    assert network.converged()
    assert network.views()["r1"] == "hi"


def test_delays_duplicates_and_reordering_are_applied() -> None:
    network = build(profile=NetworkProfile(duplicate_probability=1.0, max_delay_rounds=3))
    network.broadcast("r0", network.replica("r0").insert_text("abc"))
    assert network.stats.duplicated == 2
    network.settle()
    assert network.converged()
    assert network.stats.delivered == network.stats.sent


def test_a_partition_holds_messages_until_it_heals() -> None:
    network = build(profile=NetworkProfile(duplicate_probability=0.0, max_delay_rounds=0))
    network.partition({"r0"})
    network.broadcast("r0", network.replica("r0").insert_text("secret"))
    # Nothing is deliverable across the cut, so settle() has no work and the messages wait.
    assert network.settle() == 0
    assert network.step() == 0
    assert not network.converged()
    assert network.stats.held_by_partition > 0
    assert network.inflight == 2

    network.heal()
    network.settle()
    assert network.converged()
    assert network.stats.partitions == [("r0",)]


def test_the_simulator_refuses_unknown_replicas_and_tiny_networks() -> None:
    network = build()
    with pytest.raises(KeyError, match="unknown replicas"):
        network.partition({"nope"})
    with pytest.raises(KeyError, match="unknown replica"):
        network.broadcast("nope", [])
    with pytest.raises(ValueError, match="at least two replicas"):
        NetworkSimulator({"only": make_replica(1)})
    with pytest.raises(ValueError, match="probability"):
        NetworkProfile(duplicate_probability=2.0)
    with pytest.raises(ValueError, match="non-negative"):
        NetworkProfile(max_delay_rounds=-1)
    assert network.names == ("r0", "r1", "r2")
    network.broadcast("r0", [])
    assert network.inflight == 0


def test_settle_gives_up_on_a_message_that_can_never_arrive() -> None:
    network = build(profile=NetworkProfile(duplicate_probability=0.0, max_delay_rounds=0))
    network.broadcast("r0", network.replica("r0").insert_text("x"))
    with pytest.raises(RuntimeError, match="did not settle"):
        network.settle(max_rounds=0)
