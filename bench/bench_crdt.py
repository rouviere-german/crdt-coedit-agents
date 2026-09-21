"""CRDT benchmarks: the numbers the README quotes, measured on CPU in a couple of minutes.

Usage:
  uv run python bench/bench_crdt.py           # run, write bench/results/crdt.json
  uv run python bench/bench_crdt.py --check    # also enforce bench/thresholds_crdt.json

What is measured and why:

``local_insert``
    the whole local write path: resolve the position, sign the operation, apply it. This is what a
    keystroke costs.
``remote_apply``
    receiving someone else's operation: verify the Ed25519 signature, replay-check it, apply it.
    Signature verification dominates, which is the honest cost of not trusting the server.
``merge_only``
    applying an already-verified operation, i.e. the pure CRDT merge.
``materialize``
    turning the tree into text. This is the O(n) step the design trades for O(1) inserts, so it is
    measured at the size where it starts to matter.
``state_memory_100k``
    peak heap for a 100 000 element document, measured with ``tracemalloc``. It covers the CRDT
    state only; the signed operation history is measured separately because a replica can drop it
    after a snapshot.
``delta_vs_full_state``
    the point of delta-state synchronization: bytes on the wire for a peer that is a few
    operations behind, against a full-state transfer.

Every threshold is an upper bound, including the ones derived from throughput, so a regression is
always "a number got bigger".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from bench.bench_baseline import check_thresholds, machine, percentile, report, timed
from crdt_coedit_agents.crdt.document import DocumentState
from crdt_coedit_agents.crdt.fugue import FugueTree
from crdt_coedit_agents.crdt.ids import Dot, VersionVector
from crdt_coedit_agents.ops.apply import apply_op
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.ops.signing import ReplicaIdentity, verify_op
from crdt_coedit_agents.replica import Replica

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "crdt.json"
THRESHOLDS = ROOT / "bench" / "thresholds_crdt.json"

DOC = "bench-doc"
TREE_ELEMENTS = 100_000
SIGNED_OPS = 4_000
MATERIALIZE_ELEMENTS = 20_000


def identity(index: int) -> ReplicaIdentity:
    return ReplicaIdentity.from_seed(bytes([index]) * 32)


def typed_replica(index: int, count: int) -> tuple[Replica, list[SignedOp]]:
    """A replica that typed ``count`` characters, and the operations it produced."""
    replica = Replica(identity(index), DOC)
    ops: list[SignedOp] = []
    cursor: str | None = None
    for position in range(count):
        created = replica.insert_text("abcde"[position % 5], after=cursor)
        cursor = str(created[0].dot)
        ops.extend(created)
    return replica, ops


def build_tree(count: int) -> FugueTree:
    """A tree of ``count`` elements, built without signing (the CRDT state on its own)."""
    tree = FugueTree()
    replica_id = "a" * 32
    cursor: Dot | None = None
    for seq in range(1, count + 1):
        cursor = tree.insert_after(Dot(replica_id=replica_id, seq=seq), "x", after=cursor).dot
    return tree


def bench_local_insert() -> dict[str, float]:
    replica = Replica(identity(1), DOC)
    cursor: list[str | None] = [None]

    def step() -> None:
        created = replica.insert_text("x", after=cursor[0])
        cursor[0] = str(created[0].dot)

    return timed(step, runs=2_000, warmup=50)


def bench_remote_apply(ops: list[SignedOp]) -> dict[str, float]:
    """Verify + apply, one operation at a time, into a fresh state."""
    state = DocumentState(doc_id=DOC)
    stream = iter(ops)

    def step() -> None:
        signed = next(stream)
        verify_op(signed)
        apply_op(state, signed)

    runs = min(len(ops) - 60, 2_000)
    return timed(step, runs=runs, warmup=50)


def bench_merge_only(ops: list[SignedOp]) -> dict[str, float]:
    state = DocumentState(doc_id=DOC)
    stream = iter(ops)

    def step() -> None:
        apply_op(state, next(stream))

    runs = min(len(ops) - 60, 2_000)
    return timed(step, runs=runs, warmup=50)


def bench_materialize(tree: FugueTree) -> dict[str, float]:
    def step() -> None:
        # Measure a cold traversal, not the cache.
        tree.invalidate_order()
        tree.materialize()

    return timed(step, runs=20, warmup=2)


def bench_state_memory() -> dict[str, float]:
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tree = build_tree(TREE_ELEMENTS)
        current, peak = tracemalloc.get_traced_memory()
        elements = len(tree)
    finally:
        tracemalloc.stop()
    return {
        "elements": float(elements),
        "resident_kib": round((current - before) / 1024, 1),
        "peak_kib": round((peak - before) / 1024, 1),
        "bytes_per_element": round((current - before) / max(elements, 1), 1),
    }


def bench_delta_vs_full_state(replica: Replica, ops: list[SignedOp]) -> dict[str, float]:
    """Bytes on the wire for a peer that is ``behind`` operations out of date.

    A full-state transfer does not fit in one frame, so it is the sum of the bounded batches a
    replica would actually send — which is what delta synchronization avoids.
    """
    full = sum(delta.encoded_size() for delta in replica.deltas_for(VersionVector()))
    sizes: dict[str, float] = {"full_state_bytes": float(full), "total_ops": float(len(ops))}
    for behind in (1, 10, 100):
        vector = VersionVector({replica.replica_id: max(len(ops) - behind, 0)})
        delta = sum(item.encoded_size() for item in replica.deltas_for(vector))
        sizes[f"delta_bytes_behind_{behind}"] = float(delta)
        sizes[f"delta_fraction_behind_{behind}"] = round(delta / full, 6)
    return sizes


def bench_convergence() -> dict[str, float]:
    """Five replicas, every operation delivered in a different order to each of them."""
    import random

    rng = random.Random(20260921)
    replicas = [Replica(identity(index), DOC) for index in range(1, 6)]
    produced: list[SignedOp] = []
    for round_index in range(400):
        author = replicas[round_index % len(replicas)]
        anchors = author.view().anchors
        anchor = anchors[rng.randrange(len(anchors))] if anchors else None
        produced.extend(author.insert_text("abcde"[round_index % 5], after=anchor))

    durations: list[float] = []
    for target in replicas:
        shuffled = produced[:]
        rng.shuffle(shuffled)
        start = time.perf_counter_ns()
        target.receive(shuffled)
        durations.append((time.perf_counter_ns() - start) / 1_000.0)

    hashes = {replica.state_hash() for replica in replicas}
    views = {replica.view().text for replica in replicas}
    return {
        "replicas": float(len(replicas)),
        "operations": float(len(produced)),
        "distinct_state_hashes": float(len(hashes)),
        "distinct_views": float(len(views)),
        "converged_fraction": 1.0 if len(hashes) == 1 and len(views) == 1 else 0.0,
        "p50_us": round(statistics.median(durations), 1),
        "p95_us": round(percentile(durations, 0.95), 1),
    }


def run() -> dict[str, dict[str, float]]:
    replica, ops = typed_replica(1, SIGNED_OPS)
    tree = build_tree(MATERIALIZE_ELEMENTS)
    return {
        "local_insert": bench_local_insert(),
        "remote_apply_verified": bench_remote_apply(ops),
        "merge_only": bench_merge_only(ops),
        f"materialize_{MATERIALIZE_ELEMENTS}": bench_materialize(tree),
        "state_memory_100k": bench_state_memory(),
        "delta_vs_full_state": bench_delta_vs_full_state(replica, ops),
        "convergence_5_replicas": bench_convergence(),
    }


def print_report(results: dict[str, dict[str, float]]) -> None:
    latency = {
        name: stats for name, stats in results.items() if "p50_us" in stats and "runs" in stats
    }
    report("CRDT latency benchmarks", latency)
    print("\n| measurement | value |")
    print("|---|---:|")
    for name, stats in results.items():
        if name in latency:
            print(f"| {name} ops/s | {1_000_000 / max(stats['p50_us'], 1e-9):,.0f} |")
            continue
        for metric, value in stats.items():
            print(f"| {name}.{metric} | {value:,.6g} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CRDT benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds_crdt.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    print_report(results)
    if args.check:
        problems = check_thresholds(results, THRESHOLDS)
        if results["convergence_5_replicas"]["converged_fraction"] < 1.0:
            problems.append("convergence_5_replicas: replicas did not converge")
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
