"""Prometheus metrics for the sync service, on a per-app registry (no global state)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["SyncMetrics", "create_sync_metrics"]


@dataclass(frozen=True, slots=True)
class SyncMetrics:
    registry: CollectorRegistry
    ops_applied_total: Counter
    ops_rejected_total: Counter
    sync_lag_ops: Gauge
    ws_connections: Gauge
    merge_latency_seconds: Histogram
    pending_suggestions: Gauge
    slow_consumers_total: Counter


def create_sync_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "crdt"
) -> SyncMetrics:
    """Create the sync metric families. Labels are bounded: kind, role and reason only."""
    reg = registry if registry is not None else CollectorRegistry()
    return SyncMetrics(
        registry=reg,
        ops_applied_total=Counter(
            "ops_applied_total",
            "Operations accepted by the hub and applied to the server state.",
            ["kind", "scope"],
            namespace=namespace,
            registry=reg,
        ),
        ops_rejected_total=Counter(
            "ops_rejected_total",
            "Operations refused by the hub, by reason.",
            ["reason"],
            namespace=namespace,
            registry=reg,
        ),
        sync_lag_ops=Gauge(
            "sync_lag_ops",
            "Operations a connected replica has not acknowledged yet.",
            ["doc_id"],
            namespace=namespace,
            registry=reg,
        ),
        ws_connections=Gauge(
            "ws_connections",
            "Open synchronization WebSocket connections.",
            namespace=namespace,
            registry=reg,
        ),
        merge_latency_seconds=Histogram(
            "merge_latency_seconds",
            "Time to verify, authorize, log and apply one batch of operations.",
            namespace=namespace,
            registry=reg,
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
        ),
        pending_suggestions=Gauge(
            "pending_suggestions",
            "Suggestions present in a document that no owner has decided yet.",
            ["doc_id"],
            namespace=namespace,
            registry=reg,
        ),
        slow_consumers_total=Counter(
            "slow_consumers_total",
            "Subscribers disconnected because their outbound queue was full.",
            namespace=namespace,
            registry=reg,
        ),
    )
