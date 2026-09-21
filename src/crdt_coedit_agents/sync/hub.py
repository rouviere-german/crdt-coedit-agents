"""The document hub: validate, persist, apply and relay operations.

The hub is the only component that decides whether an operation enters a document. For every
operation it checks, in this order:

1. **signature and identity** — Ed25519 over the canonical envelope, with the replica id
   recomputed from the presented public key;
2. **authorization** — the connection's role and the replica id the token pins;
3. **clock** — the operation's hybrid-logical timestamp may not be further than
   :data:`~crdt_coedit_agents.crdt.hlc.MAX_DRIFT_MS` ahead of the server's wall clock, so no client
   can pin a last-writer-wins register (a document field, or a suggestion's accept flag) by dating
   its write in the far future;
4. **decided suggestions** — a suggestion that has already been accepted or rejected takes no
   further operations, so an agent cannot keep appending to an id a human has signed off;
5. **replay** — the unique ``(doc_id, replica_id, author_seq)`` constraint on the log;
6. **size** — the document may not grow past the configured limit.

Only then is the operation appended to the hash-chained log and applied to the server's own
replica of the document, and only operations that got that far are relayed to subscribers. The
order matters: nothing unverified is ever persisted, and nothing unpersisted is ever relayed.

Subscribers have bounded outbound queues. A consumer that cannot keep up is marked closed rather
than allowed to grow the queue, which keeps one slow client from turning into server memory
pressure.

The hub is transport-independent on purpose: the WebSocket server and the offline demo drive the
same object, so the demo exercises the real policy rather than a simplified copy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.core.logging import get_logger
from crdt_coedit_agents.crdt.document import DocumentState, SuggestionStatus
from crdt_coedit_agents.crdt.hlc import ClockDriftError, check_drift, wall_clock_ms
from crdt_coedit_agents.crdt.ids import VersionVector
from crdt_coedit_agents.metrics import SyncMetrics, create_sync_metrics
from crdt_coedit_agents.ops.apply import OpRejectedError, apply_op
from crdt_coedit_agents.ops.log import ChainVerification, OpLog, OpLogError
from crdt_coedit_agents.ops.model import SignedOp, SuggestionDecide
from crdt_coedit_agents.ops.signing import InvalidSignatureError, verify_op
from crdt_coedit_agents.sync.auth import AuthorizationError, Grant, authorize_op

__all__ = ["DocumentHub", "HubLimits", "SubmitResult", "Subscription", "TooManySubscribersError"]

_log = get_logger("crdt_coedit_agents.hub")
_APPLIED: Final = "applied"
_DUPLICATE: Final = "duplicate"


class TooManySubscribersError(RuntimeError):
    """A document already has the maximum number of live subscribers."""


@dataclass(frozen=True, slots=True)
class HubLimits:
    """Server-side bounds. Every one of them is enforced, not merely documented."""

    max_ops_per_push: int = 128
    max_document_bytes: int = 1_048_576
    max_catch_up_ops: int = 4_096
    subscriber_queue: int = 256
    max_subscribers_per_doc: int = 32


@dataclass(frozen=True, slots=True)
class SubmitResult:
    accepted: int
    duplicates: int
    rejected: dict[str, int]
    log_head: str
    applied: tuple[SignedOp, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.rejected


@dataclass(slots=True)
class Subscription:
    """One subscriber's bounded outbound queue.

    ``overflow`` is set when the queue is full. The session waits on it so a consumer that has
    stopped reading is disconnected instead of being buffered without limit.
    """

    doc_id: str
    replica_id: str
    queue: asyncio.Queue[tuple[SignedOp, ...]]
    overflow: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False
    dropped: int = 0

    def offer(self, ops: tuple[SignedOp, ...]) -> bool:
        """Enqueue operations; False (and a closed subscription) when the queue is full."""
        if self.closed:
            return False
        try:
            self.queue.put_nowait(ops)
        except asyncio.QueueFull:
            self.closed = True
            self.dropped += len(ops)
            self.overflow.set()
            return False
        return True


@dataclass(slots=True)
class _Document:
    state: DocumentState
    subscribers: list[Subscription] = field(default_factory=list)


class DocumentHub:
    """Validation, persistence and relay for every document on this server."""

    def __init__(
        self,
        log: OpLog,
        *,
        limits: HubLimits | None = None,
        metrics: SyncMetrics | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._log = log
        self._limits = limits if limits is not None else HubLimits()
        self._metrics = metrics if metrics is not None else create_sync_metrics()
        self._audit = audit
        self._documents: dict[str, _Document] = {}

    @property
    def limits(self) -> HubLimits:
        return self._limits

    @property
    def metrics(self) -> SyncMetrics:
        return self._metrics

    @property
    def op_log(self) -> OpLog:
        return self._log

    # ------------------------------------------------------------- documents

    def _document(self, doc_id: str) -> _Document:
        existing = self._documents.get(doc_id)
        if existing is not None:
            return existing
        state = DocumentState(doc_id=doc_id)
        for entry in self._log.entries(doc_id):
            verify_op(entry.signed)
            apply_op(state, entry.signed)
        document = _Document(state=state)
        self._documents[doc_id] = document
        self._refresh_gauges(doc_id, document)
        return document

    def state(self, doc_id: str) -> DocumentState:
        """The server's own replica of the document, rebuilt from the log on first access."""
        return self._document(doc_id).state

    def version_vector(self, doc_id: str) -> VersionVector:
        return self._log.version_vector(doc_id)

    def log_head(self, doc_id: str) -> str:
        return self._log.head(doc_id)[1]

    def catch_up(self, doc_id: str, peer: VersionVector) -> list[SignedOp]:
        """Everything ``peer`` is missing, oldest first, bounded by ``max_catch_up_ops``."""
        self._document(doc_id)
        return self._log.since(doc_id, peer, limit=self._limits.max_catch_up_ops)

    # ------------------------------------------------------------ submission

    def submit(self, grant: Grant, ops: Sequence[SignedOp]) -> SubmitResult:
        """Validate, persist, apply and queue for relay. Never partially persists an operation."""
        started = time.perf_counter()
        document = self._document(grant.doc_id)
        rejected: dict[str, int] = {}
        accepted: list[SignedOp] = []
        duplicates = 0

        if len(ops) > self._limits.max_ops_per_push:
            self._reject(rejected, "too_many_ops", len(ops))
            return SubmitResult(0, 0, rejected, self.log_head(grant.doc_id))

        # The size limit is checked once per batch rather than once per operation, because
        # measuring the document is linear in its size. A push can therefore overshoot the limit
        # by at most one batch, which is bounded by max_ops_per_push.
        full = document.state.size_bytes() >= self._limits.max_document_bytes

        for signed in ops:
            outcome = self._admit(grant, signed, document, full=full)
            if outcome == _DUPLICATE:
                duplicates += 1
            elif outcome == _APPLIED:
                accepted.append(signed)
            else:
                self._reject(rejected, outcome, 1)

        if accepted:
            self._relay(document, tuple(accepted))
        self._refresh_gauges(grant.doc_id, document)
        self._metrics.merge_latency_seconds.observe(time.perf_counter() - started)
        return SubmitResult(
            accepted=len(accepted),
            duplicates=duplicates,
            rejected=rejected,
            log_head=self.log_head(grant.doc_id),
            applied=tuple(accepted),
        )

    def _admit(  # noqa: PLR0911 - one early return per named rejection reason
        self, grant: Grant, signed: SignedOp, document: _Document, *, full: bool
    ) -> str:
        """Run one operation through the checks. Returns an outcome tag."""
        try:
            verify_op(signed)
        except InvalidSignatureError:
            self._audit_reject(grant, signed, "signature")
            return "signature"
        try:
            authorize_op(grant, signed)
        except AuthorizationError:
            self._audit_reject(grant, signed, "authorization")
            return "authorization"
        try:
            check_drift(signed.op.timestamp, wall_clock_ms())
        except ClockDriftError:
            self._audit_reject(grant, signed, "clock_drift")
            return "clock_drift"
        if self._is_decided(signed, document):
            self._audit_reject(grant, signed, "suggestion_decided")
            return "suggestion_decided"
        if full:
            self._audit_reject(grant, signed, "document_full")
            return "document_full"
        try:
            entry = self._log.append(signed)
        except OpLogError:
            self._audit_reject(grant, signed, "replay")
            return "replay"
        if entry is None:
            return _DUPLICATE
        try:
            apply_op(document.state, signed)
        except OpRejectedError:
            # Persisted but not applicable: the chain keeps the evidence, the state does not move.
            self._audit_reject(grant, signed, "not_applicable")
            return "not_applicable"
        scope = "main" if signed.op.suggestion_id is None else "suggestion"
        self._metrics.ops_applied_total.labels(signed.op.kind, scope).inc()
        if isinstance(signed.op.payload, SuggestionDecide):
            self._audit_decision(grant, signed, signed.op.payload)
        return _APPLIED

    def _is_decided(self, signed: SignedOp, document: _Document) -> bool:
        """True when this operation belongs to a suggestion that has already been decided.

        The seal on the decision already makes such an operation invisible everywhere; refusing it
        here keeps it out of the log as well, so the evidence a reviewer looked at stays the whole
        story. A re-send of an operation the log already holds is a duplicate, not a late addition,
        so it is let through to the duplicate path below.
        """
        suggestion = signed.op.suggestion_id
        if suggestion is None:
            return False
        if document.state.status(suggestion) == SuggestionStatus.PENDING:
            return False
        return not self._log.contains(signed.op.doc_id, signed.dot)

    def _reject(self, rejected: dict[str, int], reason: str, count: int) -> None:
        rejected[reason] = rejected.get(reason, 0) + count
        self._metrics.ops_rejected_total.labels(reason).inc(count)

    # ------------------------------------------------------------------ audit

    def _audit_reject(self, grant: Grant, signed: SignedOp, reason: str) -> None:
        _log.warning(
            "op_rejected",
            doc_id=grant.doc_id,
            principal=grant.principal,
            role=grant.role.value,
            kind=signed.op.kind,
            reason=reason,
        )
        if self._audit is not None:
            self._audit.append(
                f"replica:{grant.replica_id}",
                "op.rejected",
                {
                    "doc_id": grant.doc_id,
                    "kind": signed.op.kind,
                    "author": signed.op.replica_id,
                    "author_seq": signed.op.seq,
                    "reason": reason,
                },
            )

    def _audit_decision(self, grant: Grant, signed: SignedOp, payload: SuggestionDecide) -> None:
        if self._audit is None:
            return
        self._audit.append(
            f"principal:{grant.principal}",
            "suggestion.accepted" if payload.accept else "suggestion.rejected",
            {
                "doc_id": grant.doc_id,
                "suggestion_id": payload.suggestion_id,
                "decided_by": grant.replica_id,
                "author_seq": signed.op.seq,
                "reason": payload.reason,
            },
        )

    # ------------------------------------------------------------ subscribers

    def subscribe(self, doc_id: str, replica_id: str) -> Subscription:
        document = self._document(doc_id)
        self._prune(document)
        if len(document.subscribers) >= self._limits.max_subscribers_per_doc:
            raise TooManySubscribersError(
                f"document {doc_id} already has {self._limits.max_subscribers_per_doc} subscribers"
            )
        subscription = Subscription(
            doc_id=doc_id,
            replica_id=replica_id,
            queue=asyncio.Queue(maxsize=self._limits.subscriber_queue),
        )
        document.subscribers.append(subscription)
        self._metrics.ws_connections.inc()
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        document = self._documents.get(subscription.doc_id)
        if document is None or subscription not in document.subscribers:
            return
        document.subscribers.remove(subscription)
        subscription.closed = True
        self._metrics.ws_connections.dec()

    def subscriber_count(self, doc_id: str) -> int:
        document = self._documents.get(doc_id)
        return 0 if document is None else len(document.subscribers)

    def _prune(self, document: _Document) -> None:
        for subscription in [s for s in document.subscribers if s.closed]:
            document.subscribers.remove(subscription)
            self._metrics.ws_connections.dec()

    def _relay(self, document: _Document, ops: tuple[SignedOp, ...]) -> None:
        for subscription in list(document.subscribers):
            # The author already has these operations; sending them back would only cost bandwidth.
            outgoing = tuple(op for op in ops if op.op.replica_id != subscription.replica_id)
            if not outgoing:
                continue
            if not subscription.offer(outgoing):
                self._metrics.slow_consumers_total.inc()
                _log.warning(
                    "slow_consumer_disconnected",
                    doc_id=subscription.doc_id,
                    dropped=subscription.dropped,
                )

    def _refresh_gauges(self, doc_id: str, document: _Document) -> None:
        pending = len(document.state.pending_suggestions())
        self._metrics.pending_suggestions.labels(doc_id).set(pending)
        lag = max((subscription.queue.qsize() for subscription in document.subscribers), default=0)
        self._metrics.sync_lag_ops.labels(doc_id).set(lag)

    # ------------------------------------------------------------------ admin

    def verify_log(self, doc_id: str, *, expect_head: str | None = None) -> ChainVerification:
        return self._log.verify(doc_id, expect_head=expect_head)

    def documents(self) -> Iterable[str]:
        return self._log.documents()
