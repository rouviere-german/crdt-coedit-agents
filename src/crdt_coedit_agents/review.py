"""The human side of the review workflow: two gates, and what each one is for.

**Gate 1 — may this draft be published at all?** An agent that has planned a suggestion opens an
approval request carrying the exact text it would add, how many elements it would delete, and a
digest of all of it. :func:`approve_publish` records a human decision in the approval store, which
refuses self-approval, expired requests and requests that are not pending. Only then does
:meth:`~crdt_coedit_agents.agents.replica.AgentReplica.publish` release the operations, and only
while the digest still matches — an agent that kept editing after the human looked produces a
different digest, so the approval no longer applies.

**Gate 2 — does this draft become part of the document?** Published suggestion operations are
replicated but invisible: they carry a suggestion id and no accept flag. :func:`decide_in_document`
re-derives the digest from the deciding replica's *own* state, refuses to act if it changed, and
then writes an ordinary signed last-writer-wins flag. Because it is an ordinary write, the decision
commutes with everything else happening in the document; nothing is rebased and nothing is
re-signed.

Only the ``owner`` role may write that flag, which the sync server enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

from crdt_coedit_agents.agentic.approval import ApprovalRecord, ApprovalStatus, ApprovalStore
from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica

__all__ = [
    "DigestMismatchError",
    "ReviewItem",
    "approve_publish",
    "decide_in_document",
    "deny_publish",
    "pending_reviews",
]

PUBLISH_TOOL = "publish_suggestion"


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class DigestMismatchError(RuntimeError):
    """The suggestion changed after it was reviewed; no flag is written."""


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One pending suggestion, as a reviewer sees it."""

    request_id: str
    suggestion_id: str
    doc_id: str
    summary: str
    inserted_text: str
    deleted_elements: int
    digest: str
    requested_by: str

    @classmethod
    def from_record(cls, record: ApprovalRecord) -> ReviewItem:
        args = record.args
        return cls(
            request_id=record.id,
            suggestion_id=str(args.get("suggestion_id", "")),
            doc_id=str(args.get("doc_id", "")),
            summary=str(args.get("summary", "")),
            inserted_text=str(args.get("inserted_text", "")),
            deleted_elements=_as_int(args.get("deleted_elements")),
            digest=str(args.get("digest", "")),
            requested_by=record.requested_by,
        )


def pending_reviews(store: ApprovalStore) -> list[ReviewItem]:
    """Every undecided publish request, oldest first."""
    return [
        ReviewItem.from_record(record)
        for record in sorted(store.pending(), key=lambda record: record.requested_at)
        if record.tool == PUBLISH_TOOL
    ]


def _review(store: ApprovalStore, request_id: str) -> ApprovalRecord:
    record = store.get(request_id)
    if record.tool != PUBLISH_TOOL:
        raise ValueError(f"request {request_id} is not a suggestion review")
    if record.status is not ApprovalStatus.PENDING:
        raise ValueError(f"request {request_id} is {record.status.value}, not pending")
    return record


def approve_publish(
    store: ApprovalStore,
    audit: AuditLog,
    *,
    request_id: str,
    approver: str,
    reason: str | None = None,
) -> ReviewItem:
    """Gate 1: let the agent publish exactly the operations in this request."""
    item = ReviewItem.from_record(_review(store, request_id))
    store.decide(request_id, approver=approver, approve=True, reason=reason)
    audit.append(
        approver,
        "publish.approved",
        {
            "doc_id": item.doc_id,
            "suggestion_id": item.suggestion_id,
            "request_id": request_id,
            "digest": item.digest,
            "requested_by": item.requested_by,
            "reason": reason,
        },
    )
    return item


def deny_publish(
    store: ApprovalStore,
    audit: AuditLog,
    *,
    request_id: str,
    approver: str,
    reason: str | None = None,
) -> ReviewItem:
    """Gate 1, refused: the draft is never published, so the document never sees it."""
    item = ReviewItem.from_record(_review(store, request_id))
    store.decide(request_id, approver=approver, approve=False, reason=reason)
    audit.append(
        approver,
        "publish.denied",
        {
            "doc_id": item.doc_id,
            "suggestion_id": item.suggestion_id,
            "request_id": request_id,
            "requested_by": item.requested_by,
            "reason": reason,
        },
    )
    return item


def decide_in_document(
    owner: Replica,
    audit: AuditLog,
    item: ReviewItem,
    *,
    accept: bool,
    approver: str,
    reason: str | None = None,
) -> SignedOp:
    """Gate 2: write the accept or reject flag, but only for the reviewed operations.

    The digest is recomputed from ``owner``'s own replica, so a suggestion that grew between the
    review and the decision is refused instead of silently accepted.
    """
    current = owner.state.suggestion_digest(item.suggestion_id)
    if current != item.digest:
        raise DigestMismatchError(
            f"suggestion {item.suggestion_id} changed after it was reviewed; nothing was decided"
        )
    operation = owner.decide(item.suggestion_id, accept=accept, reason=reason)
    audit.append(
        approver,
        "suggestion.accepted" if accept else "suggestion.rejected",
        {
            "doc_id": item.doc_id,
            "suggestion_id": item.suggestion_id,
            "request_id": item.request_id,
            "digest": item.digest,
            "decided_by": owner.replica_id,
            "reason": reason,
        },
    )
    return operation
