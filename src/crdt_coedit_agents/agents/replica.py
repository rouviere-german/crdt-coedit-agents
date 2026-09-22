"""An agent replica: plan, apply as a suggestion, and wait for a human.

The flow, and why it is in this order:

1. read the **main** view — an agent never sees another agent's undecided suggestions;
2. ask the model for a typed plan (one budgeted call);
3. validate the whole plan against the least-privilege registry *before* creating any operation,
   so a plan is applied completely or not at all, and refuse it if it would sign more than
   ``max_ops_per_suggestion`` operations;
4. apply it locally under a fresh, unguessable suggestion id. The operations are real and signed,
   but they stay in the agent's outbox;
5. open an approval request bound to the suggestion's **digest** — a hash of exactly what this
   suggestion contributes — together with the text a reviewer will see.

Two gates follow, and they are different gates:

* :meth:`AgentReplica.publish` re-derives the digest from the current state and asks the approval
  gate to release it. An agent that appended more operations after a human looked at the
  suggestion produces a different digest, so the approval no longer applies and nothing is sent.
* Even once published, the operations are invisible in the main view until an **owner** writes an
  accept flag. That flag is an ordinary CRDT write, so it commutes with concurrent editing and
  needs no rebasing. This is the load-bearing gate; the first one keeps an unreviewed agent from
  filling the document with drafts in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from crdt_coedit_agents.agentic.approval import ApprovalGate, ApprovalRecord, ApprovalStore
from crdt_coedit_agents.agentic.audit import AuditLog
from crdt_coedit_agents.agentic.errors import ApprovalRequiredError
from crdt_coedit_agents.agentic.providers.base import ToolUseBlock
from crdt_coedit_agents.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    ValidatedCall,
)
from crdt_coedit_agents.agents.planner import EditPlan, PlanRejectedError, SuggestionPlanner
from crdt_coedit_agents.agents.tools import (
    AddTagArgs,
    InsertAfterArgs,
    RemoveItemArgs,
    SetFieldArgs,
)
from crdt_coedit_agents.crdt.fugue import UnknownAnchorError
from crdt_coedit_agents.ops.model import SignedOp
from crdt_coedit_agents.replica import Replica, new_suggestion_id

__all__ = [
    "MAX_OPS_PER_SUGGESTION",
    "AgentReplica",
    "PublishArgs",
    "SuggestionResult",
    "planned_operations",
]


PREVIEW_CHARS = 1_000
#: Default cap on the operations one suggestion may sign. The plan schema alone would allow 64
#: steps of up to 2 000 characters, i.e. about 128 000 operations from a single budgeted model
#: call; the budget ledger meters model steps, tokens and dollars, not operations, so the count is
#: bounded here, before the first operation is created. The default equals one maximal insert.
MAX_OPS_PER_SUGGESTION = 2_000


def planned_operations(args: object) -> int:
    """How many operations one validated tool call signs (an upper bound for ``remove_item``)."""
    match args:
        case InsertAfterArgs():
            return len(args.text)  # Replica.insert_text writes one element per character
        case RemoveItemArgs():
            return args.count
        case _:
            return 1


class PublishArgs(BaseModel):
    """What a human approves: one suggestion, its digest and the text it would add."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    suggestion_id: str
    summary: str = Field(max_length=280)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    steps: int = Field(ge=1)
    inserted_text: str = Field(default="", max_length=PREVIEW_CHARS)
    deleted_elements: int = Field(default=0, ge=0)


@dataclass(slots=True)
class SuggestionResult:
    """Everything one proposal produced."""

    suggestion_id: str
    plan: EditPlan
    ops: list[SignedOp] = field(default_factory=list)
    digest: str = ""
    approval_request_id: str = ""
    preview: str = ""

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "suggestion_id": self.suggestion_id,
            "summary": self.plan.summary,
            "steps": len(self.plan.steps),
            "ops": len(self.ops),
            "digest": self.digest,
            "approval_request_id": self.approval_request_id,
        }


def _publish_registry() -> ToolRegistry:
    """A one-tool registry whose only purpose is to hash a publish request for approval."""

    def _noop(args: PublishArgs) -> JsonValue:  # pragma: no cover - never executed
        return {"suggestion_id": args.suggestion_id}

    registry = ToolRegistry({Capability.WRITE_DATA})
    registry.register(
        ToolDefinition(
            name="publish_suggestion",
            description="Ask a human to accept one suggestion into the main document.",
            args_model=PublishArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=_noop,
        )
    )
    return registry


class AgentReplica:
    """A suggest-only replica driven by a planner."""

    def __init__(
        self,
        replica: Replica,
        planner: SuggestionPlanner,
        *,
        approvals: ApprovalStore,
        audit: AuditLog,
        principal: str | None = None,
        max_ops_per_suggestion: int = MAX_OPS_PER_SUGGESTION,
    ) -> None:
        if max_ops_per_suggestion < 1:
            raise ValueError("max_ops_per_suggestion must be at least 1")
        self._replica = replica
        self._planner = planner
        self._approvals = approvals
        self._audit = audit
        self._max_ops = max_ops_per_suggestion
        self._principal = principal or f"agent:{replica.replica_id[:12]}"
        self._registry = _publish_registry()
        self._gate = ApprovalGate(approvals, principal=self._principal)
        self._suggestions: dict[str, SuggestionResult] = {}

    @property
    def principal(self) -> str:
        return self._principal

    @property
    def replica(self) -> Replica:
        return self._replica

    @property
    def approvals(self) -> ApprovalStore:
        """The review queue this agent opens requests in."""
        return self._approvals

    def propose(self, instruction: str) -> SuggestionResult:
        """Produce one suggestion and open its approval request."""
        view = self._replica.view()
        self._audit.append(
            self._principal,
            "plan.requested",
            {"doc_id": self._replica.doc_id, "anchors": len(view.anchors), "chars": len(view.text)},
        )
        plan = self._planner.plan(view, instruction)
        calls = self._planner.validate(plan, view)
        self._audit.append(
            self._principal,
            "plan.validated",
            {
                "request_sha256": self._planner.last_request_sha256,
                "model": self._planner.last_model,
                "steps": len(plan.steps),
                "tools": [call.tool.name for call in calls],
            },
        )
        planned = sum(planned_operations(call.args) for call in calls)
        if planned > self._max_ops:
            self._audit.append(
                self._principal,
                "plan.rejected",
                {
                    "doc_id": self._replica.doc_id,
                    "reason": "too_many_operations",
                    "planned_ops": planned,
                    "limit": self._max_ops,
                },
            )
            raise PlanRejectedError(
                f"plan would sign {planned} operations, more than the {self._max_ops} allowed "
                "for one suggestion"
            )

        suggestion_id = new_suggestion_id()
        result = SuggestionResult(suggestion_id=suggestion_id, plan=plan)
        for call in calls:
            result.ops.extend(self._apply(call.args, suggestion_id))
        result.digest = self._replica.state.suggestion_digest(suggestion_id)
        result.preview = self._replica.view(suggestion_id).text
        self._suggestions[suggestion_id] = result

        try:
            self._gate.authorize(self._publish_call(suggestion_id, plan))
        except ApprovalRequiredError as exc:
            result.approval_request_id = exc.request_id
        self._audit.append(
            self._principal,
            "suggestion.proposed",
            {
                "doc_id": self._replica.doc_id,
                "suggestion_id": suggestion_id,
                "digest": result.digest,
                "ops": len(result.ops),
                "approval_request_id": result.approval_request_id,
            },
        )
        return result

    def _publish_call(self, suggestion_id: str, plan: EditPlan) -> ValidatedCall:
        """Build the publish call from the *current* state, so the digest is never stale."""
        state = self._replica.state
        block = ToolUseBlock(
            id="publish",
            name="publish_suggestion",
            arguments={
                "doc_id": self._replica.doc_id,
                "suggestion_id": suggestion_id,
                "summary": plan.summary,
                "digest": state.suggestion_digest(suggestion_id),
                "steps": len(plan.steps),
                "inserted_text": state.suggestion_text(suggestion_id)[:PREVIEW_CHARS],
                "deleted_elements": state.suggestion_deletions(suggestion_id),
            },
        )
        return self._registry.validate_call(block)

    def publish(self, suggestion_id: str) -> list[SignedOp]:
        """Release a suggestion's operations once a human approved *these* operations.

        Raises :class:`~crdt_coedit_agents.agentic.errors.ApprovalRequiredError` while the request
        is undecided, denied, or no longer matches what was approved.
        """
        result = self._suggestions.get(suggestion_id)
        if result is None:
            raise KeyError(f"unknown suggestion {suggestion_id}")
        call = self._publish_call(suggestion_id, result.plan)
        self._gate.authorize(call)
        self._audit.append(
            self._principal,
            "suggestion.published",
            {
                "doc_id": self._replica.doc_id,
                "suggestion_id": suggestion_id,
                "digest": call.canonical_args["digest"],
                "ops": len(result.ops),
            },
        )
        return list(result.ops)

    def suggestion(self, suggestion_id: str) -> SuggestionResult:
        return self._suggestions[suggestion_id]

    def _apply(self, args: object, suggestion_id: str) -> list[SignedOp]:
        """Turn one validated tool call into signed operations on this replica."""
        match args:
            case InsertAfterArgs():
                anchor = None if args.anchor_id == "START" else args.anchor_id
                return self._replica.insert_text(
                    args.text, after=anchor, suggestion_id=suggestion_id
                )
            case SetFieldArgs():
                return [self._replica.set_field(args.key, args.value, suggestion_id=suggestion_id)]
            case AddTagArgs():
                return [self._replica.add_tag(args.tag, suggestion_id=suggestion_id)]
            case RemoveItemArgs():
                try:
                    return self._replica.delete_run(
                        args.anchor_id, args.count, suggestion_id=suggestion_id
                    )
                except UnknownAnchorError as exc:
                    raise PlanRejectedError(
                        "remove_item addresses an element that is no longer visible"
                    ) from exc
        raise PlanRejectedError(f"unsupported tool arguments {type(args).__name__}")

    def pending_approval(self, request_id: str) -> ApprovalRecord:
        return self._approvals.get(request_id)
