"""Turning a document and an instruction into a typed, anchor-addressed edit plan.

The model sees three things: a trusted system prompt, the operator's instruction, and the document
wrapped in an ``<untrusted-data>`` envelope. The envelope is a hint, not the boundary — the real
boundary is that the plan schema has no step that could accept a suggestion, change a role or reach
the network, and that the plan is validated against the tool registry before a single operation is
created.

The plan is requested as a structured output (a JSON Schema on ``output_config.format`` for the
Anthropic provider), so a well-formed plan does not depend on the model formatting JSON by hand.
It is still validated locally: a plan that references an anchor the document does not contain, or
that exceeds the step limit, is rejected rather than partially applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from crdt_coedit_agents.agentic.budget import BudgetLedger
from crdt_coedit_agents.agentic.envelope import UNTRUSTED_DATA_POLICY, wrap_user_content
from crdt_coedit_agents.agentic.errors import ToolArgumentError, ToolNotAllowedError
from crdt_coedit_agents.agentic.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    TextBlock,
    ToolUseBlock,
)
from crdt_coedit_agents.agentic.tools import ValidatedCall
from crdt_coedit_agents.agents.tools import (
    MAX_INSERT_CHARS,
    EditToolArgs,
    build_edit_registry,
)
from crdt_coedit_agents.crdt.document import DocumentView
from crdt_coedit_agents.ops.model import FIELD_KEY_PATTERN, MAX_TAG_LENGTH, TAG_PATTERN

__all__ = [
    "MAX_ANCHOR_ROWS",
    "SYSTEM_PROMPT",
    "AddTagStep",
    "EditPlan",
    "EditStep",
    "InsertAfterStep",
    "PlanError",
    "PlanRejectedError",
    "RemoveItemStep",
    "SetFieldStep",
    "SuggestionPlanner",
    "anchor_table",
]

MAX_ANCHOR_ROWS = 200
_ANCHOR_PATTERN = r"^(?:START|[0-9a-f]{32}\.[1-9][0-9]{0,17})$"

SYSTEM_PROMPT = (
    "You propose edits to a document that humans and other agents are editing at the same time.\n"
    "Return an edit plan whose steps address stable anchor ids, never character offsets: an "
    "offset measured against the text you were given is wrong as soon as someone types in front "
    "of it.\n"
    "Every step you propose is a suggestion. A human reviews the whole plan and decides whether "
    "it becomes part of the document; you cannot accept your own work, and you have no tool that "
    "could.\n"
    "Keep the plan small and specific: change what the instruction asks for and nothing else.\n\n"
    + UNTRUSTED_DATA_POLICY
)


class PlanError(RuntimeError):
    """The model did not return a usable plan."""


class PlanRejectedError(PlanError):
    """The plan was well formed but referenced something it may not touch."""


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InsertAfterStep(_Step):
    """Insert text immediately after an anchor. Use ``START`` for the beginning of the document."""

    tool: Literal["insert_after"] = "insert_after"
    anchor_id: str = Field(pattern=_ANCHOR_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_INSERT_CHARS)


class SetFieldStep(_Step):
    """Set a document field to a scalar value. ``null`` clears the field."""

    tool: Literal["set_field"] = "set_field"
    key: str = Field(pattern=FIELD_KEY_PATTERN)
    value: str | bool | int | None = None


class AddTagStep(_Step):
    tool: Literal["add_tag"] = "add_tag"
    tag: str = Field(pattern=TAG_PATTERN, max_length=MAX_TAG_LENGTH)


class RemoveItemStep(_Step):
    """Delete ``count`` elements of the text starting at ``anchor_id``."""

    tool: Literal["remove_item"] = "remove_item"
    anchor_id: str = Field(pattern=r"^[0-9a-f]{32}\.[1-9][0-9]{0,17}$")
    count: int = Field(default=1, ge=1, le=MAX_INSERT_CHARS)


EditStep = Annotated[
    InsertAfterStep | SetFieldStep | AddTagStep | RemoveItemStep, Field(discriminator="tool")
]


class EditPlan(BaseModel):
    """What the model returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=280)
    steps: tuple[EditStep, ...] = Field(min_length=1, max_length=64)

    def arguments(self, step: EditStep) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = step.model_dump(mode="json")
        payload.pop("tool")
        return payload


@dataclass(frozen=True, slots=True)
class AnchorRow:
    anchor_id: str
    preview: str


def anchor_table(view: DocumentView, *, max_rows: int = MAX_ANCHOR_ROWS) -> list[AnchorRow]:
    """Anchors the model may address: the document start, each word start, and the last element.

    Listing every character would make the prompt quadratic in the document size for no benefit —
    word starts are where edits actually go, the last element is how you append, and the replica
    resolves whichever anchor the plan names. Anchors the plan invents are rejected before any
    operation is created.

    The table walks *elements*, not characters: an element may hold more than one character, so
    each row's preview is taken from that element's own offset in the text. A word that starts in
    the middle of an element has no anchor of its own and is not listed. The last element is listed
    once, as the end of the document, even when it also starts a word.
    """
    rows = [AnchorRow(anchor_id="START", preview="<start of document>")]
    text = view.text
    for anchor_id, offset in zip(view.anchors, view.anchor_offsets, strict=True):
        if len(rows) >= max_rows - 1:
            break
        after_space = offset == 0 or text[offset - 1].isspace()
        if after_space and not text[offset].isspace():
            rows.append(AnchorRow(anchor_id=anchor_id, preview=text[offset : offset + 24]))
    if view.anchors:
        if rows[-1].anchor_id == view.anchors[-1]:
            rows.pop()
        rows.append(AnchorRow(anchor_id=view.anchors[-1], preview="<end of document>"))
    return rows


def render_document(view: DocumentView, rows: Sequence[AnchorRow]) -> str:
    """The document and its anchor table, as one block of untrusted data."""
    lines = [
        f"document: {view.doc_id}",
        f"fields: {view.fields}",
        f"tags: {view.tags}",
        "",
        "text:",
        view.text,
        "",
        "anchors (anchor_id -> text that follows it):",
    ]
    lines.extend(f"{row.anchor_id} -> {row.preview!r}" for row in rows)
    return "\n".join(lines)


class SuggestionPlanner:
    """Asks a provider for an edit plan and validates it against the edit tool registry."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        budget: BudgetLedger,
        max_steps: int = 12,
        max_tokens: int = 4_096,
    ) -> None:
        self._provider = provider
        self._budget = budget
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self.last_request_sha256: str = ""
        self.last_model: str = ""

    def build_request(self, view: DocumentView, instruction: str) -> LLMRequest:
        """The exact request sent to the provider. Deterministic, so it is replayable."""
        envelope = wrap_user_content("document", render_document(view, anchor_table(view)))
        content = (
            TextBlock(text=f"Instruction: {instruction}"),
            TextBlock(text=envelope.render()),
            TextBlock(
                text=(
                    f"Return an edit plan with at most {self._max_steps} steps. "
                    "Every anchor_id must come from the anchor table above."
                )
            ),
        )
        return LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(ChatMessage(role="user", content=content),),
            max_tokens=self._max_tokens,
            output_schema=EditPlan.model_json_schema(),
            role="planner",
        )

    def plan(self, view: DocumentView, instruction: str) -> EditPlan:
        """One budgeted model call. Raises :class:`PlanError` if the answer is not a valid plan."""
        request = self.build_request(view, instruction)
        self.last_request_sha256 = request.sha256()
        self._budget.reserve_step()
        response = self._provider.complete(request)
        self.last_model = response.model
        self._budget.record_usage(model=response.model, usage=response.usage)
        payload = response.parsed
        if payload is None:
            raise PlanError("the model returned no structured output")
        try:
            plan = EditPlan.model_validate(payload)
        except ValidationError as exc:
            locations = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
            raise PlanError(f"plan failed validation at: {', '.join(locations[:8])}") from exc
        if len(plan.steps) > self._max_steps:
            raise PlanRejectedError(
                f"plan has {len(plan.steps)} steps, more than the configured {self._max_steps}"
            )
        return plan

    def validate(self, plan: EditPlan, view: DocumentView) -> list[ValidatedCall]:
        """Run every step through the least-privilege registry and the document's anchor set."""
        sink: list[EditToolArgs] = []
        registry = build_edit_registry(sink)
        known = set(view.anchors) | {"START"}
        calls: list[ValidatedCall] = []
        for index, step in enumerate(plan.steps):
            block = ToolUseBlock(id=f"step_{index}", name=step.tool, arguments=plan.arguments(step))
            try:
                call = registry.validate_call(block)
            except (ToolNotAllowedError, ToolArgumentError) as exc:
                raise PlanRejectedError(f"step {index} ({step.tool}): {exc}") from exc
            anchor = call.canonical_args.get("anchor_id")
            if isinstance(anchor, str) and anchor not in known:
                raise PlanRejectedError(
                    f"step {index} ({step.tool}) addresses an anchor that is not in the document"
                )
            calls.append(call)
        return calls
