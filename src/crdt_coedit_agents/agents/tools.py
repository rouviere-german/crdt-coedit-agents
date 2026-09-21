"""The four edit tools an agent may use, and the least-privilege registry that holds them.

Every tool addresses **stable CRDT anchor ids**, never character offsets. An offset computed
against the text an agent read is meaningless by the time the plan is applied: a human may have
typed three characters in front of it. An anchor id names one element for as long as that element
exists, so a plan stays meaningful across concurrent edits — and when the anchored element has been
deleted, the suggestion still lands next to its nearest surviving predecessor instead of at a
random offset.

The registry is the privilege boundary. There is deliberately **no** tool for accepting a
suggestion, changing roles or reaching the network: a model that is talked into wanting one finds
that it does not exist. Arguments are validated against closed pydantic models
(``extra="forbid"``, strict mode), and the canonical form of each call is what the approval record
hashes, so a reviewer approves exactly the call that runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from crdt_coedit_agents.agentic.tools import Capability, ToolDefinition, ToolRegistry
from crdt_coedit_agents.ops.model import (
    FIELD_KEY_PATTERN,
    MAX_TAG_LENGTH,
    TAG_PATTERN,
)

__all__ = [
    "EDIT_TOOL_NAMES",
    "MAX_INSERT_CHARS",
    "AddTagArgs",
    "EditToolArgs",
    "InsertAfterArgs",
    "RemoveItemArgs",
    "SetFieldArgs",
    "build_edit_registry",
]

MAX_INSERT_CHARS = 2_000
_ANCHOR_PATTERN = r"^(?:START|[0-9a-f]{32}\.[1-9][0-9]{0,17})$"
EDIT_TOOL_NAMES = ("insert_after", "set_field", "add_tag", "remove_item")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InsertAfterArgs(_Args):
    """Insert text immediately after an anchor. ``START`` means the beginning of the document."""

    anchor_id: str = Field(pattern=_ANCHOR_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_INSERT_CHARS)


class SetFieldArgs(_Args):
    """Write one document field. ``value`` may be any JSON value, including ``null`` to clear it."""

    key: str = Field(pattern=FIELD_KEY_PATTERN)
    value: JsonValue = None


class AddTagArgs(_Args):
    tag: str = Field(pattern=TAG_PATTERN, max_length=MAX_TAG_LENGTH)


class RemoveItemArgs(_Args):
    """Delete ``count`` visible elements starting at ``anchor_id``."""

    anchor_id: str = Field(pattern=r"^[0-9a-f]{32}\.[1-9][0-9]{0,17}$")
    count: int = Field(default=1, ge=1, le=MAX_INSERT_CHARS)


EditToolArgs = InsertAfterArgs | SetFieldArgs | AddTagArgs | RemoveItemArgs


@dataclass(frozen=True, slots=True)
class _Recorder:
    """Collects validated calls instead of executing them.

    Applying a plan is the replica's job, not a tool handler's: a handler that wrote operations
    straight into the document would make it impossible to validate the whole plan before any of it
    takes effect.
    """

    calls: list[EditToolArgs]

    def record(self, args: EditToolArgs) -> JsonValue:
        self.calls.append(args)
        return {"recorded": len(self.calls)}


def build_edit_registry(sink: list[EditToolArgs]) -> ToolRegistry:
    """A registry holding exactly the four edit tools, writing into ``sink``.

    The registry is granted ``WRITE_DATA`` only. It has no ``NETWORK_EGRESS``, ``SEND_MESSAGE`` or
    ``EXECUTE_CODE`` capability, so no tool that needs one can even be registered.
    """
    recorder = _Recorder(calls=sink)
    registry = ToolRegistry({Capability.WRITE_DATA})
    registry.register(
        ToolDefinition(
            name="insert_after",
            description=(
                "Insert text immediately after the element with this anchor id. "
                "Use 'START' to insert at the beginning of the document."
            ),
            args_model=InsertAfterArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=recorder.record,
        )
    )
    registry.register(
        ToolDefinition(
            name="set_field",
            description="Set a document field (for example title or status) to a JSON value.",
            args_model=SetFieldArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=recorder.record,
        )
    )
    registry.register(
        ToolDefinition(
            name="add_tag",
            description="Add a tag to the document.",
            args_model=AddTagArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=recorder.record,
        )
    )
    registry.register(
        ToolDefinition(
            name="remove_item",
            description="Delete count elements of the text starting at this anchor id.",
            args_model=RemoveItemArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=recorder.record,
        )
    )
    return registry
