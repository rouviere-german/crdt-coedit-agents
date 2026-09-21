"""The agent's edit tools: least privilege, closed schemas and anchor addressing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crdt_coedit_agents.agentic.errors import ToolArgumentError, ToolNotAllowedError
from crdt_coedit_agents.agentic.providers.base import ToolUseBlock
from crdt_coedit_agents.agentic.tools import Capability, ToolRegistry
from crdt_coedit_agents.agents.tools import (
    EDIT_TOOL_NAMES,
    MAX_INSERT_CHARS,
    AddTagArgs,
    EditToolArgs,
    InsertAfterArgs,
    RemoveItemArgs,
    SetFieldArgs,
    build_edit_registry,
)

pytestmark = pytest.mark.unit

ANCHOR = "a" * 32 + ".1"


def registry() -> tuple[ToolRegistry, list[EditToolArgs]]:
    sink: list[EditToolArgs] = []
    return build_edit_registry(sink), sink


def test_the_registry_offers_exactly_the_four_edit_tools() -> None:
    edit_registry, _ = registry()
    assert edit_registry.names == tuple(sorted(EDIT_TOOL_NAMES))
    assert edit_registry.granted == frozenset({Capability.WRITE_DATA})


def test_there_is_no_tool_for_accepting_a_suggestion_or_reaching_the_network() -> None:
    edit_registry, _ = registry()
    for forbidden in ("accept_suggestion", "grant_role", "http_get", "run_shell"):
        with pytest.raises(ToolNotAllowedError, match="not available"):
            edit_registry.get(forbidden)


def test_arguments_are_validated_against_closed_schemas() -> None:
    edit_registry, sink = registry()
    call = edit_registry.validate_call(
        ToolUseBlock(id="1", name="insert_after", arguments={"anchor_id": ANCHOR, "text": "hi"})
    )
    assert call.canonical_args == {"anchor_id": ANCHOR, "text": "hi"}
    call.run()
    assert sink == [InsertAfterArgs(anchor_id=ANCHOR, text="hi")]

    with pytest.raises(ToolArgumentError, match="invalid arguments"):
        edit_registry.validate_call(
            ToolUseBlock(
                id="2",
                name="insert_after",
                arguments={"anchor_id": ANCHOR, "text": "hi", "priority": "urgent"},
            )
        )


def test_anchors_must_be_dots_or_the_start_sentinel() -> None:
    assert InsertAfterArgs(anchor_id="START", text="x").anchor_id == "START"
    assert InsertAfterArgs(anchor_id=ANCHOR, text="x").anchor_id == ANCHOR
    with pytest.raises(ValidationError):
        InsertAfterArgs(anchor_id="12", text="x")
    with pytest.raises(ValidationError):
        RemoveItemArgs(anchor_id="START")


def test_tool_arguments_carry_the_same_limits_as_the_operations() -> None:
    with pytest.raises(ValidationError):
        InsertAfterArgs(anchor_id="START", text="x" * (MAX_INSERT_CHARS + 1))
    with pytest.raises(ValidationError):
        InsertAfterArgs(anchor_id="START", text="")
    with pytest.raises(ValidationError):
        SetFieldArgs(key="Not A Key", value=1)
    with pytest.raises(ValidationError):
        AddTagArgs(tag="tag with <html>")
    with pytest.raises(ValidationError):
        RemoveItemArgs(anchor_id=ANCHOR, count=0)
    assert SetFieldArgs(key="status", value=None).value is None


def test_call_hashes_change_with_any_argument() -> None:
    edit_registry, _ = registry()

    def call_hash(text: str) -> str:
        block = ToolUseBlock(
            id="1", name="insert_after", arguments={"anchor_id": ANCHOR, "text": text}
        )
        return edit_registry.validate_call(block).call_hash

    assert call_hash("hi") == call_hash("hi")
    assert call_hash("hi") != call_hash("hi ")


def test_the_tool_specs_sent_to_a_model_are_stable_and_closed() -> None:
    edit_registry, _ = registry()
    specs = edit_registry.specs()
    assert [spec.name for spec in specs] == sorted(EDIT_TOOL_NAMES)
    for spec in specs:
        assert spec.input_schema["additionalProperties"] is False
