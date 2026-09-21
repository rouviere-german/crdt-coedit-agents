"""Agent replicas: a typed edit plan from an LLM, applied as a suggestion a human must accept.

An agent is a full replica with the ``suggester`` role. It reads the *main* view, hands the text to
the model wrapped as untrusted data, asks for a typed plan addressing stable CRDT anchor ids, and
applies the plan under a fresh suggestion id. Nothing it writes is visible in the main view until an
owner writes an accept flag, and its role makes writing that flag impossible.
"""

from crdt_coedit_agents.agents.planner import (
    EditPlan,
    EditStep,
    PlanError,
    PlanRejectedError,
    SuggestionPlanner,
)
from crdt_coedit_agents.agents.replica import AgentReplica, SuggestionResult
from crdt_coedit_agents.agents.tools import (
    AddTagArgs,
    InsertAfterArgs,
    RemoveItemArgs,
    SetFieldArgs,
    build_edit_registry,
)

__all__ = [
    "AddTagArgs",
    "AgentReplica",
    "EditPlan",
    "EditStep",
    "InsertAfterArgs",
    "PlanError",
    "PlanRejectedError",
    "RemoveItemArgs",
    "SetFieldArgs",
    "SuggestionPlanner",
    "SuggestionResult",
    "build_edit_registry",
]
