"""Offline planner policies: a scripted model, and a deliberately gullible one.

``ScriptedPlanPolicy`` reads the anchor table out of the request exactly as a model would and hands
the plan back through the structured-output field, so tests and the demo exercise the real prompt,
the real schema and the real validation path without a network call.

``GullibleInjectionPolicy`` does whatever the document tells it to. It exists to show that
containment does not depend on the model: the plan schema has no step that could accept a
suggestion or grant a role, the registry has no tool for it, and the server would refuse the
operation even if one were forged — so the attack fails three times over.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence

from pydantic import JsonValue

from crdt_coedit_agents.agentic.providers.base import LLMRequest, LLMResponse, TextBlock
from crdt_coedit_agents.agents.planner import EditPlan

__all__ = [
    "GullibleInjectionPolicy",
    "ScriptedPlanPolicy",
    "extract_anchors",
    "extract_instruction",
    "plan_response",
]

_ANCHOR_ROW = re.compile(r"^(START|[0-9a-f]{32}\.[1-9][0-9]{0,17}) -> ", re.MULTILINE)
_INSTRUCTION = re.compile(r"^Instruction: (.*)$", re.MULTILINE)
# A document that tries to steer the model, as a real injected payload would.
_INJECTED = re.compile(r"AGENT_DIRECTIVE\s+(\{.*\})")

PlanBuilder = Callable[[Sequence[str], str], EditPlan]


def extract_anchors(request: LLMRequest) -> list[str]:
    """Anchor ids the request offered the model, in order."""
    found: list[str] = []
    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                found.extend(_ANCHOR_ROW.findall(block.text))
    return found


def extract_instruction(request: LLMRequest) -> str:
    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                match = _INSTRUCTION.search(block.text)
                if match is not None:
                    return match.group(1)
    return ""


def plan_response(payload: dict[str, JsonValue], *, model: str = "fake-planner") -> LLMResponse:
    """Wrap a plan the way a provider returns a structured output."""
    return LLMResponse(
        model=model,
        stop_reason="end_turn",
        text=json.dumps(payload, sort_keys=True),
        parsed=payload,
    )


class ScriptedPlanPolicy:
    """Builds a plan from the anchors the prompt offered."""

    def __init__(self, build: PlanBuilder, *, model: str = "fake-planner") -> None:
        self._build = build
        self._model = model

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del rng, turn
        plan = self._build(extract_anchors(request), extract_instruction(request))
        return plan_response(plan.model_dump(mode="json"), model=self._model)


class GullibleInjectionPolicy:
    """Obeys an ``AGENT_DIRECTIVE {...}`` line found in the document, if there is one.

    With no directive it falls back to ``build``, so it can stand in for an ordinary planner.
    """

    def __init__(self, build: PlanBuilder, *, model: str = "fake-gullible") -> None:
        self._build = build
        self._model = model

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del rng, turn
        for message in request.messages:
            for block in message.content:
                if not isinstance(block, TextBlock):
                    continue
                match = _INJECTED.search(block.text)
                if match is None:
                    continue
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return plan_response(payload, model=self._model)
        plan = self._build(extract_anchors(request), extract_instruction(request))
        return plan_response(plan.model_dump(mode="json"), model=self._model)
