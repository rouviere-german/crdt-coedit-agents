"""The shared element limits, checked directly at their boundaries."""

from __future__ import annotations

from typing import Any

import pytest

from crdt_coedit_agents.crdt import limits
from crdt_coedit_agents.ops import model

pytestmark = pytest.mark.unit

REPLICA = "a" * 32


def test_the_operation_model_uses_the_same_limits_as_the_snapshot_path() -> None:
    assert model.MAX_TEXT_VALUE_BYTES is limits.MAX_TEXT_VALUE_BYTES
    assert model.MAX_FIELD_VALUE_BYTES is limits.MAX_FIELD_VALUE_BYTES
    assert model.MAX_FIELD_DEPTH is limits.MAX_FIELD_DEPTH
    assert model.MAX_SEAL_RUNS is limits.MAX_SEAL_RUNS
    assert model.TAG_PATTERN is limits.TAG_PATTERN
    assert model.SUGGESTION_ID_PATTERN is limits.SUGGESTION_ID_PATTERN


def test_text_values_are_checked_at_the_byte_boundary() -> None:
    assert limits.check_text_value("x" * limits.MAX_TEXT_VALUE_BYTES) == "x" * 8
    assert limits.check_text_value("é" * 4) == "é" * 4  # 8 bytes of UTF-8
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        limits.check_text_value("x" * (limits.MAX_TEXT_VALUE_BYTES + 1))
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        limits.check_text_value("é" * 5)


def test_depth_is_checked_without_recursing_into_the_value() -> None:
    at_limit: Any = 1
    for _ in range(limits.MAX_FIELD_DEPTH):
        at_limit = [at_limit]
    limits.check_depth(at_limit)

    very_deep: Any = 1
    for _ in range(50_000):
        very_deep = {"n": very_deep}
    with pytest.raises(ValueError, match="levels deep"):
        limits.check_field_value(very_deep)


def test_seal_runs_are_bounded_well_formed_and_forward() -> None:
    runs = [f"{REPLICA}.{n}-{n}" for n in range(1, 2 * limits.MAX_SEAL_RUNS, 2)]
    assert len(limits.check_seal_runs(runs)) == limits.MAX_SEAL_RUNS
    with pytest.raises(ValueError, match="at most"):
        limits.check_seal_runs([*runs, f"{REPLICA}.999-999"])
    with pytest.raises(ValueError, match="backwards"):
        limits.check_seal_runs([f"{REPLICA}.5-2"])
    with pytest.raises(ValueError, match="must be a string"):
        limits.check_seal_runs([5])


@pytest.mark.parametrize(
    ("check", "good", "bad"),
    [
        (limits.check_field_key, "title", "Title"),
        (limits.check_tag, "needs review", "needs\nreview"),
        (limits.check_suggestion_id, "sg-" + "0" * 16, "sg-" + "0" * 15),
        (limits.check_doc_id, "notes-1", "-notes"),
    ],
)
def test_identifier_checks_accept_the_pattern_and_nothing_else(
    check: Any, good: str, bad: str
) -> None:
    assert check(good) == good
    with pytest.raises(ValueError, match="must match"):
        check(bad)
    with pytest.raises(ValueError, match="must be a string"):
        check(None)
