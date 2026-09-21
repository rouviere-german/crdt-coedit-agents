"""Differential tests against pycrdt (Yjs), on causally serialized traces only.

An independent implementation is the best check that our sequence CRDT agrees with a mature one on
the cases where agreement is *required*. Those cases are the causally serialized ones: one editor
at a time, every edit applied on top of the previous state. There both implementations must produce
the sequential result, so any disagreement is a bug in ours.

Concurrent edits are deliberately out of scope. Yjs uses YATA, which orders concurrent inserts at
the same position by client id, while Fugue's tree orders them by its own tie-break rule. Both are
correct CRDTs and both converge; they simply pick different (equally valid) orders, so a
differential test on concurrent traces would be asserting a coincidence rather than a property.
That boundary is the reason pycrdt is a dev-only dependency and never appears in ``src/``.
"""

from __future__ import annotations

from typing import Any

import pycrdt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crdt_coedit_agents.replica import Replica
from tests.conftest import make_replica

pytestmark = [pytest.mark.integration, pytest.mark.property]

ALPHABET = "abcdefghij "
Edit = tuple[bool, int, str]

EDITS = st.lists(
    st.tuples(
        st.booleans(),
        st.integers(min_value=0, max_value=40),
        st.text(alphabet=ALPHABET, min_size=1, max_size=5),
    ),
    min_size=1,
    max_size=25,
)


class Oracle:
    """pycrdt's text type, driven by the same trace."""

    def __init__(self) -> None:
        self.doc: Any = pycrdt.Doc()
        self.text: Any = pycrdt.Text()
        self.doc["text"] = self.text

    def insert(self, index: int, value: str) -> None:
        self.text.insert(index, value)

    def delete(self, index: int, length: int) -> None:
        del self.text[index : index + length]

    def read(self) -> str:
        return str(self.text)


def apply_to_replica(replica: Replica, index: int, value: str) -> None:
    anchors = replica.view().anchors
    anchor = anchors[index - 1] if index > 0 else None
    replica.insert_text(value, after=anchor)


def delete_from_replica(replica: Replica, index: int, length: int) -> None:
    anchors = replica.view().anchors
    replica.delete_run(anchors[index], min(length, len(anchors) - index))


@given(EDITS)
@settings(max_examples=40)
def test_a_serialized_trace_produces_the_same_text_as_pycrdt(edits: list[Edit]) -> None:
    replica = make_replica(1)
    oracle = Oracle()

    for is_insert, position, value in edits:
        current = len(oracle.read())
        if is_insert or current == 0:
            index = position % (current + 1)
            apply_to_replica(replica, index, value)
            oracle.insert(index, value)
        else:
            index = position % current
            length = min(len(value), current - index)
            delete_from_replica(replica, index, length)
            oracle.delete(index, length)
        assert replica.view().text == oracle.read()


@given(EDITS)
@settings(max_examples=25)
def test_a_serialized_trace_relayed_through_a_second_replica_still_matches(
    edits: list[Edit],
) -> None:
    """Editing alternates between two replicas, each fully synchronized before it writes."""
    first = make_replica(1)
    second = make_replica(2, start_ms=1_700_000_500_000)
    oracle = Oracle()

    for turn, (is_insert, position, value) in enumerate(edits):
        writer, reader = (first, second) if turn % 2 == 0 else (second, first)
        writer.receive(list(reader.ops()))
        current = len(oracle.read())
        if is_insert or current == 0:
            index = position % (current + 1)
            apply_to_replica(writer, index, value)
            oracle.insert(index, value)
        else:
            index = position % current
            length = min(len(value), current - index)
            delete_from_replica(writer, index, length)
            oracle.delete(index, length)
        reader.receive(list(writer.ops()))
        assert writer.view().text == oracle.read()
        assert reader.view().text == oracle.read()
        assert writer.state_hash() == reader.state_hash()


def test_the_oracle_and_the_replica_agree_on_a_hand_written_trace() -> None:
    replica = make_replica(1)
    oracle = Oracle()
    steps: list[tuple[str, int, str | int]] = [
        ("insert", 0, "hello"),
        ("insert", 5, " world"),
        ("insert", 0, ">> "),
        ("delete", 3, 2),
        ("insert", 3, "HE"),
        ("delete", 0, 3),
    ]
    for kind, index, argument in steps:
        if kind == "insert":
            assert isinstance(argument, str)
            apply_to_replica(replica, index, argument)
            oracle.insert(index, argument)
        else:
            assert isinstance(argument, int)
            delete_from_replica(replica, index, argument)
            oracle.delete(index, argument)
    assert replica.view().text == oracle.read() == "HEllo world"


def test_concurrent_edits_are_documented_as_outside_the_differential_scope() -> None:
    """Both converge; they need not agree with each other, so only convergence is asserted."""
    first = make_replica(1)
    second = make_replica(2, start_ms=1_700_000_500_000)
    first.insert_text("shared")
    second.receive(list(first.ops()))

    anchor = first.view().anchors[2]
    left = first.insert_text("AAA", after=anchor)
    right = second.insert_text("BBB", after=anchor)
    first.receive(right)
    second.receive(left)

    assert first.view().text == second.view().text
    text = first.view().text
    windows = {text[index : index + 3] for index in range(len(text))}
    assert {"AAA", "BBB"} <= windows
