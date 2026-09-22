"""Bounds on what one document element may hold, shared by every way into a replica's state.

There are two such ways. Operations arrive on the wire and are parsed by the pydantic models in
:mod:`crdt_coedit_agents.ops.model`; a snapshot is rebuilt by
:meth:`crdt_coedit_agents.crdt.document.DocumentState.from_json` without ever constructing an
operation. Both call the checks below, so a hand-written snapshot cannot carry anything a signed
operation could not.

The checks raise ``ValueError`` with a message that names the limit, or ``TypeError`` for a value
that is not even a string. Pydantic turns a ``ValueError`` into a validation error on the wire path
(where the model has already checked types); the snapshot path lets both propagate, and
:class:`crdt_coedit_agents.snapshot.Snapshot` reports them as a malformed snapshot.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import JsonValue

from crdt_coedit_agents.core.canonical import canonical_json
from crdt_coedit_agents.crdt.ids import DOT_RANGE_PATTERN

__all__ = [
    "DOC_ID_PATTERN",
    "FIELD_KEY_PATTERN",
    "MAX_FIELD_DEPTH",
    "MAX_FIELD_VALUE_BYTES",
    "MAX_SEAL_RUNS",
    "MAX_TAG_LENGTH",
    "MAX_TEXT_VALUE_BYTES",
    "SUGGESTION_ID_PATTERN",
    "TAG_PATTERN",
    "check_depth",
    "check_doc_id",
    "check_field_key",
    "check_field_value",
    "check_seal_runs",
    "check_suggestion_id",
    "check_tag",
    "check_text_value",
]

MAX_TEXT_VALUE_BYTES = 8
MAX_FIELD_VALUE_BYTES = 4_096
MAX_FIELD_DEPTH = 8
MAX_TAG_LENGTH = 64
#: How many contiguous ``(replica, seq)`` runs one decision may seal. A suggestion written the
#: ordinary way needs one run per contributing replica; the bound is generous so that interleaved
#: authorship still fits, and refusing beyond it keeps a decision small enough to sign and relay.
MAX_SEAL_RUNS = 128

SUGGESTION_ID_PATTERN = r"^sg-[0-9a-f]{16}$"
FIELD_KEY_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
TAG_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _/-]{0,63}$"
DOC_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"

_SUGGESTION_ID = re.compile(SUGGESTION_ID_PATTERN)
_FIELD_KEY = re.compile(FIELD_KEY_PATTERN)
_TAG = re.compile(TAG_PATTERN)
_DOC_ID = re.compile(DOC_ID_PATTERN)
_DOT_RANGE = re.compile(DOT_RANGE_PATTERN)


def _require_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    return value


def check_text_value(value: object) -> str:
    """One text element: non-empty, at most ``MAX_TEXT_VALUE_BYTES`` of UTF-8, no NUL."""
    text = _require_str(value, "a text value")
    if not text:
        raise ValueError("a text value may not be empty")
    if len(text.encode("utf-8")) > MAX_TEXT_VALUE_BYTES:
        raise ValueError(f"text values are limited to {MAX_TEXT_VALUE_BYTES} UTF-8 bytes")
    if "\x00" in text:
        raise ValueError("text values may not contain NUL")
    return text


def check_depth(value: JsonValue, depth: int = 0) -> None:
    """Refuse a JSON value that nests deeper than ``MAX_FIELD_DEPTH``.

    Iterative, so a snapshot holding a value nested thousands of levels deep is refused instead of
    exhausting the interpreter's stack while it is being checked.
    """
    pending: list[tuple[JsonValue, int]] = [(value, depth)]
    while pending:
        item, level = pending.pop()
        if level > MAX_FIELD_DEPTH:
            raise ValueError(f"field values may nest at most {MAX_FIELD_DEPTH} levels deep")
        if isinstance(item, dict):
            pending.extend((child, level + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, level + 1) for child in item)


def check_field_value(value: JsonValue) -> JsonValue:
    """A field register's value: bounded depth and bounded canonical size."""
    check_depth(value)
    if len(canonical_json(value).encode("utf-8")) > MAX_FIELD_VALUE_BYTES:
        raise ValueError(f"field values are limited to {MAX_FIELD_VALUE_BYTES} bytes")
    return value


def check_field_key(value: object) -> str:
    key = _require_str(value, "a field key")
    if not _FIELD_KEY.fullmatch(key):
        raise ValueError(f"a field key must match {FIELD_KEY_PATTERN}")
    return key


def check_tag(value: object) -> str:
    tag = _require_str(value, "a tag")
    if len(tag) > MAX_TAG_LENGTH or not _TAG.fullmatch(tag):
        raise ValueError(f"a tag must match {TAG_PATTERN} and be at most {MAX_TAG_LENGTH} long")
    return tag


def check_suggestion_id(value: object) -> str:
    suggestion_id = _require_str(value, "a suggestion id")
    if not _SUGGESTION_ID.fullmatch(suggestion_id):
        raise ValueError(f"a suggestion id must match {SUGGESTION_ID_PATTERN}")
    return suggestion_id


def check_doc_id(value: object) -> str:
    doc_id = _require_str(value, "a document id")
    if not _DOC_ID.fullmatch(doc_id):
        raise ValueError(f"a document id must match {DOC_ID_PATTERN}")
    return doc_id


def check_seal_runs(runs: Iterable[object]) -> tuple[str, ...]:
    """The runs one decision seals: bounded in number, well formed, never running backwards."""
    checked: list[str] = []
    for item in runs:
        if len(checked) >= MAX_SEAL_RUNS:
            raise ValueError(f"a decision may seal at most {MAX_SEAL_RUNS} operation runs")
        run = _require_str(item, "a sealed run")
        if not _DOT_RANGE.fullmatch(run):
            raise ValueError("sealed runs must look like <replica_id>.<low>-<high>")
        low, _, high = run.partition("-")
        if int(low.rsplit(".", 1)[1]) > int(high):
            raise ValueError("a sealed run may not run backwards")
        checked.append(run)
    return tuple(checked)
