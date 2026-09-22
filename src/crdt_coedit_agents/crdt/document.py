"""The co-edited document: a Fugue text list, an OR-Set of tags and an LWW-register field map.

``DocumentState`` is the pure CRDT payload. It is a join-semilattice: :meth:`DocumentState.join`
is the component-wise join of its parts, so it inherits their commutativity, associativity and
idempotence. Nothing in this module knows about signatures, transports or the operation log.

Suggestion visibility
---------------------
Every element carries a ``suggestion_id`` (``None`` = main document). A separate LWW register map
``decisions`` holds ``suggestion_id -> bool``; a suggestion is visible exactly when its register
says ``true``. Accepting or rejecting is therefore an ordinary write that commutes with concurrent
edits and needs no re-signing or rebasing of the suggested operations.

Two scopes matter:

``main_view``
    what the document looks like: main elements plus accepted suggestions.
``editing_scope(suggestion_id)``
    what an editor may anchor to while writing that suggestion: the main view plus its own
    suggestion. A main edit uses ``editing_scope(None)``, which is the main view.

Sealing
-------
A decision does not merely say *yes*; it says *yes, to exactly these operations*. The register
holds ``{"accept": bool, "seal": [...]}``, where the seal names the operations the decider could
see, compressed into contiguous ``(replica, seq)`` runs. An element of an accepted suggestion is
visible only when the operation that created it is inside the seal, so operations added to a
suggestion after it was decided are inert: they are stored, they replicate, and they are invisible
in every main view, for ever. Without that, ``accept`` bound only a suggestion *id*, and an agent
could keep appending to an id a human had already approved.

The seal names an exact set rather than a high-water mark, because an operation the decider had not
yet seen must stay outside it even when its sequence number is below the highest one that was
reviewed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from crdt_coedit_agents.core.canonical import canonical_json, sha256_hex
from crdt_coedit_agents.crdt.fugue import FugueNode, FugueTree, Side, Tombstone
from crdt_coedit_agents.crdt.hlc import Timestamp
from crdt_coedit_agents.crdt.ids import Dot, compress_dots, parse_dot, parse_dot_ranges
from crdt_coedit_agents.crdt.limits import (
    check_doc_id,
    check_field_key,
    check_field_value,
    check_seal_runs,
    check_suggestion_id,
    check_tag,
    check_text_value,
)
from crdt_coedit_agents.crdt.lww import LWWEntry, LWWMap, VisibilityFilter
from crdt_coedit_agents.crdt.orset import ORSet, Tag, TagRemoval

__all__ = [
    "STATE_VERSION",
    "DocumentState",
    "DocumentView",
    "Scope",
    "SuggestionStatus",
    "decision_value",
]

#: Version of the serialized state. Bumped when the encoding changes shape, so an older snapshot is
#: refused loudly instead of being read as something it is not.
STATE_VERSION = 2

#: How a decision is stored in the ``decisions`` register map.
SealRanges = Mapping[str, tuple[tuple[int, int], ...]]


def decision_value(*, accept: bool, seal: Sequence[str]) -> dict[str, Any]:
    """The register value one ``suggestion_decide`` operation writes."""
    return {"accept": accept, "seal": list(seal)}


def _read_decision(entry: LWWEntry) -> tuple[bool, tuple[str, ...]]:
    """``(accept, seal runs)`` for one decision register. A value that is not a decision
    object accepts nothing, which is the safe reading of a state nobody signed."""
    value = entry.value
    if isinstance(value, Mapping):
        raw = value.get("seal")
        runs = tuple(str(item) for item in raw) if isinstance(raw, list) else ()
        return value.get("accept") is True, runs
    return value is True, ()


class SuggestionStatus:
    """The three states a suggestion can be in, as seen by one replica."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Scope:
    """A visibility filter over ``(suggestion_id, author)``, named for logs and metrics.

    ``extra`` is the suggestion currently being drafted: its author sees the whole draft, sealed or
    not, because that is the thing under review. Everything else must be both accepted *and* inside
    that decision's seal.
    """

    name: str
    accepted: frozenset[str]
    extra: str | None = None
    seals: Mapping[str, SealRanges] = field(default_factory=dict, hash=False)

    def __call__(self, suggestion_id: str | None, author: Dot | None) -> bool:
        if suggestion_id is None or suggestion_id == self.extra:
            return True
        if suggestion_id not in self.accepted or author is None:
            return False
        ranges = self.seals.get(suggestion_id, {}).get(author.replica_id, ())
        return any(low <= author.seq <= high for low, high in ranges)


@dataclass(frozen=True, slots=True)
class DocumentView:
    """A materialized, JSON-like snapshot of the document.

    ``anchors[i]`` names the i-th visible element and ``anchor_offsets[i]`` is the character offset
    in ``text`` where that element's value begins. Locally typed text is one character per element,
    but an element may hold up to ``MAX_TEXT_VALUE_BYTES`` on the wire, so an index into ``text``
    is not an index into ``anchors``: map between them through ``anchor_offsets``.
    """

    doc_id: str
    scope: str
    text: str
    fields: dict[str, JsonValue]
    tags: list[str]
    anchors: list[str]
    anchor_offsets: list[int]

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "doc_id": self.doc_id,
            "scope": self.scope,
            "text": self.text,
            "fields": dict(self.fields),
            "tags": list(self.tags),
        }

    @property
    def content_hash(self) -> str:
        """SHA-256 over the canonical JSON of the view (anchors excluded)."""
        return sha256_hex(canonical_json(self.as_json()))


@dataclass(slots=True)
class DocumentState:
    """The replicated payload of one document."""

    doc_id: str
    text: FugueTree = field(default_factory=FugueTree)
    fields: LWWMap = field(default_factory=LWWMap)
    tags: ORSet = field(default_factory=ORSet)
    decisions: LWWMap = field(default_factory=LWWMap)

    # --------------------------------------------------------------- lattice

    def copy(self) -> DocumentState:
        return DocumentState(
            doc_id=self.doc_id,
            text=self.text.copy(),
            fields=self.fields.copy(),
            tags=self.tags.copy(),
            decisions=self.decisions.copy(),
        )

    def join(self, other: DocumentState) -> DocumentState:
        """Component-wise join. Both sides must describe the same document."""
        if self.doc_id != other.doc_id:
            raise ValueError("cannot join states of different documents")
        return DocumentState(
            doc_id=self.doc_id,
            text=self.text.join(other.text),
            fields=self.fields.join(other.fields),
            tags=self.tags.join(other.tags),
            decisions=self.decisions.join(other.decisions),
        )

    # ------------------------------------------------------------ suggestions

    def known_suggestions(self) -> frozenset[str]:
        """Suggestion ids that appear anywhere in the state, decided or not."""
        found: set[str] = set()
        for node in self.text.nodes():
            if node.suggestion_id is not None:
                found.add(node.suggestion_id)
        for _, marks in self.text.tombstones():
            found.update(mark.scope for mark in marks if mark.scope is not None)
        for (_, suggestion_id), _ in self.fields.items():
            if suggestion_id is not None:
                found.add(suggestion_id)
        for tag in self.tags.tags():
            if tag.suggestion_id is not None:
                found.add(tag.suggestion_id)
        found.update(self.decisions.keys())
        return frozenset(found)

    def status(self, suggestion_id: str) -> str:
        entry = self.decisions.entry(suggestion_id)
        if entry is None:
            return SuggestionStatus.PENDING
        accept, _ = _read_decision(entry)
        return SuggestionStatus.ACCEPTED if accept else SuggestionStatus.REJECTED

    def accepted_suggestions(self) -> frozenset[str]:
        # LWWMap.keys() returns a frozenset of register names, not a mapping view.
        decided = self.decisions.keys()
        return frozenset(key for key in decided if self.status(key) == SuggestionStatus.ACCEPTED)

    def seal_of(self, suggestion_id: str) -> tuple[str, ...]:
        """The operations one decision sealed in, as contiguous ``(replica, seq)`` runs."""
        entry = self.decisions.entry(suggestion_id)
        return () if entry is None else _read_decision(entry)[1]

    def _seals(self, accepted: frozenset[str]) -> dict[str, SealRanges]:
        return {key: parse_dot_ranges(self.seal_of(key)) for key in accepted}

    def suggestion_authors(self, suggestion_id: str) -> tuple[Dot, ...]:
        """Every operation that contributed to one suggestion, by dot.

        Raises ``ValueError`` when an element of the suggestion does not record its author, because
        such an element cannot be sealed and would silently vanish from an accepted suggestion.
        """
        dots: list[Dot] = []

        def take(author: Dot | None, what: str) -> None:
            if author is None:
                raise ValueError(f"{what} of suggestion {suggestion_id} has no recorded author")
            dots.append(author)

        for node in self.text.nodes():
            if node.suggestion_id == suggestion_id:
                dots.append(node.dot)
        for _, marks in self.text.tombstones():
            for mark in marks:
                if mark.scope == suggestion_id:
                    take(mark.author, "a delete")
        for (_, scope), entry in self.fields.items():
            if scope == suggestion_id:
                take(entry.author, "a field write")
        adds, removes = self.tags.state_items()
        for tag in adds:
            if tag.suggestion_id == suggestion_id:
                dots.append(tag.dot)
        for removal in removes:
            if removal.suggestion_id == suggestion_id:
                take(removal.author, "a tag removal")
        return tuple(dots)

    def suggestion_seal(self, suggestion_id: str) -> tuple[str, ...]:
        """The seal a reviewer deciding *now* would write: exactly what this replica can see."""
        return compress_dots(self.suggestion_authors(suggestion_id))

    def pending_suggestions(self) -> tuple[str, ...]:
        decided = self.decisions.keys()
        return tuple(sorted(self.known_suggestions() - decided))

    # ----------------------------------------------------------------- scopes

    def main_scope(self) -> Scope:
        """The main view: main elements plus the sealed part of every accepted suggestion."""
        accepted = self.accepted_suggestions()
        return Scope(name="main", accepted=accepted, seals=self._seals(accepted))

    def editing_scope(self, suggestion_id: str | None) -> Scope:
        """What an editor writing ``suggestion_id`` may anchor to."""
        if suggestion_id is None:
            return self.main_scope()
        accepted = self.accepted_suggestions()
        return Scope(
            name=f"suggestion:{suggestion_id}",
            accepted=accepted,
            extra=suggestion_id,
            seals=self._seals(accepted),
        )

    # ---------------------------------------------------------- materializing

    def view(self, scope: VisibilityFilter | None = None, *, name: str = "main") -> DocumentView:
        filter_: VisibilityFilter = scope if scope is not None else self.main_scope()
        label = getattr(filter_, "name", name)
        nodes = self.text.visible_nodes(filter_)
        offsets: list[int] = []
        position = 0
        for node in nodes:
            offsets.append(position)
            position += len(node.value)
        return DocumentView(
            doc_id=self.doc_id,
            scope=str(label),
            text="".join(node.value for node in nodes),
            fields=self.fields.materialize(filter_),
            tags=self.tags.materialize(filter_),
            anchors=[node.anchor_id for node in nodes],
            anchor_offsets=offsets,
        )

    def to_json(self) -> dict[str, Any]:
        """A deterministic, order-independent encoding of the whole replicated state.

        Two replicas that have applied the same set of operations produce identical output, which
        is what snapshots store and what the convergence tests and the demo compare.
        """
        adds, removes = self.tags.state_items()
        return {
            "v": STATE_VERSION,
            "doc_id": self.doc_id,
            "text_nodes": [
                {
                    "dot": str(node.dot),
                    "parent": str(node.parent) if node.parent is not None else None,
                    "side": node.side.value,
                    "value": node.value,
                    "suggestion_id": node.suggestion_id,
                }
                for node in self.text.nodes()
            ],
            "text_tombstones": [
                {
                    "dot": str(dot),
                    "marks": [
                        {
                            "scope": mark.scope,
                            "author": str(mark.author) if mark.author is not None else None,
                        }
                        for mark in sorted(marks, key=Tombstone.sort_key)
                    ],
                }
                for dot, marks in self.text.tombstones()
            ],
            "fields": [
                {
                    "key": key,
                    "suggestion_id": suggestion_id,
                    "value": entry.value,
                    "ts": entry.timestamp.encode(),
                    "author": str(entry.author) if entry.author is not None else None,
                }
                for (key, suggestion_id), entry in self.fields.items()
            ],
            "tag_adds": [
                {"element": tag.element, "dot": str(tag.dot), "suggestion_id": tag.suggestion_id}
                for tag in adds
            ],
            "tag_removes": [
                {
                    "dot": str(item.dot),
                    "suggestion_id": item.suggestion_id,
                    "author": str(item.author) if item.author is not None else None,
                }
                for item in removes
            ],
            "decisions": [
                {"suggestion_id": key, "value": entry.value, "ts": entry.timestamp.encode()}
                for (key, _), entry in self.decisions.items()
            ],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> DocumentState:
        """Rebuild a state from :meth:`to_json`. Raises ``ValueError`` on anything malformed.

        This is the one way into a state that does not go through an operation, so every element
        is held to the same limits the operation model enforces on the wire
        (:mod:`crdt_coedit_agents.crdt.limits`): text values, tags, field keys and values,
        suggestion ids, decisions and their seals. A snapshot's state hash cannot stand in for
        these checks, because whoever wrote the file could also have written a matching hash.
        """
        if payload.get("v") != STATE_VERSION:
            raise ValueError("unsupported document state version")
        doc_id = payload["doc_id"]
        if not isinstance(doc_id, str):
            raise ValueError("doc_id must be a string")  # noqa: TRY004 - a value error, not a type
        state = cls(doc_id=check_doc_id(doc_id))
        for item in payload["text_nodes"]:
            parent = item["parent"]
            state.text.add_node(
                FugueNode(
                    dot=parse_dot(item["dot"]),
                    parent=parse_dot(parent) if parent is not None else None,
                    side=Side(item["side"]),
                    value=check_text_value(item["value"]),
                    suggestion_id=_optional_suggestion(item["suggestion_id"]),
                )
            )
        for item in payload["text_tombstones"]:
            for mark in item["marks"]:
                state.text.tombstone(
                    parse_dot(item["dot"]),
                    _optional_suggestion(mark["scope"]),
                    _optional_dot(mark["author"]),
                )
        for item in payload["fields"]:
            state.fields.set(
                check_field_key(item["key"]),
                LWWEntry(
                    value=check_field_value(item["value"]),
                    timestamp=Timestamp.decode(item["ts"]),
                    author=_optional_dot(item["author"]),
                ),
                _optional_suggestion(item["suggestion_id"]),
            )
        for item in payload["tag_adds"]:
            state.tags.add(
                Tag(
                    element=check_tag(item["element"]),
                    dot=parse_dot(item["dot"]),
                    suggestion_id=_optional_suggestion(item["suggestion_id"]),
                )
            )
        for item in payload["tag_removes"]:
            state.tags.remove(
                TagRemoval(
                    dot=parse_dot(item["dot"]),
                    suggestion_id=_optional_suggestion(item["suggestion_id"]),
                    author=_optional_dot(item["author"]),
                )
            )
        for item in payload["decisions"]:
            state.decisions.set(
                check_suggestion_id(item["suggestion_id"]),
                LWWEntry(
                    value=_checked_decision(item["value"]),
                    timestamp=Timestamp.decode(item["ts"]),
                ),
            )
        return state

    def suggestion_payload(self, suggestion_id: str) -> dict[str, Any]:
        """Exactly what one suggestion contributes, and nothing a concurrent editor wrote.

        This is the object a reviewer approves. It is a pure function of the suggestion's own
        operations, so concurrent human edits do not change it, while an agent that appends more
        operations to an already-reviewed suggestion does.
        """
        return {
            "v": 1,
            "doc_id": self.doc_id,
            "suggestion_id": suggestion_id,
            "text_nodes": [
                {
                    "dot": str(node.dot),
                    "parent": str(node.parent) if node.parent is not None else None,
                    "side": node.side.value,
                    "value": node.value,
                }
                for node in self.text.nodes()
                if node.suggestion_id == suggestion_id
            ],
            "text_tombstones": [
                str(dot)
                for dot, marks in self.text.tombstones()
                if any(mark.scope == suggestion_id for mark in marks)
            ],
            "fields": [
                {"key": key, "value": entry.value, "ts": entry.timestamp.encode()}
                for (key, scope), entry in self.fields.items()
                if scope == suggestion_id
            ],
            "tag_adds": [
                {"element": tag.element, "dot": str(tag.dot)}
                for tag in self.tags.tags()
                if tag.suggestion_id == suggestion_id
            ],
            "tag_removes": [
                str(item.dot)
                for item in self.tags.state_items()[1]
                if item.suggestion_id == suggestion_id
            ],
        }

    def suggestion_text(self, suggestion_id: str) -> str:
        """Exactly the characters this suggestion inserts, in document order."""
        scope = self.editing_scope(suggestion_id)
        return "".join(
            node.value
            for node in self.text.visible_nodes(scope)
            if node.suggestion_id == suggestion_id
        )

    def suggestion_deletions(self, suggestion_id: str) -> int:
        """How many existing elements this suggestion would delete."""
        return sum(
            1
            for _, marks in self.text.tombstones()
            if any(mark.scope == suggestion_id for mark in marks)
        )

    def suggestion_digest(self, suggestion_id: str) -> str:
        """SHA-256 of :meth:`suggestion_payload`: the identity of what a reviewer approved."""
        return sha256_hex(canonical_json(self.suggestion_payload(suggestion_id)))

    def state_hash(self) -> str:
        """SHA-256 over :meth:`to_json`, independent of the order operations arrived in."""
        return sha256_hex(canonical_json(self.to_json()))

    def size_bytes(self) -> int:
        """Rough in-document size used by the server's maximum-document-size limit."""
        text_bytes = sum(len(node.value.encode("utf-8")) for node in self.text.nodes())
        field_bytes = sum(
            len(canonical_json(entry.value).encode("utf-8")) for _, entry in self.fields.items()
        )
        tag_bytes = sum(len(tag.element.encode("utf-8")) for tag in self.tags.tags())
        return text_bytes + field_bytes + tag_bytes


def _optional_suggestion(value: Any) -> str | None:
    """A scope read from a snapshot: ``None`` (the main document) or a well-formed suggestion id."""
    return None if value is None else check_suggestion_id(value)


def _checked_decision(value: Any) -> dict[str, Any]:
    """A decision register read from a snapshot, in exactly the shape ``apply_op`` writes."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"accept", "seal"}
        or not isinstance(value["accept"], bool)
        or not isinstance(value["seal"], list)
    ):
        raise ValueError('a decision must be {"accept": <bool>, "seal": [<run>, ...]}')
    return decision_value(accept=value["accept"], seal=check_seal_runs(value["seal"]))


def _optional_dot(value: Any) -> Dot | None:
    """Parse an author dot from a snapshot. ``None`` means the element predates any operation."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("an author must be a dot string or null")  # noqa: TRY004 - a value error
    return parse_dot(value)
