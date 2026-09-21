"""Conflict-free replicated data types: HLC, LWW-register map, OR-Set and a Fugue sequence.

Every type in this package is a join-semilattice: ``join`` is commutative, associative and
idempotent, so replicas that have seen the same set of operations hold the same state regardless
of the order in which the operations arrived. :mod:`crdt_coedit_agents.crdt.document` composes
the three types into one document and materializes it into a JSON-like view.
"""

from crdt_coedit_agents.crdt.document import (
    DocumentState,
    DocumentView,
    Scope,
    SuggestionStatus,
)
from crdt_coedit_agents.crdt.fugue import ROOT_ID, FugueNode, FugueTree, Side, UnknownAnchorError
from crdt_coedit_agents.crdt.hlc import ClockDriftError, HybridLogicalClock, Timestamp
from crdt_coedit_agents.crdt.ids import Dot, VersionVector, parse_dot
from crdt_coedit_agents.crdt.lww import LWWEntry, LWWMap
from crdt_coedit_agents.crdt.orset import ORSet, Tag, TagRemoval

__all__ = [
    "ROOT_ID",
    "ClockDriftError",
    "DocumentState",
    "DocumentView",
    "Dot",
    "FugueNode",
    "FugueTree",
    "HybridLogicalClock",
    "LWWEntry",
    "LWWMap",
    "ORSet",
    "Scope",
    "Side",
    "SuggestionStatus",
    "Tag",
    "TagRemoval",
    "Timestamp",
    "UnknownAnchorError",
    "VersionVector",
    "parse_dot",
]
