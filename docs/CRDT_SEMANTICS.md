# CRDT semantics

What this document guarantees, what it deliberately does not, and where each claim is checked.

## The state

A document is four independent CRDTs merged component-wise:

| Part | Type | State | Join |
|---|---|---|---|
| `text` | Fugue sequence | node set + per-node tombstone scopes | union |
| `fields` | LWW-register map, keyed by `(key, suggestion_id)` | one entry per register | per-register maximum by timestamp |
| `tags` | observed-remove set | add tags + removed tags | union of both halves |
| `decisions` | LWW-register map, keyed by `suggestion_id` | one entry per suggestion | per-register maximum by timestamp |

Each part is a join-semilattice, so the product is one too. `DocumentState.join` is the
component-wise join, which makes it commutative, associative and idempotent. Two replicas that
have applied the same *set* of operations therefore hold byte-identical state, whatever order the
operations arrived in and however often they were re-delivered.

Checked by `tests/property/test_algebra.py`, over states built from operations that real replicas
produced, so the laws are stated about states the system can actually reach.

## Identity

Every operation has a **dot** `(replica_id, seq)`:

* `replica_id` is `sha256(ed25519_public_key)[:16]`, so it cannot be claimed without the key;
* `seq` starts at 1 and is contiguous per replica.

A dot also names the element the operation creates — a Fugue node, an OR-Set tag. Element identity
is therefore global and needs no coordination. The laws above assume each dot names **one**
operation; that is enforced, not assumed:

* the Ed25519 signature covers `(doc_id, replica_id, seq, ts, suggestion_id, payload)`;
* the operation log has a unique index on `(doc_id, replica_id, author_seq)`;
* `FugueTree` raises `NodeConflictError` if two different nodes claim one dot
  (`tests/unit/test_fugue.py::test_two_nodes_with_the_same_dot_are_refused`).

## Time

Hybrid logical clocks (Kulkarni et al., *Logical Physical Clocks*, OPODIS 2014). A timestamp is
`(wall_ms, counter, replica_id)`, ordered lexicographically. It respects causality — if *a* happens
before *b*, then `ts(a) < ts(b)` — while staying close to wall-clock time, which matters because
the LWW registers resolve conflicts by timestamp.

Every verified remote timestamp is merged into the receiving replica's clock
(`Replica.receive` -> `HybridLogicalClock.observe`), which is what makes the clock *hybrid
logical* and not merely physical: the next local timestamp is strictly greater than anything the
replica has received, so a write that causally follows a remote one also sorts after it
(`tests/unit/test_replica.py::test_receiving_a_remote_timestamp_pulls_the_local_clock_past_it`).

A remote timestamp more than 60 seconds ahead of local wall time is **refused** rather than
adopted, on both paths that admit operations: a peer counts it in `SyncReport.rejected` and never
stores it, and the sync server refuses it with the rejection reason `clock_drift` before it reaches
the log. Without that bound, one replica could set its clock to the year 3000 and win every field
write — including a suggestion's accept flag — forever
(`tests/unit/test_hlc.py::test_a_remote_clock_too_far_ahead_is_refused`,
`tests/unit/test_replica.py::test_an_operation_dated_far_in_the_future_is_refused_on_receive`,
`tests/integration/test_hub.py::test_an_operation_dated_far_in_the_future_never_reaches_the_log`).
The cost is that a peer whose clock genuinely runs more than 60 seconds fast has its operations
dropped instead of buffered; it has to resend them once its clock is corrected.

## The sequence: Fugue

Based on Weidner and Kleppmann, *The Art of the Fugue: Minimizing Interleaving in Collaborative
Text Editing* (arXiv:2305.00583, 2023). The implementation is original; the tie-break rule below is
ours.

### Structure

Each element is a tree node with a parent, a side (`L` or `R`) and an immutable value. The list
order is the depth-first traversal

```text
traverse(n) = [traverse(c) for c in left children of n] + [n]
            + [traverse(c) for c in right children of n]
```

with a virtual root holding the top-level nodes on its right side.

To insert immediately after an element `left`:

* if `left` has no right children, the new node becomes a **right child of `left`**;
* otherwise it becomes a **left child of `left`'s successor**, which is the leftmost descendant of
  `left`'s first right child and therefore has no left children.

Exactly one case applies, so the position is always defined, and in both cases the new node is
traversed immediately after `left`.

### Why runs do not interleave

Typing a run produces a chain: each character is inserted after the previous one, which has no
right children yet. Two concurrent runs at the same position become two chains hanging off the same
node on the same side, and the traversal emits one whole subtree before the other. Typing a run
right-to-left (repeatedly inserting at the same anchor) produces a left-leaning chain with the same
property, which is where RGA interleaves and Fugue does not.

Checked for forward runs, backward runs, mixed runs and three-way concurrency in
`tests/property/test_fugue.py`.

### Tie-break

Siblings on the same side are ordered by `(seq, replica_id)` of their dot: **right children
ascending, left children descending**. The two directions mirror each other, so the structure is
symmetric under list reversal — among concurrent inserts immediately *after* a node the earliest is
closest to it, and among concurrent inserts immediately *before* a node the earliest is closest to
it as well. Any total order converges; this one was chosen because it is symmetric and because it
makes "the older edit stays nearer the anchor" true on both sides.

### Positions are resolved by the author

An insert operation carries the resolved `(parent, side)`, not the anchor the author aimed at.
Receivers never re-resolve it. If they did, placement would depend on what each replica happened to
have seen, and replicas would diverge.

### Arrival order

A node whose parent has not arrived is stored but is not reachable from the root, so it is
invisible until the parent shows up — no buffering, no causal-delivery requirement. Deletes and
observed-removes are keyed by the element they refer to, so they can also arrive first.

### Removals are keyed by element *and* scope

A delete or an observed-remove is identified by the pair `(target dot, suggestion_id)`, never by
the target alone. The same element can be removed independently in the main document and inside a
suggestion, and those are two separate facts: a state with room for only one of them would keep
whichever arrived last, so two replicas seeing the two removals in different orders would end up
with different documents. Storing the scopes as a set per element is what makes this half of the
state grow-only, and therefore what makes the join a real join.

`FugueTree` keeps `dot -> frozenset[scope]` for tombstones and `ORSet` keys removals by
`(dot, scope)`; both are unions under merge. Pinned by
`tests/unit/test_lww_orset.py::test_removing_one_tag_in_two_scopes_keeps_both_removals_in_either_order`,
and found in the first place by the randomized convergence property.

## Suggestions

Every element carries a `suggestion_id` (`None` = main document). `decisions[suggestion_id]` is an
ordinary LWW register: a suggestion is visible exactly when it says `true`.

Two scopes matter:

* **main view** — main elements plus accepted suggestions;
* **editing scope for suggestion S** — the main view plus S.

Position resolution happens *inside the editing scope*. A main edit therefore never becomes a child
of a suggested node, so the set of main nodes is closed under "parent" and the relative order of
main nodes does not depend on any suggestion.

That closure is **enforced, not assumed**. `parent` travels on the wire, so a replica that ignores
`insert_position` can name any node at all — a suggestion anchored inside a different, undecided
suggestion, or a main-scope element anchored inside one. Materialization therefore emits a node
only when the node *and every one of its ancestors* are in the view, which makes such an element
invisible everywhere its anchor is not
(`tests/unit/test_fugue.py::test_a_suggestion_anchored_to_another_suggestion_is_invisible_without_it`,
`tests/unit/test_document.py::test_a_suggestion_cannot_anchor_itself_inside_another_suggestion`).
The ancestor walk is memoized across a traversal and skipped entirely while no node carries a
suggestion id, so materializing a document without suggestions costs exactly what it did before.
That is what makes the two guarantees below exact rather than approximate.

**Rejecting leaves no trace.** After a suggestion is rejected, the main view is byte-identical to
replaying the same history with that suggestion's operations removed altogether. Checked by
replaying, not by re-reading the filter
(`tests/property/test_suggestions.py::test_a_rejected_suggestion_leaves_the_main_view_byte_identical`).

**Accepting lands next to the anchor.** An accepted run appears as one contiguous block whose
preceding visible element is the anchor the agent addressed, or — when that anchor was deleted
concurrently — the nearest element before it that survived
(`tests/property/test_suggestions.py::test_an_accepted_suggestion_lands_after_its_anchor_or_its_surviving_predecessor`).

Acceptance commutes with concurrent editing because it is an ordinary signed write: nothing is
rebased and nothing is re-signed. Decisions are themselves last-writer-wins, so accepting and then
rejecting converges on the later decision.

**One caveat, stated plainly.** Once a suggestion is accepted, later edits may anchor *inside* it,
because it is part of the main view. Rejecting it afterwards hides its own elements and, by
parent-closure, everything anchored inside them — the later edits disappear with it rather than
being re-parented, because there is no position left to re-parent them to. Acceptance is meant to
be a one-way step; re-rejecting an accepted suggestion is supported, converges, and is deterministic,
but it is not the case the guarantees above are about.

## Deletion

Deleting is a tombstone keyed by the target dot, per scope. Tombstone sets are grow-only, so
deletes commute with each other and with concurrent inserts. Deleted elements stay in the tree and
remain valid anchors, which is what lets an accepted suggestion land in the right place even when
the element it referenced is gone.

There is no tombstone garbage collection. A document's state grows with the number of operations
ever applied, not with its visible length. See Non-goals in the README.

## Delta synchronization

A replica summarizes what it has as a **version vector**: the highest contiguous sequence number
per author. "Send me what I am missing" is then a diff of two vectors, and the answer is a bounded
batch of signed operations. Measured cost for a peer ten operations behind, against a full-state
transfer: `bench/bench_crdt.py`, `delta_vs_full_state`.

Operations may be re-sent freely: applying one twice changes nothing, and the receiver reports it
as a duplicate rather than an error. Re-sending a *different* operation under a spent `(author,
seq)` is a replay and is refused.

## What is not modelled

* **Rich text.** Elements are single characters with no formatting marks.
* **Arbitrary JSON nesting.** The document has a fixed shape — one text sequence, one flat field
  map, one tag set. Field *values* may be nested JSON, but the CRDT treats each value as opaque.
* **Counters.** There is no PN-counter; a numeric field is a last-writer-wins register.
* **Move operations.** Moving text is a delete plus an insert, which interleaves badly under
  concurrency. Proper move semantics need a different construction.
* **Anti-entropy between peers without a server.** Replicas exchange deltas through the sync
  server. Peer-to-peer Merkle-tree anti-entropy is a non-goal.
