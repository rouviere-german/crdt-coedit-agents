# 2. A Fugue tree for the sequence, with author-resolved positions

- Status: accepted
- Date: 2026-09-21

## Context

The document needs a sequence CRDT that humans and agents can edit at the same time, from
different machines, some of them offline. Three families were candidates:

**Position strings** (Logoot, LSEQ). Each element carries a dense rational-like identifier, so
insertion is a local computation and there is no tree to walk. They interleave badly: two people
typing a sentence at the same spot produce a character-level zipper, because each character's
identifier is chosen independently of the ones around it. For a document a human is supposed to
read afterwards, that is disqualifying.

**RGA** (and Yjs's YATA, which is close in behaviour). Each element records the element it was
inserted after. Concurrent runs typed left-to-right do not interleave. Runs typed *right-to-left* —
type the last word, move the cursor back, type the word before it — do, because every character
names the same left origin and they are then ordered among themselves by timestamp.

**Fugue** (Weidner and Kleppmann, arXiv:2305.00583, 2023). Elements form a tree with a parent and a
side, so an element can be anchored to what is on its left *or* to what is on its right. The paper
proves maximal non-interleaving: no run of concurrent insertions is split by another.

A second, independent question: an insert operation can carry either the *intent* ("after anchor
X") or the *resolved position* ("left child of node Y"). Carrying intent lets receivers
re-interpret it, which sounds attractive for an agent whose anchor was deleted.

## Decision

Use a Fugue tree, and put the **resolved** `(parent, side)` on the wire.

Sibling ordering: same-side siblings are ordered by `(seq, replica_id)`, right children ascending
and left children descending, so the structure is symmetric under list reversal.

Insert positions are resolved inside the *editing scope* (the main view, plus the suggestion being
written), never against the full node set.

## Consequences

- Concurrent runs stay in one piece, whichever direction they were typed in. Tested for forward,
  backward, mixed and three-way concurrency in `tests/property/test_fugue.py`.
- Because positions are resolved by the author, placement cannot depend on what a receiver happened
  to have seen. If receivers re-resolved intent, two replicas with different views would compute
  different positions for the same operation and never converge — the one thing a CRDT must not do.
- The price is that an agent's anchor is a *position*, not a promise. If the anchored element is
  deleted concurrently, the suggestion still lands where that element was, next to the nearest
  surviving predecessor. That is a documented guarantee rather than a repair mechanism, and it is
  why the review record shows a human the exact text rather than an offset.
- Resolving inside the editing scope is what makes rejection exact: main nodes are closed under
  "parent", so their order does not depend on any suggestion, and a rejected suggestion leaves the
  main view byte-identical to a run without it.
- Inserting is `O(log k)` in the number of siblings at one anchor, which is effectively constant
  for typing. Materializing the text is `O(n)` and cached until the node set changes; the cost is
  measured at 20 000 elements in `bench/bench_crdt.py`.
- Tombstones are never collected, so state grows with the number of operations ever applied. At
  100 000 elements the CRDT state is about 44 MiB of Python objects (`state_memory_100k`), which is
  fine for the documents this repository targets and would not be for a year-long shared notebook.
- pycrdt (Yjs) is kept as a dev-only differential oracle, but only on causally serialized traces:
  YATA orders concurrent inserts by client id and Fugue by its tree, so both are correct and they
  need not agree. `tests/integration/test_pycrdt_differential.py` says so in code.
