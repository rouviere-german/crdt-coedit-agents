# 3. Agent edits are ordinary operations hidden behind a last-writer-wins accept flag

- Status: accepted
- Date: 2026-09-21

## Context

An agent's edits must not reach the document until a human says so. Three ways to arrange that:

**Hold the operations outside the document** — queue them in a review table and apply them when
approved. Simple to reason about, but the queued operations rot: by the time a human looks, the
text they were anchored to has moved or been deleted, and applying them means rebasing, which is
exactly the problem a CRDT exists to avoid.

**Branch the document** — give the agent its own branch and merge on approval. Powerful, but a
merge of two CRDT branches is either a full three-way merge (no longer a CRDT) or a join (in which
case the branch was never isolated in the first place).

**Apply the operations immediately and hide them.** The operations are real, signed, replicated
CRDT operations from the moment they are made; a visibility flag decides whether they are part of
the document. Acceptance is then a *write*, not a rebase.

A second question: where does the human decision live? An approval row in a database is the obvious
answer, but it does not replicate, and it does not commute with concurrent editing.

## Decision

Every operation carries an optional `suggestion_id`. A separate last-writer-wins register map,
`decisions`, holds `suggestion_id -> bool`. An element is visible in the main view when it has no
suggestion id, or when its suggestion's register says `true`.

Accepting or rejecting is an ordinary signed operation, subject to the same role policy, the same
hash chain and the same replication as any edit. Only the `owner` role may write it.

Independently, the agent must clear a second gate before its operations are published at all: an
approval record in the local store, bound to a **digest** of exactly what that suggestion
contributes.

## Consequences

- Acceptance commutes with everything else. A human accepting while two other people type produces
  no conflict and needs no coordination, because the flag is just another register write.
- Nothing is ever re-signed or rebased. The operations a reviewer looked at are the operations that
  become visible, byte for byte.
- Rejection is exact: the main view becomes byte-identical to a replay without the suggestion,
  which `tests/property/test_suggestions.py` checks by actually replaying.
- Two gates is one more than strictly necessary, and they do different jobs. The accept flag is the
  load-bearing one: it is replicated, it converges, and it is role-enforced server-side. The
  approval record adds *separation of duties* (an agent cannot approve itself), *single use*, and
  a binding to a digest, so an agent that keeps editing after a human looked at its draft finds its
  approval no longer applies. Keeping only the flag would let an unreviewed agent fill the document
  with drafts; keeping only the record would make the decision unreplicated.
- The approval store is a local JSON file with a lock. It is not tamper-evident — whoever can write
  it can forge an approved record — so it is protected like the audit log rather than trusted like
  a signature. The accept flag, which is signed, is what a third party can verify.
- Suggested writes to a field live in their own register, keyed by `(key, suggestion_id)`, so
  rejecting one lets the previous main value win again instead of leaving the key deleted.
- Rejected suggestions are never garbage-collected: their operations stay in the log and in the
  state forever. That is deliberate (the log is append-only evidence) and is a real cost.
- The honest limitation: once a suggestion is accepted, later edits may anchor inside it. Rejecting
  it afterwards hides its own elements but keeps those later edits. Acceptance is a one-way step in
  practice, and the guarantees are stated for undecided or rejected suggestions.
