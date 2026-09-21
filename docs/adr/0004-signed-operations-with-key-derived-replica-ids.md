# 4. Operations are signed, and a replica id is derived from its public key

- Status: accepted
- Date: 2026-09-21

## Context

A sync server sits between replicas and is a natural single point of trust: it could rewrite an
operation, attribute an edit to someone else, or replay an old one. A CRDT does not care about
delivery order, so a malicious relay cannot cause divergence by reordering — but it can cause
*forgery*, and forgery is the interesting attack when agents are involved: "the owner approved
this" is a claim someone might want to fabricate.

Signing operations solves authenticity but raises key distribution. Options:

**A key registry on the server.** Replicas enrol a public key; the server maps replica ids to keys.
Works, but the mapping is now something the server can lie about, which puts the trust back where
we took it from.

**Certificates.** Correct and completely out of proportion for a document-editing tool.

**Derive the identity from the key.** `replica_id = sha256(public_key)[:16]`. There is then nothing
to distribute: any peer recomputes the id from the key an operation presents and compares it with
the id inside the signed envelope.

A second question: how are per-document permissions expressed? A key proves *who*, not *what they
may do*.

## Decision

Each replica holds an Ed25519 key. `replica_id = sha256(public_key_bytes)[:16].hex()`. An operation
carries its author's public key and a signature over the canonical JSON of
`(v, doc_id, replica_id, seq, ts, suggestion_id, payload)`, domain-separated with a fixed prefix.

Verification recomputes the replica id from the presented key, checks that the timestamp names the
same author, and then checks the signature.

Authorization is separate: a short-lived HS256 JWT grants one principal one role
(`owner` / `editor` / `suggester`) on one document, and **pins it to one replica id**. Every
operation a connection submits must be authored by that replica.

## Consequences

- Identity needs no distribution and no registry. A spoofed replica id is detected by arithmetic,
  not by policy (`tests/unit/test_signing.py::test_a_spoofed_replica_id_is_detected_even_with_a_valid_signature`).
- The two mechanisms fail independently. Stealing a token lets an attacker *connect* as a
  principal; it does not let them author operations, because they would also need the private key.
  Stealing a key lets them sign operations; it does not let them connect, because they would also
  need a token. Both are required, and the server checks that they agree.
- `doc_id` and `seq` are inside the signature, so an operation cannot be moved to another document
  or renumbered. Combined with the unique `(doc_id, replica_id, author_seq)` index on the operation
  log, that is the replay guard: re-sending the same bytes is a duplicate, re-using a spent
  sequence number with different content is refused.
- The cost is real. Signing is about 120 µs and verifying about 150 µs per operation on the
  reference machine (`bench/bench_crdt.py`), and a keystroke is one operation. That caps a replica
  at roughly 8 000 local edits per second, which is far above human typing and far below what a
  batch import would want. Applying an already-verified operation is 6 µs, so the signature is
  essentially the whole cost.
- Every operation carries a 32-byte public key it did not strictly need to — the server could cache
  it per replica. Keeping it makes an operation self-verifying offline, which is the point: two
  replicas syncing a file on a USB stick can still check each other's work.
- Keys live in 0600 files created by `keygen`. There is no rotation, no revocation and no
  compromise recovery; losing a key means creating a new replica id. That is a genuine gap and is
  listed in Non-goals.
- Ed25519 comes from `cryptography`; nothing here implements a primitive. JWTs come from PyJWT with
  the algorithm list pinned, so `alg: none` and algorithm-confusion attacks do not apply
  (`tests/unit/test_auth.py::test_unsigned_tokens_are_refused_because_the_algorithm_is_pinned`).
