# sonictok Phase 1 — Plan 3: Optimization Ladder

Goal: climb single-thread throughput from Rung 0/1 (~36 MB/s) toward quicktok
native (160.7 MB/s on this M3 Pro). Each rung is measured and **test-gated**:
the full exactness suite (fixtures × 3 + oracle-diff × 2 + proptest) must stay
green, and each rung must show a `cargo bench` win or be reverted.

## Rungs (in order)

- **A — fast rank table.** Replace `HashMap<Vec<u8>, u32>` (SipHash, per-byte
  hashing) with a custom open-addressed table that hashes the whole byte slice in
  one pass (FxHash-style, vendored, dep-free). The per-pair rank lookup is the
  hottest path.
- **B — ASCII classification fast path.** A 256-entry byte-class table so ASCII
  bytes (≈99% of typical text) skip UTF-8 decode + binary search in the
  pretokenizer. Falls back to the Unicode range tables for non-ASCII.
- **C — BPE scratch reuse + small-piece specialization.** Reuse the `parts`
  buffer across pieces; avoid per-piece allocation.
- **D (stretch) — 2-byte trie + dense validity memos** (quicktok's structural
  wins), if A–C don't close the gap enough.

## Method

For each rung: implement → `cargo test --workspace` green → `cargo bench -p
sonictok` → record MB/s in `bench/BASELINE.md` → commit (or revert if no win).
