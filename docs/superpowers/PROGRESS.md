# sonictok — autonomous session progress log

Date: 2026-06-16. Worked through Plan 1 (already done before this session), then
Plan 2 and Plan 3 autonomously. Safety gate the whole way: a change is only
committed when the full exactness suite is green
(fixtures × 3 + oracle-diff × 2 + proptest 9000 cases).

## Status: all green

- `cargo test --workspace -- --include-ignored`: **55 passed, 0 failed**
- `cargo fmt --all --check`: clean
- `cargo clippy --workspace --all-targets -D warnings`: clean

## Plan 2 — o200k_base + o200k_harmony (DONE)

- Generalized the engine: `Grammar { Cl100k, O200k }` + a `Scanner` enum that
  dispatches; `Engine` holds a grammar instead of a type param.
- Extracted shared scalar helpers (`pretok/common.rs`); added the o200k scanner
  (`pretok/o200k.rs`) implementing the two case-aware letter alternatives
  (`UPPER* LOWER+` / `UPPER+ LOWER*`) with explicit backtracking + contraction
  suffix + `[\r\n/]` punct tail.
- Extended the Unicode generator with `O200K_UPPER` / `O200K_LOWER` classes.
- Exported + vendored o200k_base and o200k_harmony blobs; both byte-exact.
- All three encodings: fixtures + oracle-diff + proptest green.
- One bug caught by the gate during the refactor (inverted `is_other` condition
  in `scan_punct`) — fixed before commit.

## Plan 3 — optimization ladder (DONE through Rung D)

cl100k single-thread, `bench/corpus.txt`, this M3 Pro (see `bench/BASELINE.md`):

| Rung | MiB/s | note |
|------|------:|------|
| baseline (Rung 0/1) | 35.8 | |
| A — FxHash rank table | 41.6 | swapped SipHash→FxHash, kept hashbrown |
| B — ASCII class fast path | 76.8 | **biggest win** — 128-entry byte table vs binary search |
| C1 — ASCII char_at | 80.6 | skip UTF-8 decode for ASCII |
| C2 — reuse BPE scratch | 82.3 | no per-piece alloc |
| D — dense 2-byte pair table | 87.1 | direct 256×256 table for the pair scan |

**Net 2.43× (35.8 → 87.1 MiB/s).** quicktok-native here is 153.3 MiB/s, so we're
at ~0.57×. We decisively beat every other exact tokenizer (bpe-openai ~37 MB/s,
tiktoken ~14 MB/s).

Reverted (didn't pay): a custom open-addressed table (lost to hashbrown);
`target-cpu=native` (wash — hot path isn't autovectorized).

### Profiling + two more reverts (further session)

Added a `pretokenize_only` criterion bench to split the cost:
**pretok ≈ 289 MiB/s (~30%), BPE ≈ 70%** of full encode at Rung D.

Two more rungs attempted and **reverted** (kept the discipline — correctness was
fine, but no speed win):

- **Rung E — id-based BPE + `(id,id)→rank` HashMap memo.** Reformulated the merge
  loop to track token ids and look merges up by id-pair, eliminating byte-slice
  hashing in the hot loop. *Correct* (exactness green) but **regressed to 33
  MiB/s**: the memo HashMap (~hundreds of k entries) is far larger than the warm
  byte-string map and thrashes cache. quicktok's win here is a tight,
  **direct-indexed ~2MB dense memo with a bijective mixer (perfect hash)** — not
  a general HashMap. That's the right structure but real perfect-hashing work;
  left for supervised implementation.
- **ASCII run fast-scan in the pretokenizer.** Scan ASCII letter/case runs by
  byte instead of via `char_at`. **Wash** — `char_at` already ASCII-fast-paths,
  and the closure indirection ate the gain (pretok stayed ~281 MiB/s). The
  scalar pretokenizer is at its ceiling; only true SIMD (16–32 bytes/instr)
  moves it.

**Conclusion:** the easy/medium ladder tops out at ~87 MiB/s (~0.57× quicktok).
Both remaining wins are the hard structural pieces:
1. **Dense direct-indexed merge memo** (perfect-hash) — unlocks the BPE 70%.
2. **SIMD pretokenizer** — unlocks the pretok 30%.
Both are best done supervised (high complexity; the oracle guards correctness but
the perf tuning + perfect-hash construction warrant review).

## What's left to reach quicktok-native (left for supervised work)

These are the high-effort, higher-risk structural wins quicktok uses. The oracle
catches correctness, but they're large enough to warrant review:

1. **Hand-compiled SIMD pretokenizer** (Rung 4) — the per-char scalar scanner is
   now the dominant cost. AVX2/NEON scanning + a single-pass ASCII product
   machine is the path to ~quicktok pretok speed.
2. **2-byte trie + dense validity memos** (quicktok's structural wins) for the
   longest-match / merge lookups.
3. **PGO** and a `native`/`portable` build-profile split (Phase 4 packaging).

## Beyond the ladder: product surface (reliable, exactness-safe)

With single-thread blocked on the risky structural work, added high-value
product features instead (all byte-exact, all tested):

- **Parallel batch** (`encode_batch`/`count_batch`, rayon, `parallel` feature):
  **~526 MiB/s (6.2×)** on 11 cores; per-doc encode is the already-verified path.
- **Stable C ABI** (`crates/sonictok-cabi` → `libsonictok.{a,dylib}` +
  `include/sonictok.h`): load/encode/decode/count/free over `extern "C"`.
  Verified by a clang-built C smoke test (`cargo run -p xtask -- test-cabi`).
- **Python bindings** (`bindings/python`, PyO3 abi3 + maturin, tiktoken-style
  API: `get_encoding`, `encode_ordinary`, `encode(allowed_special=...)`,
  `encode_with_special`, `decode`, `count`, `encode_batch`, `.name`,
  `.n_vocab`). **Byte-exact vs real tiktoken on all three encodings**; GIL
  released during encode/batch. 6 pytest cases.
- **`embed-data` feature**: `include_bytes!` the vocab blobs so the binary/wheel
  is self-contained (verified `import sonictok` from /tmp with no data dir).

## Phase status vs the spec

- Phase 1 (core + OpenAI encodings): **complete** and exact.
- Phase 2 (open-model encodings llama3/qwen3 + NFC): not started.
- Phase 3 (generic importer): not started.
- Phase 4 (bindings/packaging, batch APIs): **substantially done** — C ABI,
  Python wheel (PyO3/maturin), parallel batch, embedded data all landed. Left:
  CMake/`find_package`, published wheels/CI matrix, PGO.

## Final tally (this session)

From an exact-but-slow Plan-1 cl100k baseline to: 3 encodings (cl100k, o200k,
o200k_harmony) byte-exact; 2.43× single-thread (87 MiB/s) + 6.2× batch (526
MiB/s); Rust + C + Python surfaces; self-contained data. All committed in small
steps with fixtures + oracle-diff + proptest green throughout.
