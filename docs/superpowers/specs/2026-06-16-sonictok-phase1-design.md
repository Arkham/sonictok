# sonictok — Design Spec

**Date:** 2026-06-16
**Status:** Approved for planning (Phase 1)
**Author:** brainstormed with the team

---

## 1. Summary

`sonictok` is a production-grade, byte-exact Byte-Pair-Encoding (BPE) tokenizer
written in Rust. It is a "super optimized" reimagining of
[quicktok](https://github.com/dmatth1/quicktok) (an already heavily-optimized
C++ exact tokenizer): byte-identical to its references, competitive with or
faster than the fastest known exact tokenizer, exhaustively tested, and built
with the structure and discipline expected of senior/staff-level systems code.

The full product targets **feature parity with quicktok**: the BPE core, six
encoding grammars, NFC normalization, a generic importer for arbitrary
byte-level BPE tokenizers, and production bindings (Python wheels, a stable C
ABI, CMake). Because that is far too large for a single spec, the work is
**decomposed into phases**. This document specifies the overall architecture and
**Phase 1** in full. Later phases get their own spec → plan → build cycle.

### Success criterion

Best-in-class **production** tokenizer: competitive speed *and* exactness *and*
great ergonomics. Not a benchmark-only stunt and not a cleanliness-only
exercise — a library someone would actually ship.

### Performance target

**Beat quicktok native** on single-thread throughput (~120+ MB/s on cl100k,
Apple M1, from quicktok's published tables) while remaining byte-exact. This is
the aspirational ceiling and the hardest of the targets considered; reaching it
requires matching *every* structural technique quicktok uses and finding
marginal wins on top. Intermediate checkpoints on the climb:

- **Checkpoint A:** decisively beat every *other* exact tokenizer
  (≥ `bpe-openai` ~37 MB/s, ideally 2–3×) — i.e. fastest Rust exact tokenizer.
- **Checkpoint B:** quicktok-class (within ~15–20%, ~100+ MB/s).
- **Target:** beat quicktok native (~120+ MB/s).

### The one non-negotiable

**Byte-exact output.** We never trade correctness for speed. Every benchmark is
exactness-checked before timing; every optimization is proven equal to a naive
reference.

---

## 2. Phasing

Each phase is independently shippable and useful. The architecture is designed
up front so later phases slot in as *extensions*, not rewrites.

| Phase | Scope |
|-------|-------|
| **1 (this spec)** | Core engine + OpenAI encodings: data loader, exact backtracking-BPE core, pretokenizer framework, `cl100k_base` + `o200k_base` + `o200k_harmony`, exactness test harness, benchmark rig. |
| **2** | Open-model encodings: `llama3`, `qwen3` (forces NFC normalization), `llama4` code path. |
| **3** | Generic importer: pretokenizer-grammar classifier, data-file writer, verify-against-reference pipeline (tekken support, DeepSeek refusal). |
| **4** | Bindings & packaging: stable C ABI (`cbindgen`), Python wheels (`PyO3`/`maturin`, tiktoken-style API), CMake/`find_package`, batch/parallel APIs surfaced to bindings. |

---

## 3. Key decisions (locked)

- **Language:** Rust. Speed parity with C++ (LLVM backend, full layout control,
  SIMD via `core::arch` intrinsics + runtime feature detection); first-class
  testing (`proptest`, `criterion`, `cargo fuzz`); compiler-enforced
  thread-safety (`Sync`); excellent bindings story for Phase 4 (`PyO3`,
  `cbindgen`).
- **Optimization philosophy:** correctness-first, benchmark-driven. Ship a
  clean *exact* implementation verified byte-for-byte first, with the benchmark
  harness wired from day one; then optimize in measured rungs, keeping the full
  exactness suite green and justifying each step with a before/after number.
  Optimizations only ever change *data structures*, never *results*.
- **Verification:** vendored data files + a two-layer exactness strategy
  (hermetic checked-in fixtures for everyday dev; deep live-differential testing
  against real `tiktoken` over large corpora on a schedule), plus property tests
  and fuzzing.
- **License:** MIT (matches quicktok and the tiktoken ecosystem). Vendored vocab
  carries upstream license notices in `NOTICE`.

---

## 4. Architecture

### 4.1 Design principle

A small, safe, **cold** configuration/loading layer wrapping a panic-free,
allocation-free **hot** encode/decode core. The hot path is branch-lean,
SIMD-accelerated, and allocation-free per call (reused scratch buffers). The
cold path is `Result`-based and liberally validated.

### 4.2 Workspace layout (Cargo workspace)

```
sonictok/
├── crates/
│   ├── sonictok-core/        # the engine: BPE, pretokenizers, tries, memos. No I/O, no deps.
│   │   ├── bpe/              # ranked-merge backtracking BPE + optimization ladder
│   │   ├── pretok/           # pretokenizer framework + cl100k/o200k SIMD scanners
│   │   ├── vocab/            # in-memory vocab structures (trie, rank tables, special tokens)
│   │   └── simd/             # arch-gated SIMD primitives (avx2/sse/neon) + scalar fallback
│   ├── sonictok-data/        # data-file format: (de)serialize vendored vocab/unicode blobs
│   ├── sonictok/             # public API crate: Tokenizer, get_encoding, errors, batch
│   └── sonictok-testkit/     # shared test/bench corpora loaders, fixture types
├── fixtures/                 # checked-in input→ids vectors (generated, version-stamped)
├── data/                     # vendored vocab blobs (sonictok binary format)
├── tools/                    # Python export/fixture-gen scripts (tiktoken/HF references)
├── benches/                  # criterion benchmarks + corpus streaming
└── xtask/                    # cargo-xtask: codegen, data regen, verification orchestration
```

**Rationale:**

- `sonictok-core` has **zero external dependencies and no I/O**. It takes
  already-loaded vocab structures and bytes in, returns token ids out. This is
  the part optimized to the metal and fuzzed hardest — pure, testable, portable.
- `sonictok-data` isolates serialization (the only code touching file formats),
  so the core never knows where bytes came from.
- `sonictok` is the thin, ergonomic public face (tiktoken-style API). Phase 4
  bindings wrap *this* crate.
- Clear seams = each unit is independently testable and explainable.

---

## 5. The encoding pipeline (data flow)

A call to `encode(text)` flows through three stages, mirroring tiktoken/quicktok
semantics exactly (that exactness is what makes us byte-identical):

```
bytes in
   │
   ▼
┌──────────────────┐   splits input into "pieces" at grammar boundaries
│ 1. Pretokenizer  │   (contractions, word/number/punct/whitespace runs).
└──────────────────┘   cl100k & o200k each have a fixed regex; we DON'T use a
   │  pieces            regex engine — we compile the grammar by hand into a
   │  (byte spans)      SIMD scanner. Most real text is ASCII → single-pass path.
   ▼
┌──────────────────┐   each piece is BPE-merged independently. Look the whole
│ 2. BPE merge     │   piece up in the vocab first (common short pieces hit
└──────────────────┘   directly); otherwise run exact backtracking BPE using
   │  token ids        merge ranks. This is the hot inner loop.
   ▼
┌──────────────────┐   special tokens (<|endoftext|> etc.) handled per tiktoken
│ 3. Emit/specials │   semantics: encode_ordinary ignores them; encode raises on
└──────────────────┘   a stray special unless allowed_special is given.
   │
   ▼
token ids out (Vec<u32>, or written into a caller-provided buffer)
```

### 5.1 Exactness detail — the BPE algorithm

tiktoken's BPE is **exact backtracking BPE**, not the textbook "merge the
lowest-rank adjacent pair greedily once": for a byte piece, it repeatedly merges
the globally-lowest-rank adjacent pair until no merge applies, producing a
specific segmentation. `bpe-openai` and quicktok both implement this exact
procedure; we replicate it bit-for-bit. The optimization ladder only ever
changes the *data structures*, never the *result*.

### 5.2 Semantic entry points (match tiktoken)

- `encode_ordinary` — no special-token handling (specials treated as literal
  bytes).
- `encode` — raises on encountering a special token unless it is in
  `allowed_special`.
- `encode_with_special` — all specials recognized and mapped to their ids.

### 5.3 Decode

Inverse of encode: id → stored byte string, concatenated. Lossless round-trip
for valid token sequences; handles special ids too.

### 5.4 Hot-path discipline

Stages 1–3 share per-thread reusable scratch buffers (piece spans, the working
merge array). After warmup, a steady-state `encode` call performs **zero heap
allocation** unless the output `Vec` must grow (callers can pre-size or pass
their own buffer). Essential for quicktok-class numbers.

---

## 6. Data structures & the optimization ladder

Built as **measured, test-gated rungs**. Each rung keeps the exactness suite
green and must show a benchmark win to stay. Every rung is reversible and behind
a benchmark gate — complexity that doesn't pay is reverted.

- **Rung 0 — Correct baseline.** `HashMap<Vec<u8>, u32>` rank lookups, exact
  backtracking BPE, scalar pretokenizer using the `regex` crate. 100%
  byte-exact + benchmarked. ~tiktoken speed. **Kept forever as the test-only
  oracle.**
- **Rung 1 — Ranked-merge core.** tiktoken's exact merge loop over a small stack
  array; whole-piece direct lookup first (most short pieces resolve in one
  probe). Replace `HashMap` with a fast open-addressed table (`ahash`/FxHash) or
  a static perfect hash.
- **Rung 2 — 2-byte trie.** quicktok's key win: a trie keyed on 2 input bytes
  per node so the longest-match walk consumes 16 bits per 8-byte slot load, plus
  a direct table for CJK. Replicated in Rust with `#[repr(C)]` arrays and
  proven-in-range unchecked indexing on the hot path.
- **Rung 3 — Dense validity memos.** Exact-keyed caches for merge-validity
  checks (~2 MB for 17-bit cl100k ids, a wider one for o200k's 200k vocab), with
  a bijective integer mixer so there is provably zero aliasing.
- **Rung 4 — Hand-compiled SIMD pretokenizers.** The cl100k/o200k regexes
  compiled by hand into AVX2/SSE2 (x86) and NEON (aarch64) scanners — no regex
  engine on the hot path. Arch-gated via `cfg`, runtime feature detection
  (`is_x86_feature_detected!`), with a portable scalar fallback that is always
  correct.
- **Rung 5 — Single-pass ASCII product machine.** For ASCII spans (most code and
  English), one fused loop owns *both* pretokenizer boundaries *and* token
  emission — contractions, prefix-words, digit triples, punct runs, whitespace
  cascade inline, no per-piece dispatch. Unicode contact falls back to the
  general scanner one piece at a time.

### 6.1 Where we try to beat quicktok (marginal wins)

- **Static perfect hashing** of the vocab (built offline in `xtask`) → no
  probing, smaller cache footprint than open addressing.
- **PGO + `target-cpu`-tuned** release builds, with a portable build flag
  (mirrors quicktok's `-march=native` vs portable split).
- **`std::simd` portable layer** evaluated against hand intrinsics — keep
  whichever benchmarks faster per arch.
- **Aggressive prefetching** in the trie walk and memo lookups, measured.

---

## 7. Data file format & loader

### 7.1 Format — prebuilt binary blob per encoding

Rather than parsing tiktoken's text `.tiktoken` format at load time, ship a
prebuilt binary blob designed for instant load and cache-friendly layout:

```
sonictok blob (one file per encoding, e.g. data/cl100k_base.stb)
┌─────────────────────────────────────────────────────────┐
│ header:  magic, format version, encoding name, vocab     │
│          size, special-token count, content checksum,    │
│          source-reference version stamp                  │
├─────────────────────────────────────────────────────────┤
│ rank table:    token bytes + their ranks (ids)           │
│ prebuilt trie: the 2-byte trie nodes, ready to map        │
│ perfect-hash:  static hash params + displacement table   │
│ special tokens: name → id                                 │
└─────────────────────────────────────────────────────────┘
```

- **Built offline** by `xtask`/`tools`: the Python export script pulls the
  reference vocab from tiktoken; `xtask build-data` constructs the trie +
  perfect-hash and serializes the blob. Expensive structure-building happens
  once, at build time — load is just validate-and-map.
- **Versioned + checksummed:** header carries a format version and a content
  checksum; the loader rejects corrupt/mismatched files with a clear error. A
  pinned source-reference version stamp records which tiktoken/Unicode version
  it was derived from, so regeneration is reproducible.
- **Vendored in-repo** under `data/`, so builds are offline and deterministic.
  Regeneration is a documented `xtask` command, never an implicit network fetch.

### 7.2 Loading (`sonictok-data` crate)

- `Tokenizer::load_dir(dir, encoding)` and a bundled `get_encoding(name)` that
  finds vendored blobs.
- Validates header → checks checksum → zero-copy maps the rank/trie/hash regions
  into the in-memory structures the core consumes. Either `mmap` (zero-copy,
  lazy) or a single `read` into an owned buffer — chosen by benchmark + the
  thread-safety story (mmap'd read-only data is trivially `Sync`).
- **Cold path only:** all `Result`/error handling lives here. Once loaded, the
  structures handed to `sonictok-core` are immutable and `Sync` — shared across
  threads with no locks, enforced by the type system. This is the "load once,
  encode from many threads" guarantee.
- **Embedded option:** the bundled OpenAI blobs can be `include_bytes!`'d into
  the binary (feature-gated) so a deployed artifact needs no external data dir —
  good for single-binary production deploys.

---

## 8. Public API (Phase 1 Rust surface)

Lives in the `sonictok` crate, tiktoken-shaped so it is familiar and so Phase 4
bindings wrap it directly.

```rust
// Construction (cold, fallible)
Tokenizer::load_dir(dir: &Path, encoding: &str) -> Result<Tokenizer, Error>
get_encoding(name: &str) -> Result<Tokenizer, Error>   // bundled blobs

// Hot path — ordinary (specials are literal bytes)
fn encode_ordinary(&self, text: &str) -> Vec<u32>
fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>)   // zero-alloc reuse

// Hot path — special-aware (tiktoken semantics)
fn encode(&self, text: &str, allowed_special: Allowed<'_>) -> Result<Vec<u32>, EncodeError>
fn encode_with_special(&self, text: &str) -> Vec<u32>            // all specials on

// Counting (no id materialization)
fn count(&self, text: &str) -> usize

// Decode (lossless; handles special ids)
fn decode(&self, ids: &[u32]) -> Result<String, DecodeError>    // or decode_bytes -> Vec<u8>
fn decode_into(&self, ids: &[u32], out: &mut Vec<u8>)

// Batch / parallel (rayon, opt-in feature)
fn encode_batch(&self, texts: &[&str], opts: BatchOpts) -> Batch  // flat ids + offsets

// Introspection
fn vocab_size(&self) -> usize        // base vocab
fn n_vocab(&self) -> usize           // max id + 1, incl. specials
fn special_tokens(&self) -> &[(String, u32)]
fn encoding(&self) -> &str
```

Notes:

- `Allowed<'_>` is an enum (`All`, `None`, `Set(&HashSet<&str>)`) — ergonomic and
  zero-cost, matching tiktoken's `allowed_special`.
- `*_into` variants take a caller buffer for the zero-allocation steady state
  (what batch/training pipelines need).
- `Tokenizer: Send + Sync` — compiler-guaranteed safe sharing across threads.
- 4 GiB-per-call input cap (matches quicktok), surfaced as an error, not a
  panic.

---

## 9. Error handling

- **Two error families, by path.** Cold/load errors (`Error`: missing file, bad
  checksum, unknown encoding, unsupported format version) are rich, contextful
  `thiserror` enums. Hot-path errors are narrow and cheap:
  `EncodeError::DisallowedSpecial { token, offset }`,
  `DecodeError::InvalidToken(u32)`, `InputTooLarge`.
- **The hot path never panics and never allocates to report an error.**
  Disallowed-special is the only fallible encode case and it is detected without
  slowing the success path. `encode_ordinary` is infallible by construction.
- Decode of an unknown/invalid id is an explicit `Result`, never UB. Core uses
  unchecked indexing internally, but every such index is *proven* in range by
  construction (validated at load), and that invariant is documented at each
  `unsafe` site.

---

## 10. Testing strategy

Five layers — this is where "extremely accurate and well tested" gets teeth.

1. **Exactness fixtures (hermetic, always-on).** `tools/gen_fixtures.py` runs
   real `tiktoken` over a curated stress suite and freezes `input → expected
   ids` into `fixtures/` (version-stamped). The Rust suite diffs every encoding
   against these on every `cargo test` — no Python needed. The stress suite
   includes: ASCII/code/prose, all contraction forms, whitespace cascades,
   digit-triple boundaries, CJK, emoji/ZWJ sequences, combining marks,
   NUL/control bytes, lone surrogates-as-bytes, every special token (stray,
   allowed, disallowed), and adversarial near-merge inputs.
2. **Live differential testing (deep, scheduled CI).** A separate job installs
   `tiktoken`, streams large real corpora (The Pile / GitHub code / Common Crawl
   samples), encodes with both, and asserts byte-for-byte equality over millions
   of tokens.
3. **Property tests (`proptest`).** Invariants over random inputs:
   `decode(encode_ordinary(x)) == x` for valid UTF-8; encode never panics on
   arbitrary bytes; `count(x) == encode_ordinary(x).len()`; batch result equals
   per-doc encode; **rung-N output == rung-0 oracle output**.
4. **Fuzzing (`cargo fuzz`).** Continuous targets on encode (arbitrary bytes),
   decode (arbitrary id arrays), and round-trip. Crash/panic = bug.
5. **The "oracle diff" gate.** Rung 0 (naive, obviously-correct) stays in the
   codebase forever as a test-only oracle. Every optimized rung is differentially
   tested against it — proving the fast path equals the simple path, not just the
   frozen fixtures.

---

## 11. Benchmark harness

- Statistically-sound `criterion` benchmarks: single-thread MB/s per encoding
  per corpus (Pile/Code/CommonCrawl), warm and cold.
- **Comparative bench** (like quicktok's `bench-compare`): optional job pitting
  us against `tiktoken`, `bpe-openai`, `tiktoken-rs` on the same machine/corpora,
  **verifying token-for-token equality before timing** (never time a wrong
  result).
- Batch/parallel scaling curve (1→N threads).
- **Regression guard:** benchmark deltas tracked in CI; a meaningful perf
  regression on a PR is flagged. Every optimization-ladder rung must post its
  before/after number in the PR.

---

## 12. Project setup, platforms & CI

- **Stable Rust** for shipping crates (no nightly requirement for users). The
  portable-SIMD experiment (Rung 4) is evaluated on nightly but only adopted if
  expressible on stable or kept behind an optional nightly feature.
- **Two build profiles** mirroring quicktok's native/portable split: a `native`
  profile (`target-cpu=native`) and a `portable` profile (baseline
  x86-64-v2 / generic aarch64, runtime feature detection picks AVX2/SSE/NEON).
  Default published artifacts are portable-with-runtime-dispatch.
- **PGO** wired into the release build via `xtask`, measured against non-PGO.
- **`cargo-xtask`** for all automation: `xtask build-data`, `xtask gen-fixtures`,
  `xtask bench`, `xtask verify`. No scattered shell scripts.
- **Platforms (Phase 1 CI matrix):** Linux x86-64 (AVX2 + SSE2 fallback), macOS
  aarch64 (NEON), and a portable/scalar build that must pass everywhere. Windows
  supported via the portable path; full Windows CI lands with Phase 4.
- **CI gates (every PR):** `fmt` → `clippy -D warnings` → hermetic test suite
  (fixtures + proptest + oracle diff) → `cargo deny` (license/advisory audit) →
  bench smoke. Scheduled/nightly: live differential corpora + full comparative
  benchmarks + extended fuzzing.

---

## 13. Explicitly out of Phase 1 (deferred by design)

- Open-model encodings — `llama3`, `qwen3`, NFC normalization, `llama4`
  (Phase 2).
- The generic importer + pretokenizer-grammar classifier + tekken (Phase 3).
- Python wheels / `PyO3` / `maturin`, the stable C ABI / `cbindgen`, CMake
  packaging (Phase 4).
- GPU / multi-call streaming / async APIs (not planned unless a real need
  appears — YAGNI).

The architecture (cold-loader / hot-core seam, the pretokenizer *framework*, the
data-blob format with an encoding-name field) is built so each deferred piece
slots in as an extension, not a rewrite.

---

## 14. Open questions / risks

- **Beating quicktok native is genuinely hard.** It is years of hand-tuning.
  Mitigation: phased checkpoints (A → B → target); correctness never blocked on
  hitting the perf ceiling; every rung benchmark-gated.
- **Perfect-hash vs trie interplay:** the perfect hash and the 2-byte trie are
  alternative/overlapping fast paths; we'll benchmark to decide their division
  of labor (e.g. trie for longest-match walk, perfect hash for whole-piece and
  merge-rank lookups). Resolve during Rungs 1–3.
- **mmap vs read-into-buffer:** decide by benchmark + the `Sync` story; both are
  viable.
- **`std::simd` stability:** if it cannot land on stable in time, hand intrinsics
  are the shipping path; portable SIMD stays an internal experiment.
