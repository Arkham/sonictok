# Benchmark baseline — the bar sonictok must beat

We benchmark against **quicktok** built and run **locally on the dev machine**,
not against published numbers, so the comparison is apples-to-apples on the same
silicon, compiler, and corpus.

## Machine

- **CPU:** Apple M3 Pro (11 cores)
- **OS:** Darwin 25.5.0 (arm64)
- **Compiler:** Apple clang 21.0.0
- **quicktok:** v0.4.0, built with `-O3 -std=c++20 -mcpu=native`

## Corpus

`bench/corpus.txt` — Project Gutenberg *Moby-Dick* (public domain), 1,048,555
bytes (~1.05 MB). Copied verbatim from quicktok's `bench/corpus.txt` so both
encoders run the identical input. Split into 2,286 documents (paragraphs) for the
batch test.

## quicktok v0.4.0 results (target to beat)

Captured via `make bench` in a local quicktok checkout, 2026-06-16.

| Encoding | Single-thread | Mtok/s | 1t batch | 2t | 4t | 8t |
|----------|--------------:|-------:|---------:|---:|---:|---:|
| `cl100k_base` | **160.7 MB/s** | 39.88 | 155.6 | 283.3 | 502.0 | 760.9 |
| `o200k_base`  | **145.5 MB/s** | 35.85 | 143.0 | 267.9 | 483.7 | 744.2 |

(tokens: cl100k 260,151; o200k 258,380)

## How to reproduce the quicktok baseline locally

```sh
git clone --depth 1 https://github.com/dmatth1/quicktok bench/quicktok-ref
cd bench/quicktok-ref
make bench            # builds libquicktok + runs bench/bench.cpp on bench/corpus.txt
```

The `bench/quicktok-ref/` checkout is git-ignored. sonictok's own comparative
harness (Plan 2) builds quicktok the same way and runs both encoders on this
exact corpus, verifying token-for-token equality before timing.

## sonictok progress (single-thread, this machine, `bench/corpus.txt`)

Throughput is criterion's median MiB/s for `cargo bench -p sonictok --bench
encode` (cl100k, `bench/corpus.txt`). 1 MiB/s ≈ 1.049 MB/s.

| Stage | cl100k MiB/s | vs quicktok* | Notes |
|-------|------------:|------------:|-------|
| Rung 0/1 (HashMap, scalar pretok) | 35.8 | 0.23× | byte-exact baseline; matches bpe-openai |
| Rung A — FxHash rank table | 41.6 | 0.27× | +16% |
| Rung B — ASCII class fast-path | 76.8 | 0.50× | +84% (biggest win) |
| Rung C1 — ASCII char_at fast path | 80.6 | 0.53× | +5% |
| Rung C2 — reuse BPE parts scratch | 82.3 | 0.54× | +2% |
| Rung D — dense 2-byte pair table | 87.1 | 0.57× | +6% |

*vs quicktok native 153.3 MiB/s (= 160.7 MB/s).

**Net: 35.8 → 87.1 MiB/s (2.43×)**, byte-exact throughout (fixtures + oracle-diff
+ proptest green at every rung). Decisively beats every non-quicktok exact
tokenizer (bpe-openai ~37 MB/s, tiktoken ~14 MB/s). `target-cpu=native` was a
wash (hot path isn't autovectorized).

### The big one: algorithm change (matching quicktok's approach)

The early ladder optimized tiktoken's O(n·merges) iterative merge. quicktok (and
bpe-openai) use the **`bpe` crate's linear-time algorithm**: greedy longest-match
via a trie + memoized `is_valid_token_pair` + rare backtracking. Switching to it
(plus the supporting data structures) is what closed most of the gap:

| Stage | cl100k MiB/s | vs quicktok | note |
|-------|------------:|------------:|------|
| merge-based ladder (Rungs A–D) | 99 | 0.65× | tiktoken algorithm, tuned |
| + bpe-crate algorithm + 2-byte trie | 104 | 0.68× | right algorithm |
| + packed single-u64 e2 (mixer+tag) | 110 | 0.72× | 1 load/2 bytes |
| + fused cl100k product machine | 121 | 0.79× | pretok+emit single pass |
| + inline hot path, e2 load 0.45 | 126 | 0.83× | |
| + reuse backtracking scratch | **~129** | **~0.90×** | no per-piece alloc |

**Net: 35.8 → ~129 MiB/s (3.6×), ~90% of quicktok-native single-thread**, byte-
exact throughout (fixtures + full-corpus oracle-diff vs an independent
merge-reference, both cl100k + o200k).

Reverted (measured, didn't pay): hashmap `(id,id)` memo (cache thrash), combined
r2 array (wash), u16 narrow memo (branch overhead > cache win), `target-cpu=native`.

## Session 2 — overtaking quicktok (autoresearch pass, 2026-06-17)

Picking up from ~129 MiB/s (~0.90× quicktok), a systematic autoresearch pass
closed the gap and **passed quicktok native on both encodings**, byte-exact.
Every kept change was confirmed with an **interleaved A/B** (see methodology
below); fixtures + full-corpus oracle-diff + proptest stayed green throughout.

Fresh re-measurement of the references on this machine (2026-06-17), so the
numbers below are all same-silicon/same-corpus:

| ref (cl100k / o200k single-thread) | cl100k | o200k |
|------------------------------------|-------:|------:|
| quicktok v0.4.0 (`-mcpu=native`, best-of-7) | 160.4 MB/s | 144.1 MB/s |
| tiktoken 0.13 (Python, `encode_ordinary`)   |  19.2 MB/s |  30.9 MB/s |

### Kept changes (cl100k, single-thread, cumulative)

MB/s is best-of-min over `bench/corpus.txt`; Δ is the per-step interleaved-A/B
result (paired rounds, win-rate). All byte-exact.

| Step | cl100k MB/s | Δ (interleaved) | What |
|------|------------:|----------------:|------|
| session-2 baseline | ~155 | — | end of Session 1 (elision-free trie walk) |
| + bounds-check elision | ~157 | -1.7% | `get_unchecked` on computed-index hot loads in `next_match`, `encode_with_first` greedy, `is_valid` memo, `odd_lookup` |
| + per-piece slice elision | ~157 | -0.6% | `enc_piece` passes `&t[p..end]` unchecked (mirrors quicktok's raw ptr,len), ~228K pieces/pass |
| + e2 load factor 0.45→0.225 | ~168 | **-6.0%** (12/12) | bigger 2-byte-trie probe table → shorter probe chains |
| + e2 load factor 0.225→0.11 | ~173 | -2.3% (11/12) | further; o200k also +5% |
| + otab load factor 0.5→0.11 | ~183 | **-8.65%** (14/14) | the odd-depth lookup is hit at the end of *most* `next_match` walks — a hidden bottleneck |
| + mix36 → pure odd-multiply | ~187 | -1.3% (11/14) | drop the xorshift/xorshift (still bijective mod 2^36 → index+tag reconstruct the key exactly) |
| + drop dead `& 2^36` mask | **~190** | -0.8% (8/12) | callers read only bits [0,36); the mask was a no-op |

**Net Session 2: ~155 → ~190 MB/s best-of-min (~182 MB/s criterion median),
-19–21% wall time.** The two probe-table load-factor cuts (e2, otab) are the
bulk of it.

### Reverted (measured, didn't pay — don't retry on M3)

| Idea | Result | Why |
|------|--------|-----|
| `target-cpu=native` | wash | hot path not autovectorized; also breaks wheel/C-ABI portability |
| u16 dense is_valid memo (4MB→2MB) | **-1.7%** | M3 caches are big — footprint reduction doesn't pay; sub-word atomics cost |
| combine r2node+r2best → one u64 | wash | r2 is L1-hot; M3's load ports do the 2 loads in parallel |
| `#[inline(always)]` next_match | **+12%** | flips LLVM's hot-loop codegen (quicktok warns of exactly this) |
| `#[inline(always)]` encode_with_first | +1.5% | same family |
| single-token-piece shortcut | +4.4% | added branch bloats the fused machine, disrupts codegen |
| 256-byte byte-class LUT | +3% | M3 prefers cheap ALU over an L1 load on the scan critical path |
| first-token peel / local hoists | wash | LLVM fat-LTO already does these |
| otab/plk load factor below 0.11 | wash | past the knee; pure footprint for ~0 gain (and would overfit the cache) |

**Theme:** control-flow / inlining changes lost **6/6**; every win was a data-
structure or per-op-cost change. The driving insight came from *inverting* the
failed u16-memo experiment — "M3 has big caches, so don't shrink footprint or
trade ALU for memory." The symmetric move (grow the hot linear-probe tables to
shorten chains; cheapen the per-step hash) produced every subsequent win.

### Profile shift (samply, % of encode)

| function | before | after |
|----------|-------:|------:|
| `Vocab::next_match` (trie walk) | 51% | 41% |
| `encode_cl100k_fused` (pretok)  | 29% | 35% |
| `encode_with_first` (greedy)    | 16% | 20% |
| `ivtp_slow` (is_valid miss)     | <1% | ~1% |

The probe-table work pulled the trie walk from dominant to roughly on par with
pretok (which is now ~58% of encode at 311 MB/s `pretokenize_only`).

### Measurement methodology (hard-won)

- Variance on this M3 Pro is **contention / P-vs-E core migration**, NOT thermal
  (idle at 47°C) and NOT corpus size (1× had lower CV than 4×). `taskpolicy -a`
  is neutral; `-b` is 6× slower (E-cores) and *less* consistent.
- Cross-run drift is ~3-8%; the **only trustworthy comparison is an interleaved
  A/B** — build both binaries, alternate base/opt for N rounds, compare paired
  min/median + win-rate. Sequential A/B (incl. `cargo bench --baseline` run after
  the other) is order-confounded and can show the wrong sign.
- Primary metric: criterion median + a fast best-of/min harness
  (`examples/perfbench.rs`, with a median/min CV-gated retry).

## Final head-to-head (this M3 Pro, `bench/corpus.txt`)

sonictok = `cargo bench` criterion median; quicktok = local `-mcpu=native`
best-of-7; tiktoken 0.13 via Python. **Token ids are identical across all three.**

| | sonictok | quicktok native | tiktoken | vs quicktok | vs tiktoken |
|--|---------:|----------------:|---------:|------------:|------------:|
| cl100k single-thread | **181.9 MB/s** (~190 best-of) | 160.4 MB/s | 19.2 MB/s | **+13%** (med) / +18% (min) | **~9.5×** |
| o200k single-thread  | **163.9 MB/s** | 144.1 MB/s | 30.9 MB/s | **+14%** | **~5.3×** |
| cl100k batch (all cores) | **~914 MB/s** | 782.8 MB/s (8t) | — | +17% | — |

sonictok is now the fastest exact tokenizer measured here — ahead of quicktok
(the fastest C++ exact tokenizer), ~5–10× tiktoken, and far ahead of bpe-openai
(~37 MB/s). The CJK multibyte trie path (r3/encode_mb) is still unimplemented
(helps non-Latin, not this corpus) — the one remaining structural lever.

## sonictok targets — ACHIEVED

- **Checkpoint A** — beat every other exact tokenizer (≥ bpe-openai class). ✅
- **Checkpoint B** — quicktok-class (within ~15–20%). ✅
- **Target** — beat quicktok native (> 160.4 MB/s cl100k, > 144.1 MB/s o200k
  single-thread, byte-exact). ✅ **181.9 / 163.9 MB/s.**
