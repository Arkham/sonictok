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

## Final head-to-head (this M3 Pro, `bench/corpus.txt`, cl100k)

| | sonictok | quicktok | ratio |
|--|---------:|---------:|------:|
| single-thread | ~135 MB/s | 149.5 MB/s | **0.90×** |
| batch (all cores) | ~755 MB/s | 760 MB/s (8t) | **0.99×** |

Decisively faster than every other exact tokenizer (bpe-openai ~37, tiktoken ~14
MB/s). Remaining single-thread gap is fine cache/codegen tuning; the CJK
multibyte trie path (r3/encode_mb) is unimplemented (helps non-Latin, not this
corpus).

## Parallel batch (rayon, `parallel` feature, default on)

`encode_batch` over `bench/corpus.txt` split into paragraphs, M3 Pro (11 cores):

| | MiB/s | MB/s | vs single-thread |
|--|------:|-----:|-----------------:|
| sonictok `encode_batch` | ~526 | ~552 | 6.2× |
| quicktok `encode_batch` (8 thread) | ~725 | 760 | — |

Exactness-safe (per-document `encode_ordinary`, already verified). A strong,
reliable multi-core win; further gains would come from raising the single-thread
floor (the supervised structural work above).

## sonictok targets (single-thread, this machine)

- **Checkpoint A** — beat every other exact tokenizer (≥ bpe-openai class).
- **Checkpoint B** — quicktok-class: within ~15–20% (≥ ~135 MB/s cl100k).
- **Target** — beat quicktok native: **> 160.7 MB/s cl100k, > 145.5 MB/s o200k**
  single-thread, byte-exact.
