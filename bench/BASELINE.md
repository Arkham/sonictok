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

| Stage | cl100k MB/s | vs quicktok | Notes |
|-------|------------:|------------:|-------|
| Rung 0/1 (HashMap, scalar pretok) | ~35.8 | ~0.22× | byte-exact; matches bpe-openai, ~2.5× tiktoken |

Captured via `cargo bench -p sonictok --bench encode` (Plan 1 complete). The
optimization ladder (Plans 3+) climbs from here toward the target below.

## sonictok targets (single-thread, this machine)

- **Checkpoint A** — beat every other exact tokenizer (≥ bpe-openai class).
- **Checkpoint B** — quicktok-class: within ~15–20% (≥ ~135 MB/s cl100k).
- **Target** — beat quicktok native: **> 160.7 MB/s cl100k, > 145.5 MB/s o200k**
  single-thread, byte-exact.
