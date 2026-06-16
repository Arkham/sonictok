# sonictok

Fast, exact BPE tokenizer in Rust. Byte-identical to tiktoken.

**Phase 1, Plan 1 status:** `cl100k_base` exact + tested (Rung 0/1, pre-optimization).

```rust
let enc = sonictok::get_encoding("cl100k_base")?;
let ids = enc.encode_ordinary("hello world"); // [15339, 1917]
let text = enc.decode(&ids)?;
```

## Correctness

Every `cargo test` runs hermetic tiktoken fixtures, a full-corpus oracle-diff
(hand-written scanner vs the cl100k regex), and proptest round-trip/no-panic
invariants. The data-dependent checks need the vendored blob (committed).

## Regenerate data / fixtures

Needs `pip install tiktoken regex`:

```sh
python tools/export_unicode.py                                  # unicode tables
python tools/export_cl100k.py && cargo run -p xtask -- build-data cl100k_base
python tools/gen_fixtures.py                                    # exactness vectors
```

## Benchmark vs local quicktok

```sh
cargo run -p xtask -- bench-compare
```

Builds quicktok locally and runs both encoders on the identical corpus
(`bench/corpus.txt`). See `bench/BASELINE.md` for the target to beat.

## Layout

- `crates/sonictok-core` — dependency-free exact BPE engine (BPE, pretokenizer,
  unicode, specials).
- `crates/sonictok-data` — versioned binary vocab blob (writer/reader).
- `crates/sonictok` — public tiktoken-shaped API.
- `crates/sonictok-testkit` — regex oracle + corpus loaders (dev/test).
- `xtask` — data packing + benchmark automation.

License: MIT.
