# sonictok

Fast, exact BPE tokenizer in Rust. Byte-identical to tiktoken.

**Status:** `cl100k_base`, `o200k_base`, `o200k_harmony` — all byte-exact vs
tiktoken and fully tested. Rust API, parallel batch, a stable C ABI, and Python
bindings. ~87 MiB/s single-thread, ~526 MiB/s batch (M3 Pro); see
`bench/BASELINE.md`.

### Rust

```rust
let enc = sonictok::get_encoding("cl100k_base")?;
let ids = enc.encode_ordinary("hello world"); // [15339, 1917]
let text = enc.decode(&ids)?;
let batch = enc.encode_batch(&["doc one", "doc two"]); // parallel (rayon)
```

### Python (tiktoken-style)

```sh
cd bindings/python && maturin develop --release   # build + install into venv
```
```python
import sonictok
enc = sonictok.get_encoding("cl100k_base")
enc.encode_ordinary("hello world")          # [15339, 1917]
enc.encode("a<|endoftext|>", allowed_special="all")
enc.encode_batch(["doc one", "doc two"])    # parallel, GIL released
```

### C / any language (stable ABI)

`crates/sonictok-cabi` builds `libsonictok.{a,dylib}` + `include/sonictok.h`.
Smoke-test it: `cargo run -p xtask -- test-cabi`.

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
