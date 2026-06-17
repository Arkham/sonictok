# sonictok

Fast, exact BPE tokenizer in Rust. Byte-identical to tiktoken.

**Status:** five encodings, all byte-exact and fully tested — `cl100k_base`,
`o200k_base`, `o200k_harmony` (vs tiktoken), `qwen3` (vs HF tokenizers, incl. NFC
normalization), and `llama3` (vs HF tokenizers). Linear-time backtracking BPE
(the `bpe`-crate algorithm) + 2-byte trie + fused product machines. Rust API,
parallel batch, a stable C ABI, and Python bindings. **cl100k ~182 MB/s
single-thread, ~914 MB/s batch (M3 Pro) — faster than quicktok-native and ~10x
tiktoken, byte-exact** — see `bench/BASELINE.md`.

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
enc = sonictok.get_encoding("cl100k_base")  # or o200k_base/o200k_harmony/qwen3/llama3
enc.encode_ordinary("hello world")          # [15339, 1917]
enc.encode("a<|endoftext|>", allowed_special="all")
enc.encode_batch(["doc one", "doc two"])    # parallel, GIL released
```

Regenerate qwen3/llama3 data + fixtures (needs `pip install tokenizers`):
```
python tools/export_qwen3.py  && cargo run -p xtask -- build-data qwen3
python tools/export_llama3.py && cargo run -p xtask -- build-data llama3
python tools/gen_fixtures_qwen3.py && python tools/gen_fixtures_llama3.py
```

### Importing other tokenizers

Any byte-level-BPE tokenizer whose pre-tokenizer matches one of sonictok's
hand-compiled grammars (cl100k / qwen / o200k) and whose normalizer is none or
NFC can be imported — the blob is self-describing (it carries its grammar +
normalizer), so no Rust changes are needed:

```sh
python tools/import_tokenizer.py Qwen/Qwen2.5-1.5B myqwen   # HF repo id, URL, or local path
# then, with SONICTOK_DATA pointing at data/:  sonictok.get_encoding("myqwen")
```

The importer classifies the pre-tokenizer regex, converts the vocab, packs the
blob, and **verifies token-for-token against the reference HF tokenizer** (a
mismatch fails the import). An unrecognized pattern or normalizer is refused with
the reason printed — there is no fallback regex engine.

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

## Benchmarks

Single-thread `encode_ordinary` over `bench/corpus.txt` (Project Gutenberg
*Moby-Dick*, 1.05 MB), Apple M3 Pro. sonictok is the project's own `cargo bench`
median; quicktok is built and run locally (`-O3 -mcpu=native`, best-of-7);
tiktoken 0.13 via Python. Token ids are **identical** across all three.

| encoding | sonictok | quicktok native | tiktoken | vs quicktok | vs tiktoken |
|----------|---------:|----------------:|---------:|------------:|------------:|
| `cl100k_base` | **181.9 MB/s** | 160.4 MB/s | 19.2 MB/s | **+13%** | **~9.5x** |
| `o200k_base`  | **163.9 MB/s** | 144.1 MB/s | 30.9 MB/s | **+14%** | **~5.3x** |

Batch (`encode_batch`, all cores): **~914 MB/s** cl100k. Best-of single-thread
hits ~190 MB/s cl100k; the table reports the steady-state criterion median.

Reproduce:

```sh
cargo bench -p sonictok --bench encode                 # criterion: cl100k/o200k + batch
cargo run --release -p sonictok --example perfbench     # fast best-of/median harness
cargo run -p xtask -- bench-compare                     # builds quicktok, diffs + times both
```

See `bench/BASELINE.md` for methodology and the full history.

## Layout

- `crates/sonictok-core` — dependency-free exact BPE engine (BPE, pretokenizer,
  unicode, specials).
- `crates/sonictok-data` — versioned binary vocab blob (writer/reader).
- `crates/sonictok` — public tiktoken-shaped API.
- `crates/sonictok-testkit` — regex oracle + corpus loaders (dev/test).
- `xtask` — data packing + benchmark automation.

License: MIT.
