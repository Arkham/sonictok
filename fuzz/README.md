# sonictok fuzz targets

libFuzzer targets (cargo-fuzz, nightly). Run from the repo root:

```sh
cargo +nightly fuzz run <target> -- -max_total_time=60
```

| target | what it checks |
|--------|----------------|
| `oracle_diff` | **differential exactness** — `encode_ordinary` (production backtracking engine + trie) equals the independent merge-reference (`testkit::oracle`: fancy-regex pretok + tiktoken-style merge) for cl100k & o200k. Any divergence = exactness bug. |
| `encode_no_panic` | every encode path (`encode_ordinary` / `encode` / `encode_with_special` / `count`) across all 5 encodings never panics on arbitrary UTF-8. Run under ASan/UBSan it also validates pretok's one `unsafe` (`char_at`'s `from_utf8_unchecked`). |
| `decode_no_panic` | `decode` / `decode_bytes` on arbitrary `u32` id arrays return a `Result` (out-of-range → `InvalidToken`), never panic / read OOB. |
| `roundtrip` | `decode(encode_ordinary(s)) == s` for all valid UTF-8 (cl100k has no normalizer, so the round-trip is lossless). |

Inputs go through the public `&str` API (the real contract) — we deliberately do
not feed arbitrary bytes straight into the core scanner, whose documented
precondition is valid UTF-8.

On a crash, cargo-fuzz writes a reproducer to `fuzz/artifacts/<target>/`; the
workflow is to add that input as a checked-in regression test, then fix.

Last local run (Apple M3 Pro): all four clean — oracle_diff 879k runs,
encode_no_panic 293k (ASan/UBSan), decode_no_panic 11.7M, roundtrip 2.6M.
