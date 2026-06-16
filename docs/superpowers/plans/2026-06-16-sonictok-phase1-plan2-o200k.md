# sonictok Phase 1 — Plan 2: o200k_base + o200k_harmony

Goal: add `o200k_base` and `o200k_harmony` byte-exact, by generalizing the
engine to select a pretokenizer grammar per encoding. Same TDD + oracle-diff
discipline as Plan 1.

## Facts

- o200k_base: 199998 mergeable ranks, specials `<|endoftext|>`=199999,
  `<|endofprompt|>`=200018. n_vocab 200019.
- o200k_harmony: same mergeable ranks as o200k_base + 1091 specials (mostly
  `<|reserved_*|>`); n_vocab 201088. Same pretokenizer grammar as o200k_base.
- o200k pretokenizer regex (both encodings):

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Differences vs cl100k: two case-aware letter alternatives with optional
contraction suffix (instead of `\p{L}+`), and `[\r\n/]*` (adds `/`) in the punct
alt's trailing class.

## Tasks

1. Extend `tools/export_unicode.py` to emit `O200K_UPPER` (`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`)
   and `O200K_LOWER` (`[\p{Ll}\p{Lm}\p{Lo}\p{M}]`) range tables; regenerate.
2. Add `tools/export_o200k.py` (exports o200k_base AND o200k_harmony ranks+specials);
   `xtask build-data o200k_base` / `o200k_harmony`; vendor blobs.
3. Refactor `pretok`: introduce `Grammar { Cl100k, O200k }` + a `Scanner` enum that
   dispatches; make `Engine` hold a `Grammar` instead of a type param.
4. Add `pretok/o200k.rs` scanner (the two letter branches + contraction suffix +
   `[\r\n/]*` punct tail).
5. Update public `Tokenizer` to map encoding name -> Grammar and accept the three
   encodings; `get_encoding` finds the right blob.
6. Add o200k oracle (fancy-regex with the o200k pattern) + per-encoding fixtures +
   oracle-diff over the corpus for o200k_base.
7. All green: fixtures + oracle-diff + proptest for cl100k AND o200k.

Gate: only commit when the full exactness suite is green for every encoding.
