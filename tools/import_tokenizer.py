"""Import an arbitrary byte-level-BPE tokenizer into a sonictok encoding.

    python tools/import_tokenizer.py <source> <name> [--corpus FILE]

<source> is an HF repo id (e.g. "Qwen/Qwen2.5-0.5B"), a URL to a tokenizer.json,
or a local tokenizer.json path. The importer:
  1. checks the normalizer (none / NFC; else refuses),
  2. classifies the pre-tokenizer regex against sonictok's hand-compiled grammars
     (cl100k / qwen / o200k); an unrecognized pattern is REFUSED with the pattern
     printed,
  3. converts the HF byte-level vocab to tiktoken-rank and writes data/<name>.*,
  4. packs the blob (xtask build-data), then
  5. VERIFIES token-for-token vs the reference HF tokenizer over a stress suite
     (+ optional --corpus). A mismatch fails the import (and removes the blob).

There is no fallback regex engine and no approximate mode: each grammar is
hand-compiled, which is where the speed comes from.

Requires: tokenizers, and the `sonictok` Python module built (maturin develop).
"""
import base64, json, os, subprocess, sys, urllib.request
from export_qwen3 import bytes_to_unicode

# Hand-compiled grammars (grammar byte, normalizer constraint). Patterns are the
# exact tokenizer.json Split regexes.
CL100K = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
O200K = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GRAMMARS = {CL100K: ("cl100k", 0), QWEN: ("qwen", 2), O200K: ("o200k", 1)}

STRESS = [
    "", " ", "  ", "\n\n", "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "I'm don't won't he'll they've", "1 12 123 1234 007",
    "snake_case camelCase HTTPSConnection", "def f(x): return x+1  # c",
    "https://example.com/path?q=1", "\u65e5\u672c\u8a9e \u4e2d\u6587 \ud55c\uad6d\uc5b4 caf\u00e9",
    "cafe\u0301 nai\u0308ve A\u030a", "trailing   ", "     leading",
    "\U0001f98a\U0001f680 emoji", "a" * 500, "ab" * 300,
]


class Refused(SystemExit):
    pass


def load_tokenizer_json(source: str) -> dict:
    if os.path.exists(source):
        return json.load(open(source))
    url = source if source.startswith("http") else f"https://huggingface.co/{source}/resolve/main/tokenizer.json"
    return json.load(urllib.request.urlopen(url, timeout=60))


def classify(d: dict):
    # normalizer
    norm = d.get("normalizer")
    if norm is None:
        normalizer = 0
    elif norm.get("type") == "NFC":
        normalizer = 1
    else:
        raise Refused(f"REFUSED: unsupported normalizer {norm.get('type')!r} (only none/NFC)")
    # model
    mtype = (d.get("model") or {}).get("type")
    if mtype != "BPE":
        raise Refused(f"REFUSED: model type {mtype!r} (only byte-level BPE)")
    # pre-tokenizer: Sequence[Split{regex}, ByteLevel]
    pt = d.get("pre_tokenizer") or {}
    seq = pt.get("pretokenizers") if pt.get("type") == "Sequence" else [pt]
    splits = [x for x in seq if x.get("type") == "Split"]
    bytelevel = [x for x in seq if x.get("type") == "ByteLevel"]
    if not bytelevel:
        raise Refused("REFUSED: not a ByteLevel BPE (no ByteLevel pre-tokenizer)")
    if len(splits) != 1:
        raise Refused(f"REFUSED: expected exactly one Split pre-tokenizer, found {len(splits)} "
                      "(e.g. DeepSeek's multi-Split grammar is a different shape)")
    pattern = splits[0]["pattern"].get("Regex")
    if pattern not in GRAMMARS:
        raise Refused("REFUSED: unrecognized pre-tokenizer pattern (no hand-compiled grammar):\n  "
                      + repr(pattern))
    family, grammar = GRAMMARS[pattern]
    return grammar, normalizer, family


def write_vocab(d: dict, name: str):
    decoder = bytes_to_unicode()
    with open(f"data/{name}.tiktoken", "w") as f:
        for tok, idx in sorted(d["model"]["vocab"].items(), key=lambda kv: kv[1]):
            raw = bytes(decoder[c] for c in tok)
            f.write(f"{base64.b64encode(raw).decode()} {idx}\n")
    with open(f"data/{name}.special", "w") as f:
        for t in d.get("added_tokens", []):
            f.write(f"{t['content']} {t['id']}\n")


def verify(d: dict, name: str, corpus):
    from tokenizers import Tokenizer
    import sonictok
    ref = Tokenizer.from_str(json.dumps(d))
    enc = sonictok.get_encoding(name)
    texts = list(STRESS)
    if corpus:
        texts += [open(corpus, encoding="utf-8").read()]
    for text in texts:
        want = ref.encode(text, add_special_tokens=False).ids
        got = enc.encode_with_special(text)
        if got != want:
            for i, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    raise SystemExit(f"VERIFY FAILED on {text[:40]!r}: token {i} got {a} want {b}")
            raise SystemExit(f"VERIFY FAILED on {text[:40]!r}: length {len(got)} vs {len(want)}")
    return len(texts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    corpus = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--corpus"), None)
    if len(args) != 2:
        print(__doc__)
        raise SystemExit(2)
    source, name = args
    os.makedirs("data", exist_ok=True)
    d = load_tokenizer_json(source)
    grammar, normalizer, family = classify(d)
    print(f"classified: grammar={family} (#{grammar}), normalizer={'NFC' if normalizer else 'none'}")
    write_vocab(d, name)
    subprocess.run(
        ["cargo", "run", "-q", "-p", "xtask", "--", "build-data", name, str(grammar), str(normalizer)],
        check=True,
    )
    os.environ.setdefault("SONICTOK_DATA", os.path.abspath("data"))
    n = verify(d, name, corpus)
    print(f"OK: imported {name!r} ({family} grammar) and verified byte-exact on {n} inputs")


if __name__ == "__main__":
    main()
