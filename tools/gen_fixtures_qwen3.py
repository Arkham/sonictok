"""Freeze HF-tokenizers expected ids for qwen3 into fixtures/qwen3.json.
Reference = the actual Qwen tokenizer.json (NFC normalizer + byte-level BPE).
    pip install tokenizers
    python tools/gen_fixtures_qwen3.py
"""
import base64, json, hashlib, datetime, urllib.request
from tokenizers import Tokenizer

URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json"

STRESS = [
    "", " ", "  ", "\n\n", "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "I'm don't won't he'll they've",
    "1 12 123 1234 007 42",  # single-digit grammar: each digit separate
    "snake_case camelCase HTTPSConnection",
    "def f(x): return x+1  # comment",
    "https://example.com/path?q=1",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 \u4e2d\u6587\u6587\u672c \ud55c\uad6d\uc5b4",
    "emoji \U0001f98a\U0001f680 family",
    # non-NFC inputs (decomposed) -> must be normalized to match HF
    "cafe\u0301 nai\u0308ve",        # e + combining acute, i + combining diaeresis
    "A\u030a \u1e69 \u0041\u0301",  # ring above; s+dot variants; A+acute
    "\u00c5ngstr\u00f6m",           # precomposed (already NFC)
    "trailing spaces   ",
    "<|endoftext|> mid <|im_start|>",
    "a" * 500,
]


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    tk = Tokenizer.from_str(raw.decode())

    def rec(text):
        # HF always recognizes added special tokens in the text (add_special_tokens
        # only controls auto BOS/EOS). So a piece containing a special maps to
        # encode_with_special semantics; otherwise to encode_ordinary.
        ids = tk.encode(text, add_special_tokens=False).ids
        mode = "with_special" if "<|" in text else "ordinary"
        return {"mode": mode, "ids": ids, "encoding": "qwen3", "input": text}

    records = [rec(t) for t in STRESS]
    with open("fixtures/qwen3.json", "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=0)

    # update the manifest's qwen3 hash entry alongside the tiktoken ones
    try:
        manifest = json.load(open("fixtures/manifest.json"))
    except FileNotFoundError:
        manifest = {}
    manifest.setdefault("encodings", [])
    if "qwen3" not in manifest["encodings"]:
        manifest["encodings"].append("qwen3")
    manifest["qwen3_generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["qwen3_content_sha256"] = hashlib.sha256(
        json.dumps(records, ensure_ascii=False).encode()
    ).hexdigest()
    json.dump(manifest, open("fixtures/manifest.json", "w"), indent=2)
    print(f"wrote fixtures/qwen3.json ({len(records)} records)")


if __name__ == "__main__":
    main()
