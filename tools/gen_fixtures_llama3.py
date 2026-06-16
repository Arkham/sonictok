"""Freeze HF-tokenizers expected ids for llama3 into fixtures/llama3.json.
Reference = the Llama-3 tokenizer.json (no normalizer; cl100k grammar).
    pip install tokenizers
    python tools/gen_fixtures_llama3.py
"""
import json, hashlib, datetime, urllib.request
from tokenizers import Tokenizer

URL = "https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/main/tokenizer.json"

STRESS = [
    "", " ", "  ", "\n\n", "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "I'm don't won't he'll they've we're it's",
    "1 12 123 1234 12345 007",
    "snake_case camelCase kebab-case CONSTANT_CASE HTTPSConnection",
    "def f(x): return x+1  # comment",
    "https://example.com/path?q=1&y=2#frag",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u041c\u043e\u0441\u043a\u0432\u0430",
    "trailing spaces here     ",
    "     leading spaces",
    "mixed\twhitespace\n\nand   newlines\r\n",
    "<|begin_of_text|> hi <|eot_id|>",
    "a" * 500,
    "ab" * 300,
]


def main():
    tk = Tokenizer.from_str(urllib.request.urlopen(URL, timeout=60).read().decode())

    def rec(text):
        ids = tk.encode(text, add_special_tokens=False).ids
        mode = "with_special" if "<|" in text else "ordinary"
        return {"mode": mode, "ids": ids, "encoding": "llama3", "input": text}

    records = [rec(t) for t in STRESS]
    with open("fixtures/llama3.json", "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=0)
    manifest = json.load(open("fixtures/manifest.json"))
    if "llama3" not in manifest.get("encodings", []):
        manifest.setdefault("encodings", []).append("llama3")
    manifest["llama3_generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["llama3_content_sha256"] = hashlib.sha256(
        json.dumps(records, ensure_ascii=False).encode()
    ).hexdigest()
    json.dump(manifest, open("fixtures/manifest.json", "w"), indent=2)
    print(f"wrote fixtures/llama3.json ({len(records)} records)")


if __name__ == "__main__":
    main()
