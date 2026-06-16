"""Freeze tiktoken expected ids over a stress suite into fixtures/.
    pip install tiktoken
    python tools/gen_fixtures.py
"""
import base64, json, hashlib, datetime, importlib.metadata as md, tiktoken

ENC = "cl100k_base"
enc = tiktoken.get_encoding(ENC)

STRESS = [
    "", " ", "  ", "   ", "\n", "\n\n", "\t  \n", "a", "A", "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "I'm don't won't he'll they've we're it's",
    "1 12 123 1234 12345 007",
    "snake_case camelCase kebab-case CONSTANT_CASE",
    "def f(x): return x+1  # comment",
    "https://example.com/path?q=1&y=2#frag",
    "日本語のテキスト 中文文本 한국어 텍스트",
    "emoji 🦊🚀👩‍👩‍👧‍👦 family ZWJ",
    "café naïve résumé Москва Ω≈ç√∫",
    "trailing spaces here     ",
    "     leading spaces",
    "mixed\twhitespace\n\nand   newlines\r\n",
    "<|endoftext|> in the middle <|endofprompt|>",
    "no special but looks like <| not a token |>",
    "a" * 1000,
    "ab" * 500,
    "0123456789" * 50,
]


def rec(text, mode):
    is_utf8 = True
    try:
        text.encode("utf-8")
    except Exception:
        is_utf8 = False
    ids = enc.encode_ordinary(text) if mode == "ordinary" else enc.encode(text, allowed_special="all")
    out = {"encoding": ENC, "mode": mode, "ids": ids}
    if is_utf8:
        out["input"] = text
    else:
        out["input_b64"] = base64.b64encode(text.encode("utf-8", "surrogatepass")).decode()
    return out


records = [rec(t, "ordinary") for t in STRESS]
records += [rec(t, "with_special") for t in STRESS if "<|" in t]

with open(f"fixtures/{ENC}.json", "w") as f:
    json.dump(records, f, ensure_ascii=False, indent=0)

manifest = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tiktoken_version": md.version("tiktoken"),
    "encodings": [ENC],
    "content_sha256": hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest(),
}
with open("fixtures/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"wrote fixtures/{ENC}.json ({len(records)} records) + manifest.json")
