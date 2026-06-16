"""Export the Llama-3 tokenizer (HF byte-level BPE) to vendored intermediates,
from an ungated community mirror (Meta's repo is gated).
    pip install tokenizers
    python tools/export_llama3.py
Writes data/llama3.tiktoken + data/llama3.special. Grammar == cl100k, no
normalizer. (Known: Meta's original tiktoken-rank vs HF's merge-list agree on
~99.9998% of tokens; we build from / verify against HF here.)
"""
import base64, json, os, urllib.request
from export_qwen3 import bytes_to_unicode  # reuse the GPT-2 byte map inverter

URL = "https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/main/tokenizer.json"


def main():
    os.makedirs("data", exist_ok=True)
    d = json.load(urllib.request.urlopen(URL, timeout=60))
    assert d.get("normalizer") is None, "expected no normalizer for llama3"
    decoder = bytes_to_unicode()
    vocab = d["model"]["vocab"]
    with open("data/llama3.tiktoken", "w") as f:
        for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            raw = bytes(decoder[c] for c in tok)
            f.write(f"{base64.b64encode(raw).decode()} {idx}\n")
    with open("data/llama3.special", "w") as f:
        for t in d.get("added_tokens", []):
            f.write(f"{t['content']} {t['id']}\n")
    max_id = max(t["id"] for t in d["added_tokens"]) if d.get("added_tokens") else len(vocab) - 1
    print(f"llama3: {len(vocab)} ranks, {len(d.get('added_tokens', []))} specials, max id {max_id}")


if __name__ == "__main__":
    main()
