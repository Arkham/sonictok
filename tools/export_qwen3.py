"""Export the Qwen2.5/3 tokenizer (HF byte-level BPE) to vendored intermediates.
    pip install tokenizers huggingface_hub
    python tools/export_qwen3.py
Writes:
    data/qwen3.tiktoken   # lines: <base64(raw_bytes)> <rank>   (rank = HF id)
    data/qwen3.special    # lines: <name> <id>
The HF vocab stores tokens in GPT-2 "byte-level unicode"; we invert that map to
recover the raw bytes (so the same byte-level BPE / tiktoken-rank backtracking
reproduces HF's output).
"""
import base64, json, os, urllib.request

URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json"


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    os.makedirs("data", exist_ok=True)
    d = json.load(urllib.request.urlopen(URL, timeout=60))
    assert d.get("normalizer", {}).get("type") == "NFC", "expected NFC normalizer"
    decoder = bytes_to_unicode()
    vocab = d["model"]["vocab"]  # token_string -> id

    with open("data/qwen3.tiktoken", "w") as f:
        for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            raw = bytes(decoder[c] for c in tok)
            f.write(f"{base64.b64encode(raw).decode()} {idx}\n")

    with open("data/qwen3.special", "w") as f:
        for t in d.get("added_tokens", []):
            f.write(f"{t['content']} {t['id']}\n")

    max_id = max(t["id"] for t in d["added_tokens"]) if d.get("added_tokens") else len(vocab) - 1
    print(f"qwen3: {len(vocab)} mergeable ranks, {len(d.get('added_tokens', []))} specials, max id {max_id}")


if __name__ == "__main__":
    main()
