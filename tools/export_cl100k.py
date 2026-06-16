"""Dump tiktoken's cl100k_base vocab to vendored intermediates.
    pip install tiktoken
    python tools/export_cl100k.py
Writes:
    data/cl100k_base.tiktoken   # lines: <base64(token)> <rank>
    data/cl100k_base.special    # lines: <name> <id>
"""
import base64, os, tiktoken

os.makedirs("data", exist_ok=True)
enc = tiktoken.get_encoding("cl100k_base")

with open("data/cl100k_base.tiktoken", "w") as f:
    for token_bytes, rank in sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1]):
        f.write(f"{base64.b64encode(token_bytes).decode()} {rank}\n")

with open("data/cl100k_base.special", "w") as f:
    for name, idx in sorted(enc._special_tokens.items(), key=lambda kv: kv[1]):
        f.write(f"{name} {idx}\n")

print("max id:", enc.max_token_value, "n_vocab:", enc.n_vocab)
