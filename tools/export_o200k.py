"""Dump tiktoken's o200k_base and o200k_harmony vocabs to vendored intermediates.
    pip install tiktoken
    python tools/export_o200k.py
Writes for each encoding:
    data/<enc>.tiktoken   # lines: <base64(token)> <rank>
    data/<enc>.special    # lines: <name> <id>
"""
import base64, os, tiktoken

os.makedirs("data", exist_ok=True)

for name in ["o200k_base", "o200k_harmony"]:
    enc = tiktoken.get_encoding(name)
    with open(f"data/{name}.tiktoken", "w") as f:
        for token_bytes, rank in sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(token_bytes).decode()} {rank}\n")
    with open(f"data/{name}.special", "w") as f:
        # NOTE: o200k_harmony has duplicate ids (e.g. 200018 maps to both
        # <|endofprompt|> and <|reserved_200018|>); keep the first by id then name.
        for sname, idx in sorted(enc._special_tokens.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(f"{sname} {idx}\n")
    print(f"{name}: max id {enc.max_token_value}, n_vocab {enc.n_vocab}, "
          f"{len(enc._mergeable_ranks)} ranks, {len(enc._special_tokens)} specials")
