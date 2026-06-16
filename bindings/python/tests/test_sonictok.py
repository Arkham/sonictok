"""Hermetic smoke tests for the sonictok Python bindings.

Build + install first:  cd bindings/python && maturin develop --release
Run from the repo root so the bundled data/ dir is found (or set SONICTOK_DATA).
    python -m pytest bindings/python/tests
"""
import sonictok


def test_cl100k_known_ids():
    enc = sonictok.get_encoding("cl100k_base")
    assert enc.name == "cl100k_base"
    assert enc.n_vocab == 100277
    assert enc.encode_ordinary("hello world") == [15339, 1917]
    assert enc.count("hello world") == 2


def test_o200k_known_ids():
    enc = sonictok.get_encoding("o200k_base")
    assert enc.n_vocab == 200019
    assert enc.encode_ordinary("hello world") == [24912, 2375]


def test_roundtrip_unicode():
    enc = sonictok.get_encoding("cl100k_base")
    for s in ["", "hello", "日本語 café 🦊", "def f(x): return x+1", "a" * 200]:
        assert enc.decode(enc.encode_ordinary(s)) == s


def test_batch_matches_per_doc():
    enc = sonictok.get_encoding("o200k_base")
    docs = ["one two three", "camelCase", "", "1234 numbers", "tail   "]
    assert enc.encode_batch(docs) == [enc.encode_ordinary(d) for d in docs]


def test_special_semantics():
    enc = sonictok.get_encoding("cl100k_base")
    # stray special raises unless allowed
    raised = False
    try:
        enc.encode("a<|endoftext|>")
    except ValueError:
        raised = True
    assert raised
    assert enc.encode("a<|endoftext|>", allowed_special="all")[-1] == 100257
    assert enc.encode_with_special("<|endoftext|>") == [100257]


def test_harmony_specials():
    enc = sonictok.get_encoding("o200k_harmony")
    assert enc.n_vocab == 201088
    ids = enc.encode_with_special("<|start|>hi<|end|>")
    assert ids[0] == 200006 and ids[-1] == 200007
