//! decode(encode_ordinary(s)) == s for all valid UTF-8 (cl100k: no normalizer,
//! so the round-trip is lossless).
#![no_main]
use libfuzzer_sys::fuzz_target;
use sonictok::{Tokenizer, get_encoding};

thread_local! {
    static TOK: Tokenizer = get_encoding("cl100k_base").unwrap();
}

fuzz_target!(|data: &[u8]| {
    let s = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };
    TOK.with(|t| {
        let ids = t.encode_ordinary(s);
        assert_eq!(t.decode(&ids).unwrap(), s, "round-trip mismatch: {s:?}");
    });
});
