//! decode() on arbitrary token-id arrays must return a Result (out-of-range ids
//! -> InvalidToken), never panic or read out of bounds.
#![no_main]
use libfuzzer_sys::fuzz_target;
use sonictok::{Tokenizer, get_encoding};

thread_local! {
    static TOK: Tokenizer = get_encoding("cl100k_base").unwrap();
}

fuzz_target!(|data: &[u8]| {
    let ids: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    TOK.with(|t| {
        let _ = t.decode(&ids);
        let _ = t.decode_bytes(&ids);
    });
});
