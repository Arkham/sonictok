//! Every encode path, on arbitrary valid-UTF-8 input, across all encodings, must
//! never panic. Run under ASan/UBSan, this also validates pretok's `char_at`
//! `unsafe` (the only unsafe in the engine).
#![no_main]
use libfuzzer_sys::fuzz_target;
use sonictok::{Allowed, Tokenizer, get_encoding};

const NAMES: [&str; 5] = ["cl100k_base", "o200k_base", "o200k_harmony", "qwen3", "llama3"];

thread_local! {
    static TOKS: Vec<Tokenizer> = NAMES.iter().map(|n| get_encoding(n).unwrap()).collect();
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let s = match std::str::from_utf8(&data[1..]) {
        Ok(s) => s,
        Err(_) => return,
    };
    TOKS.with(|toks| {
        let t = &toks[(data[0] as usize) % toks.len()];
        let _ = t.encode_ordinary(s);
        let _ = t.encode_with_special(s);
        let _ = t.encode(s, Allowed::All);
        let _ = t.encode(s, Allowed::None);
        let _ = t.count(s);
    });
});
