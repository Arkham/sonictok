//! Differential exactness fuzzing: the production backtracking encoder must equal
//! the independent merge-reference (fancy-regex pretok + tiktoken-style merge) for
//! cl100k and o200k. Any divergence is an exactness bug.
#![no_main]
use libfuzzer_sys::fuzz_target;
use sonictok::{Tokenizer, get_encoding};
use sonictok_testkit::oracle::Oracle;

thread_local! {
    static REFS: (Tokenizer, Oracle, Tokenizer, Oracle) = (
        get_encoding("cl100k_base").unwrap(),
        Oracle::cl100k(),
        get_encoding("o200k_base").unwrap(),
        Oracle::o200k_base(),
    );
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // first byte selects the encoding; the rest is the (valid-UTF-8) input
    let s = match std::str::from_utf8(&data[1..]) {
        Ok(s) => s,
        Err(_) => return,
    };
    REFS.with(|(ctok, cor, otok, oor)| {
        if data[0] & 1 == 0 {
            assert_eq!(ctok.encode_ordinary(s), cor.encode_ordinary(s), "cl100k divergence: {s:?}");
        } else {
            assert_eq!(otok.encode_ordinary(s), oor.encode_ordinary(s), "o200k divergence: {s:?}");
        }
    });
});
