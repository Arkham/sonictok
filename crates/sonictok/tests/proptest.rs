use proptest::prelude::*;
use sonictok::{Tokenizer, get_encoding};
use std::sync::OnceLock;

fn tok() -> &'static Tokenizer {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    T.get_or_init(|| get_encoding("cl100k_base").expect("data/cl100k_base.stb"))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn roundtrip_utf8(s in ".{0,400}") {
        let ids = tok().encode_ordinary(&s);
        prop_assert_eq!(tok().decode(&ids).unwrap(), s);
    }

    #[test]
    fn count_equals_len(s in ".{0,400}") {
        prop_assert_eq!(tok().count(&s), tok().encode_ordinary(&s).len());
    }

    #[test]
    fn never_panics_on_arbitrary_text(s in ".{0,400}") {
        let _ = tok().encode_ordinary(&s); // must not panic
    }
}
