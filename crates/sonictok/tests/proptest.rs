use proptest::prelude::*;
use sonictok::{Tokenizer, get_encoding};
use std::sync::OnceLock;

fn cl100k() -> &'static Tokenizer {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    T.get_or_init(|| get_encoding("cl100k_base").expect("data/cl100k_base.stb"))
}
fn o200k() -> &'static Tokenizer {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    T.get_or_init(|| get_encoding("o200k_base").expect("data/o200k_base.stb"))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn cl100k_roundtrip(s in ".{0,400}") {
        prop_assert_eq!(cl100k().decode(&cl100k().encode_ordinary(&s)).unwrap(), s);
    }
    #[test]
    fn cl100k_count(s in ".{0,400}") {
        prop_assert_eq!(cl100k().count(&s), cl100k().encode_ordinary(&s).len());
    }
    #[test]
    fn cl100k_no_panic(s in ".{0,400}") {
        let _ = cl100k().encode_ordinary(&s);
    }

    #[test]
    fn o200k_roundtrip(s in ".{0,400}") {
        prop_assert_eq!(o200k().decode(&o200k().encode_ordinary(&s)).unwrap(), s);
    }
    #[test]
    fn o200k_count(s in ".{0,400}") {
        prop_assert_eq!(o200k().count(&s), o200k().encode_ordinary(&s).len());
    }
    #[test]
    fn o200k_no_panic(s in ".{0,400}") {
        let _ = o200k().encode_ordinary(&s);
    }
}
