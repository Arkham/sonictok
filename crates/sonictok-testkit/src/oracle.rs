//! Independent reference tokenizer: the encoding's regex (fancy-regex) + core BPE.
use fancy_regex::Regex;
use sonictok_core::bpe::byte_pair_encode;
use sonictok_core::rank::{Rank, RankMap};

const CL100K_PAT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const O200K_PAT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Oracle {
    re: Regex,
    ranks: RankMap,
}

impl Oracle {
    pub fn cl100k() -> Self {
        Self::from(CL100K_PAT, crate::corpora::blob("cl100k_base"))
    }
    pub fn o200k_base() -> Self {
        Self::from(O200K_PAT, crate::corpora::blob("o200k_base"))
    }

    fn from(pattern: &str, blob: sonictok_data::VocabBlob) -> Self {
        Self {
            re: Regex::new(pattern).unwrap(),
            ranks: RankMap::from_pairs(blob.ranks),
        }
    }

    /// encode_ordinary via the reference regex.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.re.find_iter(text) {
            let m = m.unwrap();
            debug_assert_eq!(m.start(), last, "regex left a gap");
            last = m.end();
            byte_pair_encode(&text.as_bytes()[m.start()..m.end()], &self.ranks, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires data/*.stb"]
    fn oracle_basic() {
        assert_eq!(
            Oracle::cl100k().encode_ordinary("hello world"),
            vec![15339, 1917]
        );
        // o200k_base: "hello world" — known tiktoken ids
        assert_eq!(
            Oracle::o200k_base().encode_ordinary("hello world"),
            vec![24912, 2375]
        );
    }
}
