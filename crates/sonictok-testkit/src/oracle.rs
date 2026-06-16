//! Independent reference tokenizer: cl100k regex (fancy-regex) + core BPE.
use fancy_regex::Regex;
use sonictok_core::bpe::byte_pair_encode;
use sonictok_core::rank::{Rank, RankMap};

const CL100K_PAT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Oracle {
    re: Regex,
    ranks: RankMap,
}

impl Oracle {
    pub fn cl100k() -> Self {
        let blob = crate::corpora::cl100k_blob();
        Self { re: Regex::new(CL100K_PAT).unwrap(), ranks: RankMap::from_pairs(blob.ranks) }
    }
    /// encode_ordinary via the reference regex.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.re.find_iter(text) {
            let m = m.unwrap();
            debug_assert_eq!(m.start(), last, "regex left a gap");
            last = m.end();
            byte_pair_encode(text[m.start()..m.end()].as_bytes(), &self.ranks, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires data/cl100k_base.stb"]
    fn oracle_basic() {
        let o = Oracle::cl100k();
        // "hello world" — known tiktoken ids
        assert_eq!(o.encode_ordinary("hello world"), vec![15339, 1917]);
    }
}
