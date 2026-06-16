//! sonictok public API. tiktoken-shaped.
#![forbid(unsafe_code)]
pub mod error;
pub use error::{DecodeError, EncodeError, Error};

use std::collections::HashSet;
use std::path::Path;

use sonictok_core::encoding::{Decoder, Engine};
use sonictok_core::pretok::Grammar;
use sonictok_core::rank::{Rank, RankMap};
use sonictok_core::specials::SpecialTokens;
use sonictok_data::VocabBlob;

/// Map a bundled encoding name to its pretokenizer grammar.
fn grammar_for(encoding: &str) -> Option<Grammar> {
    match encoding {
        "cl100k_base" => Some(Grammar::Cl100k),
        "o200k_base" | "o200k_harmony" => Some(Grammar::O200k),
        _ => None,
    }
}

/// Which special tokens `encode` will accept without erroring.
pub enum Allowed<'a> {
    All,
    None,
    Set(&'a HashSet<&'a str>),
}

struct DenseDecoder {
    by_id: Vec<Option<Vec<u8>>>,
}
impl Decoder for DenseDecoder {
    #[inline]
    fn bytes_for(&self, id: Rank) -> Option<&[u8]> {
        self.by_id.get(id as usize).and_then(|o| o.as_deref())
    }
}

pub struct Tokenizer {
    encoding: String,
    grammar: Grammar,
    ranks: RankMap,
    decoder: DenseDecoder,
    specials: SpecialTokens,
    n_vocab: usize,
    vocab_size: usize,
}

// Tokenizer is automatically Send + Sync (all fields are Send + Sync and
// immutable after construction). Assert it at compile time — no unsafe needed.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tokenizer>();
};

impl Tokenizer {
    pub fn from_blob(blob: VocabBlob) -> Result<Self, Error> {
        let grammar =
            grammar_for(&blob.name).ok_or_else(|| Error::UnsupportedEncoding(blob.name.clone()))?;
        let vocab_size = blob.ranks.len();
        let n_vocab = (blob.max_id as usize) + 1;
        let mut by_id: Vec<Option<Vec<u8>>> = vec![None; n_vocab];
        for (bytes, id) in &blob.ranks {
            by_id[*id as usize] = Some(bytes.clone());
        }
        let ranks = RankMap::from_pairs(blob.ranks);
        let specials = SpecialTokens::new(blob.specials);
        Ok(Tokenizer {
            encoding: blob.name,
            grammar,
            ranks,
            decoder: DenseDecoder { by_id },
            specials,
            n_vocab,
            vocab_size,
        })
    }

    pub fn load_dir(dir: &Path, encoding: &str) -> Result<Self, Error> {
        if grammar_for(encoding).is_none() {
            return Err(Error::UnsupportedEncoding(encoding.to_string()));
        }
        let path = dir.join(format!("{encoding}.stb"));
        let bytes = std::fs::read(path)?;
        let blob = VocabBlob::from_bytes(&bytes)?;
        Self::from_blob(blob)
    }

    fn engine(&self) -> Engine<'_, RankMap, DenseDecoder> {
        Engine::new(&self.ranks, &self.decoder, &self.specials, self.grammar)
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.engine().encode_ordinary_into(text, &mut out);
        out
    }
    pub fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) {
        self.engine().encode_ordinary_into(text, out);
    }
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.engine().encode_with_special_into(text, &mut out);
        out
    }
    pub fn encode(&self, text: &str, allowed: Allowed<'_>) -> Result<Vec<u32>, EncodeError> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        let pred = self.allow_pred(allowed);
        self.engine()
            .encode_into(text, &pred, &mut out)
            .map_err(|e| EncodeError {
                token: String::from_utf8_lossy(&e.token).into_owned(),
                offset: e.offset,
            })?;
        Ok(out)
    }
    pub fn count(&self, text: &str) -> usize {
        self.engine().count(text)
    }

    /// Encode many documents into one flat id buffer plus offsets:
    /// document `i` is `tokens[offsets[i]..offsets[i + 1]]` (offsets has
    /// `texts.len() + 1` entries). `encode_ordinary` semantics per document.
    /// Parallel across documents with the `parallel` feature (default on);
    /// `Tokenizer` is `Sync`, so this is lock-free.
    pub fn encode_batch(&self, texts: &[&str]) -> Batch {
        let per_doc: Vec<Vec<u32>> = self.encode_each(texts);
        let mut offsets = Vec::with_capacity(texts.len() + 1);
        let total: usize = per_doc.iter().map(Vec::len).sum();
        let mut tokens = Vec::with_capacity(total);
        offsets.push(0i64);
        for ids in &per_doc {
            tokens.extend_from_slice(ids);
            offsets.push(tokens.len() as i64);
        }
        Batch { tokens, offsets }
    }

    /// Per-document token counts (for budgeting), parallel with `parallel`.
    pub fn count_batch(&self, texts: &[&str]) -> Vec<usize> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            texts.par_iter().map(|t| self.count(t)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            texts.iter().map(|t| self.count(t)).collect()
        }
    }

    #[cfg(feature = "parallel")]
    fn encode_each(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode_ordinary(t)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    fn encode_each(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode_ordinary(t)).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeError> {
        let mut out = Vec::with_capacity(ids.len() * 3);
        self.engine()
            .decode_into(ids, &mut out)
            .map_err(|e| DecodeError(e.0))?;
        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    pub fn encoding(&self) -> &str {
        &self.encoding
    }
    pub fn special_tokens(&self) -> Vec<(String, u32)> {
        self.specials
            .iter()
            .map(|(b, id)| (String::from_utf8_lossy(b).into_owned(), id))
            .collect()
    }

    fn allow_pred(&self, allowed: Allowed<'_>) -> Box<dyn Fn(Rank) -> bool + '_> {
        match allowed {
            Allowed::All => Box::new(|_: Rank| true),
            Allowed::None => Box::new(|_: Rank| false),
            Allowed::Set(set) => {
                let ids: HashSet<Rank> = self
                    .specials
                    .iter()
                    .filter(|(b, _)| set.contains(&*String::from_utf8_lossy(b)))
                    .map(|(_, id)| id)
                    .collect();
                Box::new(move |id| ids.contains(&id))
            }
        }
    }
}

/// Flat batch result: `tokens[offsets[i]..offsets[i + 1]]` is document `i`.
pub struct Batch {
    pub tokens: Vec<u32>,
    pub offsets: Vec<i64>,
}

/// Bundled-encoding lookup: finds vendored blobs under the repo `data/` dir.
pub fn get_encoding(name: &str) -> Result<Tokenizer, Error> {
    let data_dir = std::env::var("SONICTOK_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
        });
    Tokenizer::load_dir(&data_dir, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        get_encoding("cl100k_base").unwrap()
    }

    #[test]
    #[ignore = "requires data/cl100k_base.stb"]
    fn known_ids() {
        let t = tok();
        assert_eq!(t.encode_ordinary("hello world"), vec![15339, 1917]);
        assert_eq!(t.encoding(), "cl100k_base");
        assert_eq!(t.n_vocab(), 100277);
        assert_eq!(t.vocab_size(), 100256);
    }

    #[test]
    #[ignore = "requires data/cl100k_base.stb"]
    fn roundtrip() {
        let t = tok();
        let s = "The quick brown 🦊 jumps — 日本語 1234!";
        let ids = t.encode_ordinary(s);
        assert_eq!(t.decode(&ids).unwrap(), s);
    }

    #[test]
    #[ignore = "requires data/o200k_base.stb"]
    fn o200k_known_ids() {
        let t = get_encoding("o200k_base").unwrap();
        assert_eq!(t.encode_ordinary("hello world"), vec![24912, 2375]);
        assert_eq!(t.n_vocab(), 200019);
        assert_eq!(t.vocab_size(), 199998);
        let s = "camelCase don't 日本語 1234!";
        assert_eq!(t.decode(&t.encode_ordinary(s)).unwrap(), s);
    }

    #[test]
    #[ignore = "requires data/o200k_harmony.stb"]
    fn o200k_harmony_loads() {
        let t = get_encoding("o200k_harmony").unwrap();
        assert_eq!(t.encoding(), "o200k_harmony");
        assert_eq!(t.n_vocab(), 201088);
        // harmony chat specials are recognized
        let ids = t.encode_with_special("<|start|>hi<|end|>");
        assert_eq!(ids.first(), Some(&200006));
        assert_eq!(ids.last(), Some(&200007));
    }

    #[test]
    #[ignore = "requires data/cl100k_base.stb"]
    fn special_semantics() {
        let t = tok();
        // stray special -> error under Allowed::None
        assert!(t.encode("a<|endoftext|>", Allowed::None).is_err());
        // allowed -> emits id 100257
        let ids = t.encode("a<|endoftext|>", Allowed::All).unwrap();
        assert_eq!(*ids.last().unwrap(), 100257);
    }
}
