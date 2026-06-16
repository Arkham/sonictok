//! sonictok public API. tiktoken-shaped.
#![forbid(unsafe_code)]
pub mod error;
pub use error::{DecodeError, EncodeError, Error};

use std::collections::HashSet;
use std::path::Path;

use sonictok_core::encoding::{Decoder, Engine};
use sonictok_core::pretok::Grammar;
use sonictok_core::rank::Rank;
use sonictok_core::specials::SpecialTokens;
use sonictok_core::vocab::Vocab;
use sonictok_data::VocabBlob;

/// Map a bundled encoding name to its pretokenizer grammar.
fn grammar_for(encoding: &str) -> Option<Grammar> {
    match encoding {
        "cl100k_base" => Some(Grammar::Cl100k),
        "o200k_base" | "o200k_harmony" => Some(Grammar::O200k),
        "qwen3" => Some(Grammar::Qwen),
        "llama3" => Some(Grammar::Cl100k), // same grammar as cl100k, no normalizer
        _ => None,
    }
}

/// Blob grammar byte -> Grammar (0=cl100k, 1=o200k, 2=qwen).
fn grammar_from_u8(g: u8) -> Option<Grammar> {
    match g {
        0 => Some(Grammar::Cl100k),
        1 => Some(Grammar::O200k),
        2 => Some(Grammar::Qwen),
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
    nfc: bool,
    vocab: Vocab,
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
        // Self-describing v2 blobs carry their grammar + normalizer; legacy v1
        // blobs fall back to a name-based lookup.
        let (grammar, nfc) = if blob.grammar != sonictok_data::GRAMMAR_UNKNOWN {
            let g = grammar_from_u8(blob.grammar)
                .ok_or_else(|| Error::UnsupportedEncoding(blob.name.clone()))?;
            (g, blob.normalizer == 1)
        } else {
            let g = grammar_for(&blob.name)
                .ok_or_else(|| Error::UnsupportedEncoding(blob.name.clone()))?;
            (g, blob.name == "qwen3")
        };
        let vocab_size = blob.ranks.len();
        let n_vocab = (blob.max_id as usize) + 1;
        let mut by_id: Vec<Option<Vec<u8>>> = vec![None; n_vocab];
        for (bytes, id) in &blob.ranks {
            by_id[*id as usize] = Some(bytes.clone());
        }
        let vocab = Vocab::from_pairs(blob.ranks);
        let specials = SpecialTokens::new(blob.specials);
        Ok(Tokenizer {
            encoding: blob.name,
            grammar,
            nfc,
            vocab,
            decoder: DenseDecoder { by_id },
            specials,
            n_vocab,
            vocab_size,
        })
    }

    /// Load an encoding from a directory. Any valid `<encoding>.stb` blob loads
    /// (including imported ones) — the blob is self-describing.
    pub fn load_dir(dir: &Path, encoding: &str) -> Result<Self, Error> {
        let path = dir.join(format!("{encoding}.stb"));
        let bytes = std::fs::read(path)?;
        let blob = VocabBlob::from_bytes(&bytes)?;
        Self::from_blob(blob)
    }

    fn engine(&self) -> Engine<'_, DenseDecoder> {
        Engine::new(&self.vocab, &self.decoder, &self.specials, self.grammar)
    }

    /// Normalize input per the encoding's normalizer (qwen3 = NFC). Clean input
    /// pays only a quick scan; only non-NFC text is rewritten. HF tokenizers uses
    /// the same `unicode-normalization` crate, so this matches byte-for-byte.
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};
        if self.nfc {
            match is_nfc_quick(text.chars()) {
                IsNormalized::Yes => std::borrow::Cow::Borrowed(text),
                _ => std::borrow::Cow::Owned(text.nfc().collect()),
            }
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let text = self.normalize(text);
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.engine().encode_ordinary_into(&text, &mut out);
        out
    }
    pub fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) {
        let text = self.normalize(text);
        self.engine().encode_ordinary_into(&text, out);
    }
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let text = self.normalize(text);
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        self.engine().encode_with_special_into(&text, &mut out);
        out
    }
    pub fn encode(&self, text: &str, allowed: Allowed<'_>) -> Result<Vec<u32>, EncodeError> {
        let text = self.normalize(text);
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        let pred = self.allow_pred(allowed);
        self.engine()
            .encode_into(&text, &pred, &mut out)
            .map_err(|e| EncodeError {
                token: String::from_utf8_lossy(&e.token).into_owned(),
                offset: e.offset,
            })?;
        Ok(out)
    }
    pub fn count(&self, text: &str) -> usize {
        let text = self.normalize(text);
        self.engine().count(&text)
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

/// Embedded vocab blobs (feature `embed-data`), so the binary is self-contained.
#[cfg(feature = "embed-data")]
fn embedded_blob(name: &str) -> Option<&'static [u8]> {
    match name {
        "cl100k_base" => Some(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/cl100k_base.stb"
        ))),
        "o200k_base" => Some(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/o200k_base.stb"
        ))),
        "o200k_harmony" => Some(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/o200k_harmony.stb"
        ))),
        "qwen3" => Some(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/qwen3.stb"
        ))),
        "llama3" => Some(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/llama3.stb"
        ))),
        _ => None,
    }
}

/// Bundled-encoding lookup. Resolution order: `SONICTOK_DATA` env override, then
/// embedded blobs (feature `embed-data`), then the in-repo `data/` dir (dev).
pub fn get_encoding(name: &str) -> Result<Tokenizer, Error> {
    if let Ok(dir) = std::env::var("SONICTOK_DATA") {
        return Tokenizer::load_dir(std::path::Path::new(&dir), name);
    }
    #[cfg(feature = "embed-data")]
    if let Some(bytes) = embedded_blob(name) {
        return Tokenizer::from_blob(VocabBlob::from_bytes(bytes)?);
    }
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    Tokenizer::load_dir(&dir, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        get_encoding("cl100k_base").unwrap()
    }

    /// A self-describing v2 blob loads under an arbitrary name (the core guarantee
    /// that makes the generic importer work), carrying its own grammar + NFC.
    #[test]
    fn imported_blob_is_self_describing() {
        let ranks: Vec<(Vec<u8>, u32)> = (0u16..256).map(|b| (vec![b as u8], b as u32)).collect();
        let blob = VocabBlob {
            name: "custom_enc".into(),
            max_id: 255,
            grammar: 2,    // qwen
            normalizer: 1, // NFC
            ranks,
            specials: vec![],
        };
        // round-trips through the serialized form (what the importer writes).
        let t = Tokenizer::from_blob(VocabBlob::from_bytes(&blob.to_bytes()).unwrap()).unwrap();
        assert_eq!(t.encoding(), "custom_enc"); // name not in grammar_for, still loads
        // grammar=qwen (single-digit numbers) came from the blob:
        assert_eq!(t.encode_ordinary("12"), vec![b'1' as u32, b'2' as u32]);
        // normalizer=NFC came from the blob: decomposed input normalizes.
        assert_eq!(t.decode(&t.encode_ordinary("e\u{0301}")).unwrap(), "\u{e9}");
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
