//! The encode/decode engine over a loaded vocab. Generic over the pretokenizer
//! and rank backing so optimization rungs slot in without touching this layer.
use crate::bpe::byte_pair_encode;
use crate::pretok::{Grammar, Pretokenizer, Scanner};
use crate::rank::{Rank, RankLookup};
use crate::specials::SpecialTokens;

/// Disallowed-special error on the special-aware `encode` path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisallowedSpecial {
    pub token: Vec<u8>,
    pub offset: usize,
}

/// Decode error: an id with no byte string and no special name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidToken(pub Rank);

/// Reverse map id -> bytes for decode. Rung 0 uses a Vec indexed by id.
pub trait Decoder {
    fn bytes_for(&self, id: Rank) -> Option<&[u8]>;
}

pub struct Engine<'a, R: RankLookup, D: Decoder> {
    pub ranks: &'a R,
    pub decoder: &'a D,
    pub specials: &'a SpecialTokens,
    pub grammar: Grammar,
}

impl<'a, R: RankLookup, D: Decoder> Engine<'a, R, D> {
    pub fn new(
        ranks: &'a R,
        decoder: &'a D,
        specials: &'a SpecialTokens,
        grammar: Grammar,
    ) -> Self {
        Self {
            ranks,
            decoder,
            specials,
            grammar,
        }
    }

    /// encode_ordinary: specials are literal bytes. Appends ids to `out`.
    pub fn encode_ordinary_into(&self, text: &str, out: &mut Vec<Rank>) {
        self.encode_ordinary_bytes(text.as_bytes(), out);
    }

    fn encode_ordinary_bytes(&self, bytes: &[u8], out: &mut Vec<Rank>) {
        let mut pre = Scanner::new(self.grammar);
        let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(32);
        while let Some((a, z)) = pre.next_piece(bytes) {
            byte_pair_encode(&bytes[a..z], self.ranks, &mut parts, out);
        }
    }

    pub fn count(&self, text: &str) -> usize {
        // Same work as encode but only counts (Rung 0: count emitted ids).
        let mut tmp = Vec::new();
        self.encode_ordinary_into(text, &mut tmp);
        tmp.len()
    }

    /// encode_with_special: all specials recognized -> their ids.
    pub fn encode_with_special_into(&self, text: &str, out: &mut Vec<Rank>) {
        let allow_all = |_id: Rank| true;
        self.encode_special_inner(text.as_bytes(), &allow_all, out)
            .expect("all allowed");
    }

    /// encode: raises on a special not in `allowed`.
    pub fn encode_into(
        &self,
        text: &str,
        allowed: &dyn Fn(Rank) -> bool,
        out: &mut Vec<Rank>,
    ) -> Result<(), DisallowedSpecial> {
        self.encode_special_inner(text.as_bytes(), allowed, out)
    }

    /// Shared special-aware path. `allowed` gates which specials are emitted;
    /// a special present in text but NOT allowed is an error (tiktoken semantics).
    fn encode_special_inner(
        &self,
        bytes: &[u8],
        allowed: &dyn Fn(Rank) -> bool,
        out: &mut Vec<Rank>,
    ) -> Result<(), DisallowedSpecial> {
        if self.specials.is_empty() {
            self.encode_ordinary_bytes(bytes, out);
            return Ok(());
        }
        // tiktoken semantics: if any DISALLOWED special appears literally, error
        // (find_next returns the earliest such occurrence).
        if let Some((s, e, _id)) = self.specials.find_next(bytes, 0, &|id: Rank| !allowed(id)) {
            return Err(DisallowedSpecial {
                token: bytes[s..e].to_vec(),
                offset: s,
            });
        }
        // Now split on allowed specials and encode ordinary spans between.
        let mut pos = 0;
        while pos < bytes.len() {
            match self.specials.find_next(bytes, pos, allowed) {
                Some((s, e, id)) => {
                    if s > pos {
                        self.encode_ordinary_bytes(&bytes[pos..s], out);
                    }
                    out.push(id);
                    pos = e;
                }
                None => {
                    self.encode_ordinary_bytes(&bytes[pos..], out);
                    break;
                }
            }
        }
        Ok(())
    }

    /// decode ids -> bytes (lossless). Handles special ids too.
    pub fn decode_into(&self, ids: &[Rank], out: &mut Vec<u8>) -> Result<(), InvalidToken> {
        for &id in ids {
            if let Some(b) = self.decoder.bytes_for(id) {
                out.extend_from_slice(b);
            } else if let Some(name) = self.specials.name_of(id) {
                out.extend_from_slice(name);
            } else {
                return Err(InvalidToken(id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretok::Grammar;
    use crate::rank::RankMap;

    struct VecDecoder(Vec<Option<Vec<u8>>>);
    impl Decoder for VecDecoder {
        fn bytes_for(&self, id: Rank) -> Option<&[u8]> {
            self.0.get(id as usize).and_then(|o| o.as_deref())
        }
    }

    // Build a byte-only vocab (ids 0..256 == bytes) + decoder for round-trip.
    fn byte_vocab() -> (RankMap, VecDecoder) {
        let pairs: Vec<(Vec<u8>, Rank)> = (0u16..256).map(|b| (vec![b as u8], b as Rank)).collect();
        let dec: Vec<Option<Vec<u8>>> = (0u16..256).map(|b| Some(vec![b as u8])).collect();
        (RankMap::from_pairs(pairs), VecDecoder(dec))
    }

    #[test]
    fn ordinary_roundtrip() {
        let (v, d) = byte_vocab();
        let sp = SpecialTokens::new(vec![]);
        let eng = Engine::new(&v, &d, &sp, Grammar::Cl100k);
        let mut ids = vec![];
        eng.encode_ordinary_into("hello world", &mut ids);
        let mut back = vec![];
        eng.decode_into(&ids, &mut back).unwrap();
        assert_eq!(back, b"hello world");
    }

    #[test]
    fn count_matches_len() {
        let (v, d) = byte_vocab();
        let sp = SpecialTokens::new(vec![]);
        let eng = Engine::new(&v, &d, &sp, Grammar::Cl100k);
        let mut ids = vec![];
        eng.encode_ordinary_into("abc 123", &mut ids);
        assert_eq!(eng.count("abc 123"), ids.len());
    }

    #[test]
    fn disallowed_special_errors() {
        let (v, d) = byte_vocab();
        let sp = SpecialTokens::new(vec![(b"<|endoftext|>".to_vec(), 100257)]);
        let eng = Engine::new(&v, &d, &sp, Grammar::Cl100k);
        let mut ids = vec![];
        let deny_all = |_id| false;
        let err = eng
            .encode_into("a<|endoftext|>b", &deny_all, &mut ids)
            .unwrap_err();
        assert_eq!(err.offset, 1);
        assert_eq!(err.token, b"<|endoftext|>");
    }

    #[test]
    fn with_special_emits_id() {
        let (v, d) = byte_vocab();
        let sp = SpecialTokens::new(vec![(b"<|endoftext|>".to_vec(), 100257)]);
        let eng = Engine::new(&v, &d, &sp, Grammar::Cl100k);
        let mut ids = vec![];
        eng.encode_with_special_into("a<|endoftext|>", &mut ids);
        assert_eq!(*ids.last().unwrap(), 100257);
        let mut back = vec![];
        eng.decode_into(&ids, &mut back).unwrap();
        assert_eq!(back, b"a<|endoftext|>");
    }
}
