//! The encode/decode engine over a loaded vocab. Generic over the pretokenizer
//! and rank backing so optimization rungs slot in without touching this layer.
use crate::pretok::Grammar;
use crate::rank::Rank;
use crate::specials::SpecialTokens;
use crate::vocab::Vocab;

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

pub struct Engine<'a, D: Decoder> {
    pub vocab: &'a Vocab,
    pub decoder: &'a D,
    pub specials: &'a SpecialTokens,
    pub grammar: Grammar,
}

impl<'a, D: Decoder> Engine<'a, D> {
    pub fn new(
        vocab: &'a Vocab,
        decoder: &'a D,
        specials: &'a SpecialTokens,
        grammar: Grammar,
    ) -> Self {
        Self {
            vocab,
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
        match self.grammar {
            Grammar::Cl100k => self.encode_cl100k_fused(bytes, out, 3),
            Grammar::Qwen => self.encode_cl100k_fused(bytes, out, 1),
            Grammar::O200k => self.encode_o200k_fused(bytes, out),
        }
    }

    /// Fused single-pass o200k product machine (o200k_base / o200k_harmony):
    /// case-aware letter alts (prefix-first; UPPER*LOWER+ then UPPER+LOWER*) with
    /// attached contractions, '/' in the punct tail, lastnl-first whitespace.
    /// Non-ASCII contact falls back to the exact scalar scanner.
    fn encode_o200k_fused(&self, t: &[u8], out: &mut Vec<Rank>) {
        use crate::pretok::o200k::piece_end;
        fn a_let(b: u8) -> bool {
            (b | 0x20).wrapping_sub(b'a') <= 25
        }
        fn a_dig(b: u8) -> bool {
            b.wrapping_sub(b'0') <= 9
        }
        fn a_ws(b: u8) -> bool {
            b.wrapping_sub(9) <= 4 || b == b' '
        }
        fn a_pun(b: u8) -> bool {
            b < 0x80 && !a_ws(b) && !a_let(b) && !a_dig(b)
        }
        fn upper_run(t: &[u8], mut q: usize) -> usize {
            while q < t.len() && t[q].wrapping_sub(b'A') <= 25 {
                q += 1;
            }
            q
        }
        fn lower_run(t: &[u8], mut q: usize) -> usize {
            while q < t.len() && t[q].wrapping_sub(b'a') <= 25 {
                q += 1;
            }
            q
        }
        // UPPER* LOWER+ ; returns end (0 = no match). Sets hitmb on non-ASCII.
        fn match_ul(t: &[u8], st: usize, hitmb: &mut bool) -> usize {
            let i = upper_run(t, st);
            if i < t.len() && t[i] >= 0x80 {
                *hitmb = true;
                return 0;
            }
            let j = lower_run(t, i);
            if j < t.len() && t[j] >= 0x80 {
                *hitmb = true;
                return 0;
            }
            if j > i { j } else { 0 }
        }
        // UPPER+ LOWER* ; returns end (0 = no match). Sets hitmb on non-ASCII.
        fn match_upl(t: &[u8], st: usize, hitmb: &mut bool) -> usize {
            let i = upper_run(t, st);
            if i == st {
                return 0;
            }
            if i < t.len() && t[i] >= 0x80 {
                *hitmb = true;
                return 0;
            }
            let j = lower_run(t, i);
            if j < t.len() && t[j] >= 0x80 {
                *hitmb = true;
                return 0;
            }
            j
        }
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)? attached after a letter run ending at `e`.
        fn o_contraction(t: &[u8], e: usize) -> usize {
            if e >= t.len() || t[e] != b'\'' || e + 1 >= t.len() {
                return e;
            }
            let c1 = t[e + 1] | 0x20;
            if c1 == b's' || c1 == b't' || c1 == b'm' || c1 == b'd' {
                return e + 2;
            }
            if e + 2 < t.len() {
                let c2 = t[e + 2] | 0x20;
                if (c2 == b'e' && (c1 == b'r' || c1 == b'v')) || (c1 == b'l' && c2 == b'l') {
                    return e + 3;
                }
            }
            e
        }

        let l = t.len();
        let mut p = 0usize;
        while p < l {
            let b0 = t[p];
            let mut fb = b0 >= 0x80;
            let mut adv = 0usize;
            if !fb {
                'ascii: {
                    let mut hitmb = false;
                    let prefelig = !a_let(b0) && !a_dig(b0) && b0 != b'\r' && b0 != b'\n';
                    if prefelig && p + 1 < l && t[p + 1] >= 0x80 {
                        fb = true;
                        break 'ascii;
                    }
                    let mut e = 0usize;
                    if prefelig && p + 1 < l {
                        e = match_ul(t, p + 1, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'ascii;
                    }
                    if e == 0 {
                        e = match_ul(t, p, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'ascii;
                    }
                    if e == 0 && prefelig && p + 1 < l {
                        e = match_upl(t, p + 1, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'ascii;
                    }
                    if e == 0 {
                        e = match_upl(t, p, &mut hitmb);
                    }
                    if hitmb {
                        fb = true;
                        break 'ascii;
                    }
                    if e != 0 {
                        let end = o_contraction(t, e);
                        self.vocab.encode(&t[p..end], out);
                        adv = end - p;
                        break 'ascii;
                    }
                    // Alt 3: \p{N}{1,3}
                    if a_dig(b0) {
                        let mut q = p + 1;
                        let mut c = 1;
                        while q < l && c < 3 && a_dig(t[q]) {
                            q += 1;
                            c += 1;
                        }
                        if c < 3 && q < l && t[q] >= 0x80 {
                            fb = true;
                            break 'ascii;
                        }
                        self.vocab.encode(&t[p..q], out);
                        adv = q - p;
                        break 'ascii;
                    }
                    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
                    {
                        let mut q = p + usize::from(b0 == b' ');
                        let s4 = q;
                        while q < l && a_pun(t[q]) {
                            q += 1;
                        }
                        if q > s4 {
                            if q < l && t[q] >= 0x80 {
                                fb = true;
                                break 'ascii;
                            }
                            while q < l && (t[q] == b'\r' || t[q] == b'\n' || t[q] == b'/') {
                                q += 1;
                            }
                            self.vocab.encode(&t[p..q], out);
                            adv = q - p;
                            break 'ascii;
                        }
                    }
                    // Alt 5-7: whitespace cascade (o200k order: lastnl-first)
                    if a_ws(b0) {
                        let mut e = p;
                        let mut lastnl = usize::MAX;
                        while e < l && a_ws(t[e]) {
                            if t[e] == b'\r' || t[e] == b'\n' {
                                lastnl = e;
                            }
                            e += 1;
                        }
                        if e < l && t[e] >= 0x80 {
                            fb = true;
                            break 'ascii;
                        }
                        let plen = if lastnl != usize::MAX {
                            lastnl + 1 - p
                        } else if e == l {
                            e - p
                        } else if e - p > 1 {
                            e - p - 1
                        } else {
                            1
                        };
                        self.vocab.encode(&t[p..p + plen], out);
                        adv = plen;
                        break 'ascii;
                    }
                    fb = true; // unreachable for ASCII
                }
            }
            if fb {
                let end = piece_end(t, p);
                self.vocab.encode(&t[p..end], out);
                p = end;
            } else {
                p += adv;
            }
        }
    }

    /// Fused single-pass cl100k product machine: pretok boundaries + token
    /// emission in one loop over ASCII bytes; any non-ASCII contact falls back
    /// to the exact scalar scanner for one piece (so output is byte-exact).
    fn encode_cl100k_fused(&self, t: &[u8], out: &mut Vec<Rank>, num_max: usize) {
        use crate::pretok::cl100k::piece_end;
        let l = t.len();
        let mut p = 0usize;
        let a_let = |b: u8| (b | 0x20).wrapping_sub(b'a') <= 25;
        let a_dig = |b: u8| b.wrapping_sub(b'0') <= 9;
        let a_ws = |b: u8| b.wrapping_sub(9) <= 4 || b == b' ';
        let a_pun = |b: u8| b < 0x80 && !a_ws(b) && !a_let(b) && !a_dig(b);
        while p < l {
            let b0 = t[p];
            let mut fb = b0 >= 0x80;
            let mut adv = 0usize;
            if !fb {
                'ascii: {
                    // Alt 1: '(?i:[sdmt]|ll|ve|re)
                    if b0 == b'\'' && p + 1 < l {
                        let c1 = t[p + 1] | 0x20;
                        if c1 == b's' || c1 == b'd' || c1 == b'm' || c1 == b't' {
                            self.vocab.encode(&t[p..p + 2], out);
                            adv = 2;
                            break 'ascii;
                        }
                        if p + 2 < l {
                            let c2 = t[p + 2] | 0x20;
                            if (c1 == b'l' && c2 == b'l')
                                || (c1 == b'v' && c2 == b'e')
                                || (c1 == b'r' && c2 == b'e')
                            {
                                self.vocab.encode(&t[p..p + 3], out);
                                adv = 3;
                                break 'ascii;
                            }
                        }
                    }
                    // Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+
                    let ls = if a_let(b0) {
                        p
                    } else if b0 < 0x80
                        && !a_dig(b0)
                        && b0 != b'\r'
                        && b0 != b'\n'
                        && p + 1 < l
                        && a_let(t[p + 1])
                    {
                        p + 1
                    } else {
                        usize::MAX
                    };
                    if ls != usize::MAX {
                        let mut we = ls;
                        while we < l && a_let(t[we]) {
                            we += 1;
                        }
                        if we < l && t[we] >= 0x80 {
                            fb = true;
                            break 'ascii;
                        }
                        self.vocab.encode(&t[p..we], out);
                        adv = we - p;
                        break 'ascii;
                    }
                    // Alt 3: \p{N}{1,num_max}
                    if a_dig(b0) {
                        let mut q = p + 1;
                        let mut c = 1;
                        while q < l && c < num_max && a_dig(t[q]) {
                            q += 1;
                            c += 1;
                        }
                        if c < num_max && q < l && t[q] >= 0x80 {
                            fb = true;
                            break 'ascii;
                        }
                        self.vocab.encode(&t[p..q], out);
                        adv = q - p;
                        break 'ascii;
                    }
                    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
                    {
                        let mut q = p + usize::from(b0 == b' ');
                        let s4 = q;
                        while q < l && a_pun(t[q]) {
                            q += 1;
                        }
                        if q > s4 {
                            if q < l && t[q] >= 0x80 {
                                fb = true;
                                break 'ascii;
                            }
                            while q < l && (t[q] == b'\r' || t[q] == b'\n') {
                                q += 1;
                            }
                            self.vocab.encode(&t[p..q], out);
                            adv = q - p;
                            break 'ascii;
                        }
                    }
                    // Alt 5-7: whitespace cascade (cl100k order)
                    if a_ws(b0) {
                        let mut e = p;
                        let mut lastnl = usize::MAX;
                        while e < l && a_ws(t[e]) {
                            if t[e] == b'\r' || t[e] == b'\n' {
                                lastnl = e;
                            }
                            e += 1;
                        }
                        if e < l && t[e] >= 0x80 {
                            fb = true;
                            break 'ascii;
                        }
                        let plen = if e == l {
                            e - p
                        } else if lastnl != usize::MAX {
                            lastnl + 1 - p
                        } else if e - p > 1 {
                            e - p - 1
                        } else {
                            1
                        };
                        self.vocab.encode(&t[p..p + plen], out);
                        adv = plen;
                        break 'ascii;
                    }
                    fb = true; // unreachable for ASCII
                }
            }
            if fb {
                let end = piece_end(t, p, num_max);
                self.vocab.encode(&t[p..end], out);
                p = end;
            } else {
                p += adv;
            }
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
    use crate::vocab::Vocab;

    struct VecDecoder(Vec<Option<Vec<u8>>>);
    impl Decoder for VecDecoder {
        fn bytes_for(&self, id: Rank) -> Option<&[u8]> {
            self.0.get(id as usize).and_then(|o| o.as_deref())
        }
    }

    // Build a byte-only vocab (ids 0..256 == bytes) + decoder for round-trip.
    fn byte_vocab() -> (Vocab, VecDecoder) {
        let pairs: Vec<(Vec<u8>, Rank)> = (0u16..256).map(|b| (vec![b as u8], b as Rank)).collect();
        let dec: Vec<Option<Vec<u8>>> = (0u16..256).map(|b| Some(vec![b as u8])).collect();
        (Vocab::from_pairs(pairs), VecDecoder(dec))
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
