//! Exact tiktoken backtracking BPE. The result is identical to tiktoken's
//! `_byte_pair_merge`: merge the globally-minimum-rank adjacent pair, ties
//! broken leftmost (strict `<`). Single-byte and whole-piece are shortcuts.
use crate::rank::{RANK_MAX, Rank, RankLookup};

/// Exact operation counters (feature `profile`), to attribute BPE cost.
#[cfg(feature = "profile")]
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static PIECES: AtomicU64 = AtomicU64::new(0);
    pub static SINGLE_BYTE: AtomicU64 = AtomicU64::new(0);
    pub static WHOLE_HIT: AtomicU64 = AtomicU64::new(0);
    pub static MERGE_PIECES: AtomicU64 = AtomicU64::new(0);
    pub static MERGE_ITERS: AtomicU64 = AtomicU64::new(0);
    pub static LOOKUPS: AtomicU64 = AtomicU64::new(0); // get() calls (whole + byte_id + pair_rank)
    pub static OUT_TOKENS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub(super) fn inc(c: &AtomicU64, n: u64) {
        c.fetch_add(n, Relaxed);
    }
    pub fn snapshot() -> [(&'static str, u64); 7] {
        [
            ("pieces", PIECES.load(Relaxed)),
            ("single_byte", SINGLE_BYTE.load(Relaxed)),
            ("whole_hit", WHOLE_HIT.load(Relaxed)),
            ("merge_pieces", MERGE_PIECES.load(Relaxed)),
            ("merge_iters", MERGE_ITERS.load(Relaxed)),
            ("lookups", LOOKUPS.load(Relaxed)),
            ("out_tokens", OUT_TOKENS.load(Relaxed)),
        ]
    }
}

#[cfg(feature = "profile")]
macro_rules! prof_inc {
    ($c:ident, $n:expr) => {
        prof::inc(&prof::$c, $n)
    };
}
#[cfg(not(feature = "profile"))]
macro_rules! prof_inc {
    ($c:ident, $n:expr) => {{}};
}

/// A working part: (byte offset, rank of the pair with the next part, token id
/// of the token starting here). Tracking the id lets emission be lookup-free.
type Part = (u32, Rank, Rank);

#[inline]
fn pair_rank<R: RankLookup>(piece: &[u8], parts: &[Part], i: usize, ranks: &R) -> Rank {
    if i + 3 < parts.len() {
        ranks
            .get(&piece[parts[i].0 as usize..parts[i + 3].0 as usize])
            .unwrap_or(RANK_MAX)
    } else {
        RANK_MAX
    }
}

/// Encode one non-empty pretokenized `piece` into token ids, appended to `out`.
/// `parts` is a caller-owned scratch buffer reused across pieces (cleared here).
/// Precondition: every single byte of `piece` is present in `ranks` (true for
/// all tiktoken byte-level vocabs).
pub fn byte_pair_encode<R: RankLookup>(
    piece: &[u8],
    ranks: &R,
    parts: &mut Vec<Part>,
    out: &mut Vec<Rank>,
) {
    debug_assert!(!piece.is_empty());
    prof_inc!(PIECES, 1);
    prof_inc!(OUT_TOKENS, 1); // approx; corrected for merge pieces below
    if piece.len() == 1 {
        prof_inc!(SINGLE_BYTE, 1);
        prof_inc!(LOOKUPS, 1);
        out.push(ranks.get(piece).expect("single byte must be a token"));
        return;
    }
    prof_inc!(LOOKUPS, 1);
    if let Some(t) = ranks.get(piece) {
        prof_inc!(WHOLE_HIT, 1);
        out.push(t);
        return;
    }
    prof_inc!(MERGE_PIECES, 1);

    // parts[k] = (offset, pair-rank with next, token id at k). The last entry is
    // the end sentinel (offset = piece.len()).
    parts.clear();
    let n = piece.len();
    let mut min_rank: (Rank, usize) = (RANK_MAX, usize::MAX);
    for i in 0..n {
        let id = ranks
            .get(&piece[i..i + 1])
            .expect("single byte must be a token");
        let rank = if i + 1 < n {
            ranks.get_pair(piece[i], piece[i + 1])
        } else {
            RANK_MAX
        };
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i as u32, rank, id));
    }
    parts.push((n as u32, RANK_MAX, RANK_MAX));

    prof_inc!(LOOKUPS, n as u64); // initial byte_id lookups
    while min_rank.0 != RANK_MAX {
        prof_inc!(MERGE_ITERS, 1);
        prof_inc!(LOOKUPS, if min_rank.1 > 0 { 2 } else { 1 }); // pair_rank calls
        let i = min_rank.1;
        // The triggering pair rank IS the id of the merged token starting at i.
        parts[i].2 = min_rank.0;
        if i > 0 {
            parts[i - 1].1 = pair_rank(piece, parts, i - 1, ranks);
        }
        parts[i].1 = pair_rank(piece, parts, i, ranks);
        parts.remove(i + 1);

        min_rank = (RANK_MAX, usize::MAX);
        for (j, &(_, rank, _)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, j);
            }
        }
    }

    // Emit tracked ids (no re-lookup); skip the end sentinel.
    prof_inc!(OUT_TOKENS, (parts.len() - 2) as u64); // correct the +1 approx above
    for &(_, _, id) in &parts[..parts.len() - 1] {
        out.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rank::RankMap;

    // A toy vocab where bytes are 0..=255 and a few merges exist.
    fn toy() -> RankMap {
        let mut pairs: Vec<(Vec<u8>, Rank)> =
            (0u16..256).map(|b| (vec![b as u8], b as Rank)).collect();
        // merges (rank order matters): "ab"=300 (lowest non-byte), "abc"=301
        pairs.push((b"ab".to_vec(), 300));
        pairs.push((b"abc".to_vec(), 301));
        RankMap::from_pairs(pairs)
    }

    #[test]
    fn single_byte() {
        let v = toy();
        let mut out = vec![];
        byte_pair_encode(b"a", &v, &mut Vec::new(), &mut out);
        assert_eq!(out, vec![b'a' as Rank]);
    }

    #[test]
    fn whole_piece_shortcut() {
        let v = toy();
        let mut out = vec![];
        byte_pair_encode(b"abc", &v, &mut Vec::new(), &mut out);
        assert_eq!(out, vec![301]); // "abc" is a single token
    }

    #[test]
    fn merges_lowest_rank_first() {
        let v = toy();
        let mut out = vec![];
        // "abx": "ab"(300) merges, then "abx" not a token -> ["ab","x"]
        byte_pair_encode(b"abx", &v, &mut Vec::new(), &mut out);
        assert_eq!(out, vec![300, b'x' as Rank]);
    }

    #[test]
    fn no_merges_falls_back_to_bytes() {
        let v = toy();
        let mut out = vec![];
        byte_pair_encode(b"xy", &v, &mut Vec::new(), &mut out);
        assert_eq!(out, vec![b'x' as Rank, b'y' as Rank]);
    }

    #[test]
    fn leftmost_tie_break() {
        // Two equal-rank pairs; leftmost must merge first.
        let mut pairs: Vec<(Vec<u8>, Rank)> =
            (0u16..256).map(|b| (vec![b as u8], b as Rank)).collect();
        pairs.push((b"aa".to_vec(), 300)); // both "aa" pairs in "aaa" share rank 300
        let v = RankMap::from_pairs(pairs);
        let mut out = vec![];
        byte_pair_encode(b"aaa", &v, &mut Vec::new(), &mut out);
        // leftmost merge: ["aa","a"]
        assert_eq!(out, vec![300, b'a' as Rank]);
    }
}
