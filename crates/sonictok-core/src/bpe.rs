//! Exact tiktoken backtracking BPE. The result is identical to tiktoken's
//! `_byte_pair_merge`: merge the globally-minimum-rank adjacent pair, ties
//! broken leftmost (strict `<`). Single-byte and whole-piece are shortcuts.
use crate::rank::{RANK_MAX, Rank, RankLookup};

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
    if piece.len() == 1 {
        out.push(ranks.get(piece).expect("single byte must be a token"));
        return;
    }
    if let Some(t) = ranks.get(piece) {
        out.push(t);
        return;
    }

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

    while min_rank.0 != RANK_MAX {
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
