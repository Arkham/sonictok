//! Token rank (= id) storage and lookup. Rung 0/1 backing is a HashMap; later
//! rungs swap in a perfect hash behind the same `RankLookup` trait.

/// A BPE rank, identical to a token id for mergeable tokens.
pub type Rank = u32;

/// Sentinel meaning "no rank" (pair not in vocab). Never a real id.
pub const RANK_MAX: Rank = Rank::MAX;

/// Lookup of a byte sequence to its rank. The only interface `bpe` needs.
pub trait RankLookup {
    fn get(&self, key: &[u8]) -> Option<Rank>;
    /// Rank of the 2-byte token `[a, b]`, or `RANK_MAX` if absent. The BPE
    /// initial scan calls this for every adjacent pair; a dense direct table
    /// beats hashing. Default falls back to `get`.
    #[inline]
    fn get_pair(&self, a: u8, b: u8) -> Rank {
        self.get(&[a, b]).unwrap_or(RANK_MAX)
    }
}

/// Vocab: owned byte strings -> rank, over hashbrown + a cheap FxHash, plus a
/// dense 256x256 table for 2-byte tokens (the BPE hot path).
#[derive(Debug, Clone)]
pub struct RankMap {
    map: crate::hash::FxHashMap<Vec<u8>, Rank>,
    /// byte2[(a as usize) << 8 | b as usize] = rank, or RANK_MAX if not a token.
    byte2: Box<[Rank]>,
}

impl Default for RankMap {
    fn default() -> Self {
        Self {
            map: crate::hash::FxHashMap::default(),
            byte2: vec![RANK_MAX; 65536].into_boxed_slice(),
        }
    }
}

impl RankMap {
    pub fn from_pairs(pairs: impl IntoIterator<Item = (Vec<u8>, Rank)>) -> Self {
        let mut byte2 = vec![RANK_MAX; 65536].into_boxed_slice();
        let mut map = crate::hash::FxHashMap::default();
        for (bytes, rank) in pairs {
            if bytes.len() == 2 {
                byte2[(bytes[0] as usize) << 8 | bytes[1] as usize] = rank;
            }
            map.insert(bytes, rank);
        }
        Self { map, byte2 }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Reverse lookup id -> bytes, used by decode. O(n); replaced by a
    /// dense table at load time in the public crate. Provided here for tests.
    pub fn bytes_for(&self, id: Rank) -> Option<&[u8]> {
        self.map
            .iter()
            .find(|&(_, &r)| r == id)
            .map(|(k, _)| k.as_slice())
    }
}

impl RankLookup for RankMap {
    #[inline]
    fn get(&self, key: &[u8]) -> Option<Rank> {
        self.map.get(key).copied()
    }
    #[inline]
    fn get_pair(&self, a: u8, b: u8) -> Rank {
        self.byte2[(a as usize) << 8 | b as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_hit_and_miss() {
        let m = RankMap::from_pairs([(b"ab".to_vec(), 7), (b"a".to_vec(), 1)]);
        assert_eq!(m.get(b"ab"), Some(7));
        assert_eq!(m.get(b"a"), Some(1));
        assert_eq!(m.get(b"zzz"), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn reverse_lookup() {
        let m = RankMap::from_pairs([(b"ab".to_vec(), 7)]);
        assert_eq!(m.bytes_for(7), Some(&b"ab"[..]));
        assert_eq!(m.bytes_for(9), None);
    }
}
