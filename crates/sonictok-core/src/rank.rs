//! Token rank (= id) storage and lookup, tuned for the BPE hot path.
//!
//! Lookups dispatch by key length:
//!   len 1   -> `byte_id[256]`     (direct array)
//!   len 2   -> `byte2[65536]`     (direct array)
//!   len 3..=7 -> `short`          (open-addressed table, inline u64 keys)
//!   len >=8 -> `long`             (hashbrown + FxHash, heap Vec keys)
//!
//! Short keys (the vast majority of tokens and merge spans) live in a compact
//! ~2 MB table with keys stored *inline* as u64 — no heap-pointer chase on key
//! compare, so it stays L2-resident. All structures are immutable after build.

/// A BPE rank, identical to a token id for mergeable tokens.
pub type Rank = u32;

/// Sentinel meaning "no rank" (pair not in vocab). Never a real id.
pub const RANK_MAX: Rank = Rank::MAX;

/// Lookup of a byte sequence to its rank, plus the dense fast paths.
pub trait RankLookup {
    fn get(&self, key: &[u8]) -> Option<Rank>;
    /// Rank of the 2-byte token `[a, b]`, or `RANK_MAX` (initial pair scan).
    fn get_pair(&self, a: u8, b: u8) -> Rank {
        self.get(&[a, b]).unwrap_or(RANK_MAX)
    }
}

/// Pack a byte slice of length 1..=7 into a u64 (bytes in low 56 bits, length in
/// the top byte). Distinct for distinct (bytes, len); never zero (len >= 1).
#[inline]
fn pack(b: &[u8]) -> u64 {
    debug_assert!((1..=7).contains(&b.len()));
    let mut v = (b.len() as u64) << 56;
    for (i, &byte) in b.iter().enumerate() {
        v |= (byte as u64) << (i * 8);
    }
    v
}

#[inline]
fn mix64(key: u64) -> u64 {
    // Fibonacci hashing; high bits are well-mixed.
    key.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Open-addressed (linear probe) table with inline u64 keys; key 0 = empty.
#[derive(Debug, Clone, Default)]
struct ShortTable {
    keys: Box<[u64]>,
    vals: Box<[Rank]>,
    mask: usize,
    shift: u32,
}

impl ShortTable {
    fn build(entries: &[(u64, Rank)]) -> Self {
        let mut cap = 8usize;
        let want = (entries.len() * 10 / 7).max(1);
        while cap < want {
            cap <<= 1;
        }
        let shift = 64 - cap.trailing_zeros();
        let mut t = ShortTable {
            keys: vec![0u64; cap].into_boxed_slice(),
            vals: vec![RANK_MAX; cap].into_boxed_slice(),
            mask: cap - 1,
            shift,
        };
        for &(k, v) in entries {
            debug_assert!(k != 0);
            let mut slot = (mix64(k) >> t.shift) as usize & t.mask;
            while t.keys[slot] != 0 {
                slot = (slot + 1) & t.mask;
            }
            t.keys[slot] = k;
            t.vals[slot] = v;
        }
        t
    }

    /// Find the packed key whose value equals `id` (O(n); tests only).
    fn key_for(&self, id: Rank) -> Option<u64> {
        self.keys
            .iter()
            .zip(self.vals.iter())
            .find(|&(&k, &v)| k != 0 && v == id)
            .map(|(&k, _)| k)
    }

    #[inline]
    fn get(&self, key: u64) -> Option<Rank> {
        if self.keys.is_empty() {
            return None;
        }
        let mut slot = (mix64(key) >> self.shift) as usize & self.mask;
        loop {
            let k = self.keys[slot];
            if k == key {
                return Some(self.vals[slot]);
            }
            if k == 0 {
                return None;
            }
            slot = (slot + 1) & self.mask;
        }
    }
}

/// Vocab with length-dispatched fast paths.
#[derive(Debug, Clone)]
pub struct RankMap {
    byte_id: Box<[Rank]>,                        // 256
    byte2: Box<[Rank]>,                          // 65536
    short: ShortTable,                           // len 3..=7
    long: crate::hash::FxHashMap<Vec<u8>, Rank>, // len >= 8
    n: usize,
}

impl Default for RankMap {
    fn default() -> Self {
        Self {
            byte_id: vec![RANK_MAX; 256].into_boxed_slice(),
            byte2: vec![RANK_MAX; 65536].into_boxed_slice(),
            short: ShortTable::default(),
            long: crate::hash::FxHashMap::default(),
            n: 0,
        }
    }
}

impl RankMap {
    pub fn from_pairs(pairs: impl IntoIterator<Item = (Vec<u8>, Rank)>) -> Self {
        let mut byte_id = vec![RANK_MAX; 256].into_boxed_slice();
        let mut byte2 = vec![RANK_MAX; 65536].into_boxed_slice();
        let mut short_entries: Vec<(u64, Rank)> = Vec::new();
        let mut long: crate::hash::FxHashMap<Vec<u8>, Rank> = crate::hash::FxHashMap::default();
        let mut n = 0;
        for (bytes, rank) in pairs {
            n += 1;
            match bytes.len() {
                0 => {}
                1 => byte_id[bytes[0] as usize] = rank,
                2 => byte2[(bytes[0] as usize) << 8 | bytes[1] as usize] = rank,
                3..=7 => short_entries.push((pack(&bytes), rank)),
                _ => {
                    long.insert(bytes, rank);
                }
            }
        }
        RankMap {
            byte_id,
            byte2,
            short: ShortTable::build(&short_entries),
            long,
            n,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Reverse lookup id -> bytes (O(n); tests only). The public crate uses a
    /// dense id-indexed decoder.
    pub fn bytes_for(&self, id: Rank) -> Option<Vec<u8>> {
        for b in 0u16..256 {
            if self.byte_id[b as usize] == id {
                return Some(vec![b as u8]);
            }
        }
        for hi in 0u16..256 {
            for lo in 0u16..256 {
                if self.byte2[(hi as usize) << 8 | lo as usize] == id {
                    return Some(vec![hi as u8, lo as u8]);
                }
            }
        }
        if let Some(key) = self.short.key_for(id) {
            let len = (key >> 56) as usize;
            return Some((0..len).map(|i| (key >> (i * 8)) as u8).collect());
        }
        self.long
            .iter()
            .find(|&(_, &r)| r == id)
            .map(|(k, _)| k.clone())
    }
}

impl RankLookup for RankMap {
    #[inline]
    fn get(&self, key: &[u8]) -> Option<Rank> {
        let r = match key.len() {
            0 => return None,
            1 => self.byte_id[key[0] as usize],
            2 => self.byte2[(key[0] as usize) << 8 | key[1] as usize],
            3..=7 => return self.short.get(pack(key)),
            _ => return self.long.get(key).copied(),
        };
        if r == RANK_MAX { None } else { Some(r) }
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
    fn lookup_all_lengths() {
        let m = RankMap::from_pairs([
            (b"a".to_vec(), 1),
            (b"ab".to_vec(), 2),
            (b"abc".to_vec(), 3),
            (b"abcdefg".to_vec(), 7),      // len 7 (short max)
            (b"abcdefgh".to_vec(), 8),     // len 8 (long)
            (b"abcdefghijk".to_vec(), 11), // long
        ]);
        assert_eq!(m.get(b"a"), Some(1));
        assert_eq!(m.get(b"ab"), Some(2));
        assert_eq!(m.get(b"abc"), Some(3));
        assert_eq!(m.get(b"abcdefg"), Some(7));
        assert_eq!(m.get(b"abcdefgh"), Some(8));
        assert_eq!(m.get(b"abcdefghijk"), Some(11));
        assert_eq!(m.get(b"zzz"), None);
        assert_eq!(m.get(b""), None);
        assert_eq!(m.get_pair(b'a', b'b'), 2);
        assert_eq!(m.get_pair(b'b', b'a'), RANK_MAX);
        assert_eq!(m.len(), 6);
    }

    #[test]
    fn reverse_lookup() {
        let m = RankMap::from_pairs([(b"ab".to_vec(), 7), (b"hello".to_vec(), 9)]);
        assert_eq!(m.bytes_for(7).as_deref(), Some(&b"ab"[..]));
        assert_eq!(m.bytes_for(9).as_deref(), Some(&b"hello"[..]));
        assert_eq!(m.bytes_for(42), None);
    }

    #[test]
    fn short_table_many() {
        let pairs: Vec<(Vec<u8>, Rank)> = (0u32..6000)
            .map(|i| (format!("t{i}").into_bytes(), i))
            .collect();
        let m = RankMap::from_pairs(pairs.clone());
        for (k, v) in &pairs {
            assert_eq!(m.get(k), Some(*v));
        }
        assert_eq!(m.get(b"t999999"), None);
    }
}
