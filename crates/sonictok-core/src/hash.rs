//! A small, fast, dependency-free FxHash-style hasher for byte-slice keys.
//! SipHash (std default) dominates the BPE hot path; this is ~an order of
//! magnitude cheaper per key while keeping hashbrown's table quality.
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let v = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            self.add(v);
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let v = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64;
            self.add(v);
            bytes = &bytes[4..];
        }
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type BuildFxHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildFxHasher>;
