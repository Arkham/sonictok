//! sonictok binary vocab blob. Little-endian throughout.
//!
//! layout (v1):
//!   magic:    8 bytes  = b"SONICTK\0"
//!   version:  u16      = 1
//!   flags:    u16      = 0
//!   checksum: u64      = FNV-1a over everything AFTER this field
//!   name_len: u16, name: [u8; name_len]   (encoding name, e.g. "cl100k_base")
//!   n_ranks:  u32
//!   n_special:u32
//!   max_id:   u32      (n_vocab = max_id + 1)
//!   [v2+] grammar: u8  (0=cl100k, 1=o200k, 2=qwen) — the self-describing field
//!   [v2+] normalizer: u8 (0=none, 1=NFC)           that lets imported encodings
//!                                                  carry their config
//!   rank section:    n_ranks * { len: u32, bytes: [u8; len], id: u32 }
//!   special section: n_special * { len: u16, bytes: [u8; len], id: u32 }
use crate::error::DataError;

pub const MAGIC: &[u8; 8] = b"SONICTK\0";
pub const VERSION: u16 = 2;

/// Grammar carried by v1 blobs (unknown — caller infers from the name).
pub const GRAMMAR_UNKNOWN: u8 = 255;

#[derive(Debug, Clone)]
pub struct VocabBlob {
    pub name: String,
    pub max_id: u32,
    /// 0=cl100k, 1=o200k, 2=qwen (GRAMMAR_UNKNOWN for legacy v1 blobs).
    pub grammar: u8,
    /// 0=none, 1=NFC.
    pub normalizer: u8,
    pub ranks: Vec<(Vec<u8>, u32)>,
    pub specials: Vec<(Vec<u8>, u32)>,
}

#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

impl VocabBlob {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        let name = self.name.as_bytes();
        body.extend_from_slice(&(name.len() as u16).to_le_bytes());
        body.extend_from_slice(name);
        body.extend_from_slice(&(self.ranks.len() as u32).to_le_bytes());
        body.extend_from_slice(&(self.specials.len() as u32).to_le_bytes());
        body.extend_from_slice(&self.max_id.to_le_bytes());
        body.push(self.grammar);
        body.push(self.normalizer);
        for (b, id) in &self.ranks {
            body.extend_from_slice(&(b.len() as u32).to_le_bytes());
            body.extend_from_slice(b);
            body.extend_from_slice(&id.to_le_bytes());
        }
        for (b, id) in &self.specials {
            body.extend_from_slice(&(b.len() as u16).to_le_bytes());
            body.extend_from_slice(b);
            body.extend_from_slice(&id.to_le_bytes());
        }
        let checksum = fnv1a(&body);
        let mut out = Vec::with_capacity(8 + 2 + 2 + 8 + body.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, DataError> {
        let mut r = Reader { buf, pos: 0 };
        let magic = r.take(8)?;
        if magic != MAGIC {
            return Err(DataError::BadMagic);
        }
        let version = r.u16()?;
        if version > VERSION {
            return Err(DataError::UnsupportedVersion(version, VERSION));
        }
        let _flags = r.u16()?;
        let checksum = r.u64()?;
        let body = &buf[r.pos..];
        let actual = fnv1a(body);
        if actual != checksum {
            return Err(DataError::Checksum {
                expected: checksum,
                actual,
            });
        }
        let name_len = r.u16()? as usize;
        let name = String::from_utf8(r.take(name_len)?.to_vec())
            .map_err(|_| DataError::Corrupt("name utf8"))?;
        let n_ranks = r.u32()? as usize;
        let n_special = r.u32()? as usize;
        let max_id = r.u32()?;
        let (grammar, normalizer) = if version >= 2 {
            (r.u8()?, r.u8()?)
        } else {
            (GRAMMAR_UNKNOWN, 0)
        };
        let mut ranks = Vec::with_capacity(n_ranks);
        for _ in 0..n_ranks {
            let len = r.u32()? as usize;
            let b = r.take(len)?.to_vec();
            let id = r.u32()?;
            ranks.push((b, id));
        }
        let mut specials = Vec::with_capacity(n_special);
        for _ in 0..n_special {
            let len = r.u16()? as usize;
            let b = r.take(len)?.to_vec();
            let id = r.u32()?;
            specials.push((b, id));
        }
        Ok(Self {
            name,
            max_id,
            grammar,
            normalizer,
            ranks,
            specials,
        })
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8, DataError> {
        Ok(self.take(1)?[0])
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DataError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(DataError::Corrupt("overflow"))?;
        if end > self.buf.len() {
            return Err(DataError::Corrupt("truncated"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, DataError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, DataError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, DataError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VocabBlob {
        VocabBlob {
            name: "cl100k_base".into(),
            max_id: 100276,
            grammar: 2,
            normalizer: 1,
            ranks: vec![(b"a".to_vec(), 0), (b"ab".to_vec(), 1)],
            specials: vec![(b"<|endoftext|>".to_vec(), 100257)],
        }
    }

    #[test]
    fn round_trip() {
        let b = sample();
        let bytes = b.to_bytes();
        let got = VocabBlob::from_bytes(&bytes).unwrap();
        assert_eq!(got.name, "cl100k_base");
        assert_eq!(got.max_id, 100276);
        assert_eq!(got.grammar, 2);
        assert_eq!(got.normalizer, 1);
        assert_eq!(got.ranks, b.ranks);
        assert_eq!(got.specials, b.specials);
    }

    #[test]
    fn bad_magic() {
        let mut bytes = sample().to_bytes();
        bytes[0] = b'X';
        assert!(matches!(
            VocabBlob::from_bytes(&bytes),
            Err(DataError::BadMagic)
        ));
    }

    #[test]
    fn checksum_detects_corruption() {
        let mut bytes = sample().to_bytes();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF; // flip a body byte
        assert!(matches!(
            VocabBlob::from_bytes(&bytes),
            Err(DataError::Checksum { .. })
        ));
    }

    #[test]
    fn truncated() {
        let bytes = sample().to_bytes();
        assert!(matches!(
            VocabBlob::from_bytes(&bytes[..bytes.len() - 3]),
            Err(DataError::Checksum { .. }) | Err(DataError::Corrupt(_))
        ));
    }

    #[test]
    #[ignore = "requires data/cl100k_base.stb (run: cargo run -p xtask -- build-data cl100k_base)"]
    fn loads_real_cl100k() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/cl100k_base.stb"
        ))
        .unwrap();
        let blob = VocabBlob::from_bytes(&bytes).unwrap();
        assert_eq!(blob.name, "cl100k_base");
        assert_eq!(blob.max_id, 100276);
        assert_eq!(blob.ranks.len(), 100256);
        assert_eq!(blob.specials.len(), 5);
    }
}
