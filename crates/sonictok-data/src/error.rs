#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: not a sonictok blob")]
    BadMagic,
    #[error("unsupported format version {0} (max supported {1})")]
    UnsupportedVersion(u16, u16),
    #[error("blob truncated or corrupt: {0}")]
    Corrupt(&'static str),
    #[error("checksum mismatch: header {expected:#x}, computed {actual:#x}")]
    Checksum { expected: u64, actual: u64 },
    #[error("encoding not found: {0}")]
    EncodingNotFound(String),
}
