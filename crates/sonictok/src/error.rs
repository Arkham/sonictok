#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Data(#[from] sonictok_data::DataError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported encoding for Plan 1: {0} (only cl100k_base)")]
    UnsupportedEncoding(String),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("disallowed special token {token:?} at byte offset {offset}")]
pub struct EncodeError {
    pub token: String,
    pub offset: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid token id {0}")]
pub struct DecodeError(pub u32);
