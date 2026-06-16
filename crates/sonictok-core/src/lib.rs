//! sonictok-core: dependency-free, allocation-light exact BPE engine.
#![deny(unsafe_op_in_unsafe_fn)] // the only unsafe is pretok::char_at (documented SAFETY).

pub mod bpe;
pub mod pretok;
pub mod rank;
pub mod specials;
pub mod unicode;
pub use bpe::byte_pair_encode;
pub use rank::{Rank, RankLookup, RankMap, RANK_MAX};
pub use specials::SpecialTokens;
