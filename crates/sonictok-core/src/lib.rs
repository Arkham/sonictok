//! sonictok-core: dependency-free, allocation-light exact BPE engine.
#![deny(unsafe_op_in_unsafe_fn)] // the only unsafe is pretok::char_at (documented SAFETY).

pub mod rank;
pub mod unicode;
pub use rank::{Rank, RankLookup, RankMap, RANK_MAX};
