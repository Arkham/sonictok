//! Pretokenization: split input bytes into "pieces" (byte spans) at grammar
//! boundaries. A `Pretokenizer` yields byte ranges into the input.
pub mod cl100k;
pub(crate) mod common;
pub mod o200k;

/// Yields the next piece as a byte range `[start, end)` into `input`, or None at end.
/// Implementations must be deterministic and cover the whole input with no gaps.
pub trait Pretokenizer {
    fn next_piece(&mut self, input: &[u8]) -> Option<(usize, usize)>;
}

/// Which fixed pretokenizer grammar an encoding uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grammar {
    Cl100k,
    O200k,
    /// cl100k grammar but single-digit numbers (\p{N}); used by qwen3.
    Qwen,
}

/// A grammar-dispatched scanner; the engine constructs the right one per call.
pub enum Scanner {
    Cl100k(cl100k::Cl100kPretokenizer),
    O200k(o200k::O200kPretokenizer),
}

impl Scanner {
    pub fn new(g: Grammar) -> Self {
        match g {
            Grammar::Cl100k => Scanner::Cl100k(cl100k::Cl100kPretokenizer::new()),
            Grammar::Qwen => Scanner::Cl100k(cl100k::Cl100kPretokenizer::qwen()),
            Grammar::O200k => Scanner::O200k(o200k::O200kPretokenizer::new()),
        }
    }
}

impl Pretokenizer for Scanner {
    #[inline]
    fn next_piece(&mut self, input: &[u8]) -> Option<(usize, usize)> {
        match self {
            Scanner::Cl100k(s) => s.next_piece(input),
            Scanner::O200k(s) => s.next_piece(input),
        }
    }
}
