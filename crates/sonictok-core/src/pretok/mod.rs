//! Pretokenization: split input bytes into "pieces" (byte spans) at grammar
//! boundaries. A `Pretokenizer` yields byte ranges into the input.
pub mod cl100k;

/// Yields the next piece as a byte range `[start, end)` into `input`, or None at end.
/// Implementations must be deterministic and cover the whole input with no gaps.
pub trait Pretokenizer {
    fn next_piece(&mut self, input: &[u8]) -> Option<(usize, usize)>;
}
