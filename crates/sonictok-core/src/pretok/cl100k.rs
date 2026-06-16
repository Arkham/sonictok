//! Hand-written scalar scanner for the cl100k_base pretokenizer grammar:
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
use crate::pretok::Pretokenizer;
use crate::pretok::common::{
    char_at, is_cr_or_lf, match_contraction, scan_number, scan_punct, scan_whitespace,
};
use crate::unicode::{is_letter, is_number, is_whitespace};

#[derive(Default)]
pub struct Cl100kPretokenizer {
    pos: usize,
}

impl Cl100kPretokenizer {
    pub fn new() -> Self {
        Self { pos: 0 }
    }
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// End index of the cl100k piece starting at `start` (scalar; handles Unicode).
pub fn piece_end(input: &[u8], start: usize) -> usize {
    let (c0, w0) = char_at(input, start);

    // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c0 == '\''
        && let Some(len) = match_contraction(input, start)
    {
        return start + len;
    }

    // Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+
    {
        let mut j = start;
        let mut c = c0;
        if !is_cr_or_lf(c) && !is_letter(c) && !is_number(c) {
            let nj = j + w0;
            if nj < input.len() {
                let (c2, _) = char_at(input, nj);
                if is_letter(c2) {
                    j = nj;
                    c = c2;
                }
            }
        }
        if is_letter(c) {
            let mut k = j;
            while k < input.len() {
                let (ck, wk) = char_at(input, k);
                if is_letter(ck) {
                    k += wk;
                } else {
                    break;
                }
            }
            return k;
        }
    }

    // Alt 3: \p{N}{1,3}
    if is_number(c0) {
        return scan_number(input, start);
    }

    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    if let Some(end) = scan_punct(input, start, false) {
        return end;
    }

    // Alt 5-7: whitespace cascade
    if is_whitespace(c0) {
        return scan_whitespace(input, start);
    }

    // Fallback: unreachable for valid inputs; guarantee progress.
    start + w0.max(1)
}

impl Pretokenizer for Cl100kPretokenizer {
    fn next_piece(&mut self, input: &[u8]) -> Option<(usize, usize)> {
        let start = self.pos;
        if start >= input.len() {
            return None;
        }
        self.pos = piece_end(input, start);
        Some((start, self.pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<&str> {
        let b = s.as_bytes();
        let mut p = Cl100kPretokenizer::new();
        let mut out = vec![];
        while let Some((a, z)) = p.next_piece(b) {
            assert!(z > a, "piece must make progress");
            out.push(std::str::from_utf8(&b[a..z]).unwrap());
        }
        out
    }

    #[test]
    fn words_and_spaces() {
        assert_eq!(pieces("hello world"), vec!["hello", " world"]);
    }
    #[test]
    fn contractions() {
        assert_eq!(pieces("I'm don't"), vec!["I", "'m", " don", "'t"]);
    }
    #[test]
    fn numbers_triples() {
        assert_eq!(pieces("1234"), vec!["123", "4"]);
    }
    #[test]
    fn punct_and_newlines() {
        assert_eq!(pieces("a!!\n"), vec!["a", "!!\n"]);
    }
    #[test]
    fn leading_space_run() {
        assert_eq!(pieces("a   b"), vec!["a", "  ", " b"]);
    }
    #[test]
    fn covers_whole_input() {
        for s in ["", "x", "  ", "日本語 test 99", "\n\n\t  end"] {
            let total: usize = pieces(s).iter().map(|p| p.len()).sum();
            assert_eq!(total, s.len(), "input {s:?} not fully covered");
        }
    }
}
