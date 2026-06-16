//! Hand-written scalar scanner for the cl100k_base pretokenizer grammar:
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
use crate::pretok::Pretokenizer;
use crate::unicode::{is_letter, is_number, is_whitespace};

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
impl Default for Cl100kPretokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode the UTF-8 char starting at `i`. Returns (char, byte_len).
#[inline]
fn char_at(input: &[u8], i: usize) -> (char, usize) {
    // SAFETY: `input` always originates from a `&str` (encode operates on
    // `&str`), and every alternative advances by whole UTF-8 char widths, so `i`
    // is always on a char boundary; the tail is therefore valid UTF-8.
    let s = unsafe { std::str::from_utf8_unchecked(&input[i..]) };
    match s.chars().next() {
        Some(c) => (c, c.len_utf8()),
        None => ('\u{0}', 0),
    }
}

#[inline]
fn is_cr_or_lf(c: char) -> bool {
    c == '\r' || c == '\n'
}

impl Pretokenizer for Cl100kPretokenizer {
    fn next_piece(&mut self, input: &[u8]) -> Option<(usize, usize)> {
        let start = self.pos;
        if start >= input.len() {
            return None;
        }
        let (c0, w0) = char_at(input, start);

        // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c0 == '\''
            && let Some(len) = match_contraction(input, start)
        {
            self.pos = start + len;
            return Some((start, self.pos));
        }

        // Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+
        {
            let mut j = start;
            let mut c = c0;
            // optional single leading non-(CR/LF/letter/number)
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
                // consume \p{L}+
                let mut k = j;
                while k < input.len() {
                    let (ck, wk) = char_at(input, k);
                    if is_letter(ck) {
                        k += wk;
                    } else {
                        break;
                    }
                }
                self.pos = k;
                return Some((start, k));
            }
        }

        // Alt 3: \p{N}{1,3}
        if is_number(c0) {
            let mut k = start;
            let mut count = 0;
            while k < input.len() && count < 3 {
                let (ck, wk) = char_at(input, k);
                if is_number(ck) {
                    k += wk;
                    count += 1;
                } else {
                    break;
                }
            }
            self.pos = k;
            return Some((start, k));
        }

        // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let mut j = start;
            let mut c = c0;
            if c == ' ' {
                let nj = j + w0;
                if nj < input.len() {
                    let (c2, _) = char_at(input, nj);
                    if !is_whitespace(c2) && !is_letter(c2) && !is_number(c2) {
                        j = nj;
                        c = c2;
                    }
                }
            }
            if !is_whitespace(c) && !is_letter(c) && !is_number(c) {
                let mut k = j;
                while k < input.len() {
                    let (ck, wk) = char_at(input, k);
                    if !is_whitespace(ck) && !is_letter(ck) && !is_number(ck) {
                        k += wk;
                    } else {
                        break;
                    }
                }
                // trailing [\r\n]*
                while k < input.len() {
                    let (ck, wk) = char_at(input, k);
                    if is_cr_or_lf(ck) {
                        k += wk;
                    } else {
                        break;
                    }
                }
                self.pos = k;
                return Some((start, k));
            }
        }

        // Alt 5: \s*[\r\n]+  — a whitespace prefix that ends at the last CR/LF.
        if is_whitespace(c0) {
            let mut last_nl_end: Option<usize> = None;
            let mut t = start;
            while t < input.len() {
                let (ct, wt) = char_at(input, t);
                if !is_whitespace(ct) {
                    break;
                }
                t += wt;
                if is_cr_or_lf(ct) {
                    // extend over a contiguous CR/LF run
                    while t < input.len() {
                        let (cu, wu) = char_at(input, t);
                        if is_cr_or_lf(cu) {
                            t += wu;
                        } else {
                            break;
                        }
                    }
                    last_nl_end = Some(t);
                }
            }
            if let Some(end) = last_nl_end {
                self.pos = end;
                return Some((start, end));
            }
            // Alt 6: \s+(?!\S)  / Alt 7: \s+. Take the maximal whitespace run; if a
            // non-space follows, leave its LAST whitespace char to join the word.
            let mut end = start;
            while end < input.len() {
                let (ce, we) = char_at(input, end);
                if is_whitespace(ce) {
                    end += we;
                } else {
                    break;
                }
            }
            if end < input.len() {
                let last_start = prev_char_start(input, end);
                if last_start > start {
                    self.pos = last_start;
                    return Some((start, last_start));
                }
                // single whitespace before a non-space: alt 6 fails, alt 7 takes it
            }
            self.pos = end;
            return Some((start, end));
        }

        // Fallback: unreachable for valid inputs; consume one char to guarantee
        // progress. Oracle-diff would flag any reachable case.
        self.pos = start + w0.max(1);
        Some((start, self.pos))
    }
}

/// Start byte index of the char immediately before byte index `end`.
#[inline]
fn prev_char_start(input: &[u8], end: usize) -> usize {
    if end == 0 {
        return 0;
    }
    let mut i = end - 1;
    while i > 0 && (input[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d) starting at `start` (which is the `'`).
/// Returns total byte length of the match including the apostrophe, or None.
fn match_contraction(input: &[u8], start: usize) -> Option<usize> {
    let rest = &input[start + 1..];
    let lc = |b: u8| b.to_ascii_lowercase();
    let g = |n: usize| rest.get(n).copied().map(lc);
    match (g(0), g(1)) {
        (Some(b's'), _) | (Some(b't'), _) | (Some(b'm'), _) | (Some(b'd'), _) => Some(2),
        (Some(b'r'), Some(b'e')) | (Some(b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => Some(3),
        _ => None,
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
