//! Hand-written scalar scanner for the o200k_base / o200k_harmony grammar:
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:contr)?
//! |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:contr)?
//! |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
use crate::pretok::Pretokenizer;
use crate::pretok::common::{
    char_at, is_cr_or_lf, match_contraction, prev_char_start, scan_number, scan_punct,
    scan_whitespace,
};
use crate::unicode::{is_letter, is_number, is_o200k_lower, is_o200k_upper, is_whitespace};

#[derive(Default)]
pub struct O200kPretokenizer {
    pos: usize,
}

impl O200kPretokenizer {
    pub fn new() -> Self {
        Self { pos: 0 }
    }
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Greedy `UPPER* LOWER+` with backtracking, starting at `q`. Returns end index
/// of the match, or None (no LOWER char reachable).
fn match_upper_star_lower_plus(input: &[u8], q: usize) -> Option<usize> {
    // greedy UPPER*
    let mut i = q;
    while i < input.len() {
        let (c, w) = char_at(input, i);
        if is_o200k_upper(c) {
            i += w;
        } else {
            break;
        }
    }
    // need LOWER+ (>=1); backtrack the UPPER* run one char at a time until the
    // char at the cursor is in LOWER, then consume the maximal LOWER run.
    loop {
        if i < input.len() {
            let (c, _) = char_at(input, i);
            if is_o200k_lower(c) {
                return Some(scan_lower_run(input, i));
            }
        }
        if i <= q {
            return None;
        }
        i = prev_char_start(input, i);
    }
}

/// Greedy `UPPER+ LOWER*` starting at `q`. Returns end index, or None (no UPPER).
fn match_upper_plus_lower_star(input: &[u8], q: usize) -> Option<usize> {
    if q >= input.len() {
        return None;
    }
    let (c0, _) = char_at(input, q);
    if !is_o200k_upper(c0) {
        return None;
    }
    let mut i = q;
    while i < input.len() {
        let (c, w) = char_at(input, i);
        if is_o200k_upper(c) {
            i += w;
        } else {
            break;
        }
    }
    Some(scan_lower_run(input, i))
}

/// Consume the maximal run of LOWER chars from `from`; returns the end index.
#[inline]
fn scan_lower_run(input: &[u8], from: usize) -> usize {
    let mut k = from;
    while k < input.len() {
        let (c, w) = char_at(input, k);
        if is_o200k_lower(c) {
            k += w;
        } else {
            break;
        }
    }
    k
}

/// Optional trailing contraction after a letter run.
#[inline]
fn with_contraction(input: &[u8], end: usize) -> usize {
    match match_contraction(input, end) {
        Some(len) => end + len,
        None => end,
    }
}

/// Match the two letter alternatives (A then B), each trying the optional
/// leading non-letter char (greedy) before falling back to no leading char.
/// Returns the end index of the piece starting at `start`, or None.
fn match_letters(input: &[u8], start: usize) -> Option<usize> {
    let (c0, w0) = char_at(input, start);
    let lead_ok = !is_cr_or_lf(c0) && !is_letter(c0) && !is_number(c0);
    let q_lead = start + w0;

    // Alt A: UPPER* LOWER+
    if lead_ok
        && q_lead < input.len()
        && let Some(e) = match_upper_star_lower_plus(input, q_lead)
    {
        return Some(with_contraction(input, e));
    }
    if let Some(e) = match_upper_star_lower_plus(input, start) {
        return Some(with_contraction(input, e));
    }
    // Alt B: UPPER+ LOWER*
    if lead_ok
        && q_lead < input.len()
        && let Some(e) = match_upper_plus_lower_star(input, q_lead)
    {
        return Some(with_contraction(input, e));
    }
    if let Some(e) = match_upper_plus_lower_star(input, start) {
        return Some(with_contraction(input, e));
    }
    None
}

/// End index of the o200k piece starting at `start` (scalar; handles Unicode).
pub fn piece_end(input: &[u8], start: usize) -> usize {
    let (c0, w0) = char_at(input, start);

    // Alts 1-2: case-aware letter sequences (+ optional contraction).
    if let Some(end) = match_letters(input, start) {
        return end;
    }
    // Alt 3: \p{N}{1,3}
    if is_number(c0) {
        return scan_number(input, start, 3);
    }
    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*   (note: '/' in the trailing class)
    if let Some(end) = scan_punct(input, start, true) {
        return end;
    }
    // Alt 5-7: whitespace cascade
    if is_whitespace(c0) {
        return scan_whitespace(input, start);
    }
    // Fallback: unreachable for valid inputs; guarantee progress.
    start + w0.max(1)
}

impl Pretokenizer for O200kPretokenizer {
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
        let mut p = O200kPretokenizer::new();
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
    fn camel_case_splits() {
        assert_eq!(pieces("camelCase"), vec!["camel", "Case"]);
    }
    #[test]
    fn upper_run() {
        assert_eq!(pieces("ABC"), vec!["ABC"]);
    }
    #[test]
    fn contraction_attaches() {
        // o200k keeps contractions attached (unlike cl100k's separate alt).
        assert_eq!(pieces("don't"), vec!["don't"]);
        assert_eq!(pieces("I'm"), vec!["I'm"]);
    }
    #[test]
    fn numbers_triples() {
        assert_eq!(pieces("1234"), vec!["123", "4"]);
    }
    #[test]
    fn covers_whole_input() {
        for s in [
            "",
            "x",
            "  ",
            "日本語 test 99",
            "\n\n\t  end",
            "HTTPSConnection",
            "a/b/c",
        ] {
            let total: usize = pieces(s).iter().map(|p| p.len()).sum();
            assert_eq!(total, s.len(), "input {s:?} not fully covered");
        }
    }
}
