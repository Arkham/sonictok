//! Shared scalar pretokenizer helpers used by the cl100k and o200k grammars.

/// Decode the UTF-8 char starting at `i`. Returns (char, byte_len).
#[inline]
pub(crate) fn char_at(input: &[u8], i: usize) -> (char, usize) {
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
pub(crate) fn is_cr_or_lf(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Start byte index of the char immediately before byte index `end`.
#[inline]
pub(crate) fn prev_char_start(input: &[u8], end: usize) -> usize {
    if end == 0 {
        return 0;
    }
    let mut i = end - 1;
    while i > 0 && (input[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d) starting at `at` (which must be the `'`).
/// Returns total byte length of the match including the apostrophe, or None.
pub(crate) fn match_contraction(input: &[u8], at: usize) -> Option<usize> {
    if input.get(at) != Some(&b'\'') {
        return None;
    }
    let rest = &input[at + 1..];
    let lc = |b: u8| b.to_ascii_lowercase();
    let g = |n: usize| rest.get(n).copied().map(lc);
    match (g(0), g(1)) {
        (Some(b's'), _) | (Some(b't'), _) | (Some(b'm'), _) | (Some(b'd'), _) => Some(2),
        (Some(b'r'), Some(b'e')) | (Some(b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => Some(3),
        _ => None,
    }
}

/// Alt: `\p{N}{1,3}` starting at `start` (caller checked the first char is a
/// number). Returns the end index.
#[inline]
pub(crate) fn scan_number(input: &[u8], start: usize) -> usize {
    let mut k = start;
    let mut count = 0;
    while k < input.len() && count < 3 {
        let (ck, wk) = char_at(input, k);
        if crate::unicode::is_number(ck) {
            k += wk;
            count += 1;
        } else {
            break;
        }
    }
    k
}

/// Alt: ` ?[^\s\p{L}\p{N}]+(trailing)*` where `trailing` is `[\r\n]` for cl100k
/// or `[\r\n/]` for o200k. Returns Some(end) if matched, else None.
#[inline]
pub(crate) fn scan_punct(input: &[u8], start: usize, slash_in_tail: bool) -> Option<usize> {
    let (c0, w0) = char_at(input, start);
    let mut j = start;
    let mut c = c0;
    if c == ' ' {
        let nj = j + w0;
        if nj < input.len() {
            let (c2, _) = char_at(input, nj);
            if is_other(c2) {
                j = nj;
                c = c2;
            }
        }
    }
    if is_other(c) {
        let mut k = j;
        while k < input.len() {
            let (ck, wk) = char_at(input, k);
            if is_other(ck) {
                k += wk;
            } else {
                break;
            }
        }
        // trailing [\r\n]* (+ '/' for o200k)
        while k < input.len() {
            let (ck, wk) = char_at(input, k);
            if is_cr_or_lf(ck) || (slash_in_tail && ck == '/') {
                k += wk;
            } else {
                break;
            }
        }
        Some(k)
    } else {
        None
    }
}

/// `[^\s\p{L}\p{N}]` — neither whitespace, letter, nor number.
#[inline]
fn is_other(c: char) -> bool {
    !crate::unicode::is_whitespace(c)
        && !crate::unicode::is_letter(c)
        && !crate::unicode::is_number(c)
}

/// Alts: `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`, starting at `start` (caller checked
/// the first char is whitespace). Returns the end index.
#[inline]
pub(crate) fn scan_whitespace(input: &[u8], start: usize) -> usize {
    // \s*[\r\n]+ — a whitespace prefix that ends at the last CR/LF.
    let mut last_nl_end: Option<usize> = None;
    let mut t = start;
    while t < input.len() {
        let (ct, wt) = char_at(input, t);
        if !crate::unicode::is_whitespace(ct) {
            break;
        }
        t += wt;
        if is_cr_or_lf(ct) {
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
        return end;
    }
    // \s+(?!\S) / \s+: take the maximal whitespace run; if a non-space follows,
    // leave its LAST whitespace char to join the following word.
    let mut end = start;
    while end < input.len() {
        let (ce, we) = char_at(input, end);
        if crate::unicode::is_whitespace(ce) {
            end += we;
        } else {
            break;
        }
    }
    if end < input.len() {
        let last_start = prev_char_start(input, end);
        if last_start > start {
            return last_start;
        }
    }
    end
}
