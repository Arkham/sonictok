//! Unicode class lookup over generated ranges. Rung 0 uses binary search;
//! Rung 4 replaces this with table-driven SIMD classification.
mod tables;

#[inline]
fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    ranges
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

#[inline]
pub fn is_letter(c: char) -> bool {
    in_ranges(c as u32, tables::LETTER)
}
#[inline]
pub fn is_number(c: char) -> bool {
    in_ranges(c as u32, tables::NUMBER)
}
#[inline]
pub fn is_whitespace(c: char) -> bool {
    in_ranges(c as u32, tables::WHITE_SPACE)
}
/// o200k upper-class: `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
#[inline]
pub fn is_o200k_upper(c: char) -> bool {
    in_ranges(c as u32, tables::O200K_UPPER)
}
/// o200k lower-class: `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
#[inline]
pub fn is_o200k_lower(c: char) -> bool {
    in_ranges(c as u32, tables::O200K_LOWER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_letters() {
        assert!(is_letter('a') && is_letter('Z'));
        assert!(!is_letter('1') && !is_letter(' ') && !is_letter('!'));
    }
    #[test]
    fn unicode_letters() {
        assert!(is_letter('é') && is_letter('日') && is_letter('Ω'));
    }
    #[test]
    fn numbers() {
        assert!(is_number('7') && is_number('٣') /* arabic-indic 3 */);
        assert!(!is_number('a'));
    }
    #[test]
    fn whitespace() {
        assert!(is_whitespace(' ') && is_whitespace('\t') && is_whitespace('\n'));
        assert!(is_whitespace('\u{00A0}') /* nbsp */);
        assert!(!is_whitespace('a'));
    }
    #[test]
    fn o200k_classes() {
        // 'A' is Lu -> upper only; 'a' is Ll -> lower only; '9' neither.
        assert!(is_o200k_upper('A') && !is_o200k_lower('A'));
        assert!(is_o200k_lower('a') && !is_o200k_upper('a'));
        assert!(!is_o200k_upper('9') && !is_o200k_lower('9'));
        // combining mark (U+0301) is in BOTH classes.
        assert!(is_o200k_upper('\u{0301}') && is_o200k_lower('\u{0301}'));
    }
    #[test]
    fn ranges_are_sorted_nonoverlapping() {
        for t in [
            tables::LETTER,
            tables::NUMBER,
            tables::WHITE_SPACE,
            tables::O200K_UPPER,
            tables::O200K_LOWER,
        ] {
            for w in t.windows(2) {
                assert!(w[0].1 < w[1].0, "ranges must be sorted & disjoint");
            }
            for &(lo, hi) in t {
                assert!(lo <= hi);
            }
        }
    }
}
