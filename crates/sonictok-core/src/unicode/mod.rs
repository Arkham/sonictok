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
    fn ranges_are_sorted_nonoverlapping() {
        for t in [tables::LETTER, tables::NUMBER, tables::WHITE_SPACE] {
            for w in t.windows(2) {
                assert!(w[0].1 < w[1].0, "ranges must be sorted & disjoint");
            }
            for &(lo, hi) in t {
                assert!(lo <= hi);
            }
        }
    }
}
