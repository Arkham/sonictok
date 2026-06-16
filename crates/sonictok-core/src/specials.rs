//! Special-token registry and a scanner that finds the next special token
//! occurrence in input. Specials are matched as exact literal byte strings.
use crate::rank::Rank;

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// (literal bytes, id), kept sorted by descending length so longest-match wins.
    entries: Vec<(Vec<u8>, Rank)>,
}

impl SpecialTokens {
    pub fn new(mut entries: Vec<(Vec<u8>, Rank)>) -> Self {
        entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        Self { entries }
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], Rank)> {
        self.entries.iter().map(|(b, r)| (b.as_slice(), *r))
    }
    pub fn id_of(&self, name: &[u8]) -> Option<Rank> {
        self.entries.iter().find(|(b, _)| b == name).map(|(_, r)| *r)
    }
    pub fn name_of(&self, id: Rank) -> Option<&[u8]> {
        self.entries.iter().find(|(_, r)| *r == id).map(|(b, _)| b.as_slice())
    }

    /// Find the earliest special-token occurrence at or after `from`, restricted
    /// to the `allowed` set (by id). Returns (start, end, id).
    pub fn find_next(
        &self,
        input: &[u8],
        from: usize,
        allowed: &dyn Fn(Rank) -> bool,
    ) -> Option<(usize, usize, Rank)> {
        let mut best: Option<(usize, usize, Rank)> = None;
        for (bytes, id) in self.iter() {
            if !allowed(id) {
                continue;
            }
            if let Some(off) = find_sub(&input[from..], bytes) {
                let s = from + off;
                let cand = (s, s + bytes.len(), id);
                best = match best {
                    Some(b) if b.0 <= cand.0 => Some(b),
                    _ => Some(cand),
                };
            }
        }
        best
    }
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn specials() -> SpecialTokens {
        SpecialTokens::new(vec![
            (b"<|endoftext|>".to_vec(), 100257),
            (b"<|endofprompt|>".to_vec(), 100276),
        ])
    }
    #[test]
    fn lookup() {
        let s = specials();
        assert_eq!(s.id_of(b"<|endoftext|>"), Some(100257));
        assert_eq!(s.name_of(100276), Some(&b"<|endofprompt|>"[..]));
    }
    #[test]
    fn find_first_allowed() {
        let s = specials();
        let input = b"a<|endoftext|>b<|endofprompt|>";
        let all = |_id| true;
        assert_eq!(s.find_next(input, 0, &all), Some((1, 14, 100257)));
    }
    #[test]
    fn respects_allowed_set() {
        let s = specials();
        let input = b"a<|endoftext|>b<|endofprompt|>";
        let only_prompt = |id| id == 100276;
        assert_eq!(s.find_next(input, 0, &only_prompt), Some((15, 30, 100276)));
    }
}
