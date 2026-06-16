//! Linear-time exact BPE via the `bpe` crate's backtracking algorithm (the same
//! one quicktok and bpe-openai use; byte-identical to tiktoken). Instead of
//! tiktoken's O(n·merges) iterative merge, this is greedy **longest-match** of a
//! token prefix (a trie walk) plus a memoized `is_valid_token_pair` check, with
//! backtracking only on the rare incompatible pair.
//!
//! Built once at load; immutable afterwards except the `is_valid` memo, which is
//! a pure-function cache updated via relaxed atomics (so `&self` encode is
//! `Sync` — concurrent callers see an old-or-new entry, both correct).
use crate::hash::FxHashMap;
use crate::rank::{RANK_MAX, Rank};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

const IVBITS: u32 = 20; // 2^20 memo slots (4 MB of AtomicU32)
const IV_TAGBITS: u32 = 36 - IVBITS;

/// Bijective 36-bit mixer (self-inverse xorshift / odd-multiply / xorshift), so
/// index bits + tag bits reconstruct the key exactly — a tagged slot never
/// aliases a different key, keeping the memo exact.
#[inline]
fn mix36(k: u64) -> u64 {
    let mut m = k ^ (k >> 18);
    m = m.wrapping_mul((0x9E37_79B9_7F4A_7C15 & 0xF_FFFF_FFFF) | 1) & 0xF_FFFF_FFFF;
    m ^ (m >> 18)
}

pub struct Vocab {
    tlen: Vec<u8>, // token byte length (<= 255)
    n: u32,
    // byte trie (open-addressed edges)
    root_child: [u32; 256], // direct edges from root (node 0)
    etab: Vec<u64>,         // slot: (key+1)<<32 | child, key = node<<8|byte; 0 = empty
    emask: u32,
    tnode_tok: Vec<u32>, // per node: token id ending here, or RANK_MAX
    // first-2-bytes direct tables (zero probes for the common short prefixes):
    // r2node[b0<<8|b1] = trie node after those 2 bytes; r2best = deepest token in
    // the first <= 2 bytes.
    r2node: Vec<u32>,
    r2best: Vec<u32>,
    // 2-byte trie: one slot load consumes 2 bytes. e2 keyed on node<<16|b1b2.
    e2key: Vec<u64>, // key+1, 0 = empty
    e2val: Vec<u64>, // child<<32 | best_token (best = RANK_MAX if none)
    e2mask: u32,
    // odd-depth tokens: a token ending one byte past an even-depth node.
    otab: Vec<u64>, // (key+1)<<18 | token, key = node<<8|byte; 0 = empty
    omask: u32,
    // backtracking tables
    npm: Vec<u32>,          // next_prefix[id]: longest proper-prefix token, or MAX
    split: Vec<(u32, u32)>, // the two tokens id merged from, or (id, id)
    plk: Vec<u64>,          // pair_lookup: ((t1<<18|t2)+1)<<19 | mergedid; 0 = empty
    plmask: u32,
    // is_valid_token_pair memo (pure-function cache)
    ivm: Vec<AtomicU32>,
    ivmask: usize,
}

#[inline]
fn mul_shift(k: u64, mask: u32) -> u32 {
    (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as u32 & mask
}

impl Vocab {
    #[inline]
    pub fn n_tokens(&self) -> usize {
        self.n as usize
    }

    #[inline]
    fn edge(&self, node: u32, b: u8) -> u32 {
        if node == 0 {
            return self.root_child[b as usize];
        }
        let k = ((node as u64) << 8) | b as u64;
        let key = (k + 1) as u32;
        let mut i = mul_shift(k, self.emask);
        loop {
            let s = self.etab[i as usize];
            if s == 0 {
                return 0;
            }
            if (s >> 32) as u32 == key {
                return s as u32;
            }
            i = (i + 1) & self.emask;
        }
    }

    /// Longest token that is a prefix of `text` (greedy match), or RANK_MAX.
    /// The first 2 bytes resolve via a direct table (no probes); the tail walks
    /// the byte trie.
    #[inline]
    fn next_match(&self, text: &[u8]) -> u32 {
        let len = text.len();
        if len == 0 {
            return RANK_MAX;
        }
        if len == 1 {
            let n = self.root_child[text[0] as usize];
            return if n != 0 {
                self.tnode_tok[n as usize]
            } else {
                RANK_MAX
            };
        }
        let idx = ((text[0] as usize) << 8) | text[1] as usize;
        let mut node = self.r2node[idx];
        let mut best = self.r2best[idx];
        let mut i = 2;
        while node != 0 && i + 1 < len {
            let k = ((node as u64) << 16) | ((text[i] as u64) << 8) | text[i + 1] as u64;
            let want = k + 1;
            let mut h = mul_shift(k, self.e2mask);
            let mut found = false;
            loop {
                let ek = self.e2key[h as usize];
                if ek == 0 {
                    break;
                }
                if ek == want {
                    found = true;
                    break;
                }
                h = (h + 1) & self.e2mask;
            }
            if !found {
                // no 2-byte step: an odd-depth token may extend one byte
                let o = self.odd_lookup(node, text[i]);
                if o != RANK_MAX {
                    best = o;
                }
                return best;
            }
            let val = self.e2val[h as usize];
            let b = val as u32;
            if b != RANK_MAX {
                best = b;
            }
            node = (val >> 32) as u32;
            i += 2;
        }
        if node != 0 && i < len {
            let o = self.odd_lookup(node, text[i]);
            if o != RANK_MAX {
                best = o;
            }
        }
        best
    }

    #[inline]
    fn odd_lookup(&self, node: u32, b: u8) -> u32 {
        if self.otab.is_empty() {
            return RANK_MAX;
        }
        let k = ((node as u64) << 8) | b as u64;
        let want = k + 1;
        let mut h = mul_shift(k, self.omask);
        loop {
            let s = self.otab[h as usize];
            if s == 0 {
                return RANK_MAX;
            }
            if (s >> 18) == want {
                return (s & 0x3_FFFF) as u32;
            }
            h = (h + 1) & self.omask;
        }
    }

    #[inline]
    fn next_prefix(&self, id: u32) -> u32 {
        self.npm[id as usize]
    }

    #[inline]
    fn pl_get(&self, key: u64) -> u32 {
        // key = t1<<18 | t2
        let want = (key + 1) << 19;
        let mut i = mul_shift(key, self.plmask);
        loop {
            let s = self.plk[i as usize];
            if s == 0 {
                return RANK_MAX;
            }
            if (s & !0x7_FFFF) == want {
                return (s & 0x7_FFFF) as u32;
            }
            i = (i + 1) & self.plmask;
        }
    }

    fn pl_put(&mut self, key: u64, val: u32) {
        let want = (key + 1) << 19;
        let mut i = mul_shift(key, self.plmask);
        loop {
            let s = self.plk[i as usize];
            if s == 0 {
                self.plk[i as usize] = want | val as u64;
                return;
            }
            if (s & !0x7_FFFF) == want {
                self.plk[i as usize] = want | val as u64;
                return;
            }
            i = (i + 1) & self.plmask;
        }
    }

    /// Whether the sequence `[.., t1, t2]` is a valid BPE output (i.e. BPE would
    /// not have merged across the t1|t2 boundary). Memoized.
    #[inline]
    fn is_valid_token_pair(&self, t1: u32, t2: u32) -> bool {
        let m = mix36(((t1 as u64) << 18) | t2 as u64);
        let h = (m >> IV_TAGBITS) as usize & self.ivmask;
        let want = 0x8000_0000u32 | (((m as u32) & ((1 << IV_TAGBITS) - 1)) << 1);
        let s = self.ivm[h].load(Relaxed);
        if (s & 0xFFFF_FFFE) == want {
            return (s & 1) != 0;
        }
        let res = self.ivtp_slow(t1, t2);
        self.ivm[h].store(want | res as u32, Relaxed);
        res
    }

    fn ivtp_slow(&self, mut t1: u32, mut t2: u32) -> bool {
        let mut limit = RANK_MAX;
        loop {
            let c = self.pl_get(((t1 as u64) << 18) | t2 as u64);
            if c != RANK_MAX && c < limit {
                return false;
            }
            if t1 > t2 {
                limit = t1;
                t1 = self.split[t1 as usize].1;
                if t1 == limit {
                    limit = t2 + 1;
                    t2 = self.split[t2 as usize].0;
                    if t2 + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = t2 + 1;
                t2 = self.split[t2 as usize].0;
                if t2 + 1 == limit {
                    limit = t1;
                    t1 = self.split[t1 as usize].1;
                    if t1 == limit {
                        return true;
                    }
                }
            }
        }
    }

    /// Encode one pretokenized piece into token ids, appended to `out`.
    pub fn encode(&self, text: &[u8], out: &mut Vec<Rank>) {
        if text.is_empty() {
            return;
        }
        let first = self.next_match(text);
        self.encode_with_first(text, first, out);
    }

    fn encode_with_first(&self, text: &[u8], first: u32, out: &mut Vec<Rank>) {
        let len = text.len();
        if len == 0 {
            return;
        }
        let out_start = out.len();
        // Greedy fast path: backtracking only triggers on an invalid pair (rare),
        // so for any greedily-tokenizable piece the bitfield machinery is skipped.
        {
            let mut pos = 0usize;
            let mut last = RANK_MAX;
            let mut nt = first;
            let mut ok = true;
            while pos < len {
                let token = nt;
                if last != RANK_MAX && !self.is_valid_token_pair(last, token) {
                    ok = false;
                    break;
                }
                out.push(token);
                last = token;
                pos += self.tlen[token as usize] as usize;
                nt = if pos < len {
                    self.next_match(&text[pos..])
                } else {
                    RANK_MAX
                };
            }
            if ok {
                return;
            }
            out.truncate(out_start);
        }
        // Full backtracking (rare).
        let mut toks: Vec<u32> = Vec::new();
        let words = (len + 1 + 63) >> 6;
        let mut bf = vec![u64::MAX; words];
        let is_set = |bf: &[u64], b: usize| (bf[b >> 6] >> (b & 63)) & 1 != 0;

        let mut pos = 0usize;
        let mut next_token = first;
        while next_token != RANK_MAX {
            let mut token = next_token;
            let last = toks.last().copied().unwrap_or(RANK_MAX);
            loop {
                let end = pos + self.tlen[token as usize] as usize;
                if is_set(&bf, end) && (last == RANK_MAX || self.is_valid_token_pair(last, token)) {
                    toks.push(token);
                    pos = end;
                    next_token = if pos < len {
                        self.next_match(&text[pos..])
                    } else {
                        RANK_MAX
                    };
                    break;
                }
                let shorter = self.next_prefix(token);
                if shorter != RANK_MAX {
                    token = shorter;
                    continue;
                }
                bf[pos >> 6] &= !(1u64 << (pos & 63));
                if !toks.is_empty() {
                    toks.pop();
                    pos -= self.tlen[last as usize] as usize;
                }
                next_token = last;
                break;
            }
        }
        out.extend_from_slice(&toks);
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (Vec<u8>, Rank)>) -> Self {
        // Collect tokens by id (rank order).
        let entries: Vec<(Vec<u8>, Rank)> = pairs.into_iter().collect();
        let n = entries.len() as u32;
        let mut tb: Vec<Vec<u8>> = vec![Vec::new(); n as usize];
        for (bytes, rank) in entries {
            tb[rank as usize] = bytes;
        }

        let mut tlen = vec![0u8; n as usize];
        let mut total_bytes = 0usize;
        let mut b2id: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for (id, t) in tb.iter().enumerate() {
            tlen[id] = t.len() as u8;
            total_bytes += t.len();
            b2id.insert(t.clone(), id as u32);
        }

        // Byte trie.
        let mut ecap = 1usize;
        while ecap < total_bytes * 2 {
            ecap <<= 1;
        }
        if ecap < 1024 {
            ecap = 1024;
        }
        let mut v = Vocab {
            tlen,
            n,
            root_child: [0u32; 256],
            etab: vec![0u64; ecap],
            emask: (ecap - 1) as u32,
            tnode_tok: vec![RANK_MAX; 1], // root = node 0
            r2node: Vec::new(),
            r2best: Vec::new(),
            e2key: Vec::new(),
            e2val: Vec::new(),
            e2mask: 0,
            otab: Vec::new(),
            omask: 0,
            npm: Vec::new(),
            split: Vec::new(),
            plk: Vec::new(),
            plmask: 0,
            ivm: Vec::new(),
            ivmask: 0,
        };
        for (id, t) in tb.iter().enumerate() {
            let mut node = 0u32;
            for &byte in t {
                node = v.edge_build(node, byte);
            }
            v.tnode_tok[node as usize] = id as u32;
        }

        // first-2-bytes direct tables.
        v.r2node = vec![0u32; 65536];
        v.r2best = vec![RANK_MAX; 65536];
        for b0 in 0..256u32 {
            let n1 = v.root_child[b0 as usize];
            if n1 == 0 {
                continue;
            }
            let tok1 = v.tnode_tok[n1 as usize];
            for b1 in 0..256u32 {
                let idx = (b0 << 8 | b1) as usize;
                let n2 = v.edge(n1, b1 as u8);
                if n2 != 0 {
                    v.r2node[idx] = n2;
                    let t2 = v.tnode_tok[n2 as usize];
                    v.r2best[idx] = if t2 != RANK_MAX { t2 } else { tok1 };
                } else {
                    v.r2node[idx] = 0;
                    v.r2best[idx] = tok1;
                }
            }
        }

        // 2-byte trie (e2) + odd-token side table (otab), from the complete trie.
        {
            let mut e2map: FxHashMap<u64, (u32, u32)> = FxHashMap::default();
            let mut otmap: FxHashMap<u64, u32> = FxHashMap::default();
            let mut path: Vec<u32> = Vec::new();
            for (id, t) in tb.iter().enumerate() {
                let len = t.len();
                path.clear();
                path.push(0);
                let mut node = 0u32;
                for (j, &byte) in t.iter().enumerate() {
                    node = if j == 0 {
                        v.root_child[byte as usize]
                    } else {
                        v.edge(node, byte)
                    };
                    path.push(node);
                }
                let mut d = 2;
                while d + 2 <= len {
                    let key = ((path[d] as u64) << 16) | ((t[d] as u64) << 8) | t[d + 1] as u64;
                    e2map.entry(key).or_insert_with(|| {
                        let b2 = v.tnode_tok[path[d + 2] as usize];
                        let best = if b2 != RANK_MAX {
                            b2
                        } else {
                            v.tnode_tok[path[d + 1] as usize]
                        };
                        (path[d + 2], best)
                    });
                    d += 2;
                }
                if len >= 3 && len % 2 == 1 {
                    let key = ((path[len - 1] as u64) << 8) | t[len - 1] as u64;
                    otmap.entry(key).or_insert(id as u32);
                }
            }
            // pack e2 into an open-addressed table (load factor ~0.5).
            let mut e2cap = 1024usize;
            while (e2map.len() as f64) / (e2cap as f64) > 0.5 {
                e2cap <<= 1;
            }
            v.e2key = vec![0u64; e2cap];
            v.e2val = vec![0u64; e2cap];
            v.e2mask = (e2cap - 1) as u32;
            for (k, (child, best)) in e2map {
                let mut h = mul_shift(k, v.e2mask);
                while v.e2key[h as usize] != 0 {
                    h = (h + 1) & v.e2mask;
                }
                v.e2key[h as usize] = k + 1;
                v.e2val[h as usize] = ((child as u64) << 32) | best as u64;
            }
            // pack otab.
            let mut ocap = 1024usize;
            while (otmap.len() as f64) / (ocap as f64) > 0.5 {
                ocap <<= 1;
            }
            v.otab = vec![0u64; ocap];
            v.omask = (ocap - 1) as u32;
            for (k, tok) in otmap {
                let mut h = mul_shift(k, v.omask);
                while v.otab[h as usize] != 0 {
                    h = (h + 1) & v.omask;
                }
                v.otab[h as usize] = ((k + 1) << 18) | tok as u64;
            }
        }

        // next_prefix[id]: longest proper-prefix token.
        v.npm = vec![RANK_MAX; n as usize];
        for (id, t) in tb.iter().enumerate() {
            let mut node = 0u32;
            let mut best = RANK_MAX;
            let upto = t.len().saturating_sub(1);
            for &byte in &t[..upto] {
                node = v.edge(node, byte);
                if node == 0 {
                    break;
                }
                let tk = v.tnode_tok[node as usize];
                if tk != RANK_MAX {
                    best = tk;
                }
            }
            v.npm[id] = best;
        }

        // pair_lookup + split + (construction-time) is_valid memo.
        let mut pcap = 1usize;
        while pcap < (n as usize) * 2 {
            pcap <<= 1;
        }
        if pcap < 1024 {
            pcap = 1024;
        }
        v.plk = vec![0u64; pcap];
        v.plmask = (pcap - 1) as u32;
        v.ivm = (0..(1usize << IVBITS)).map(|_| AtomicU32::new(0)).collect();
        v.ivmask = (1usize << IVBITS) - 1;
        v.split = Vec::with_capacity(n as usize);
        for id in 0..n {
            let t = &tb[id as usize];
            let mut token1 = v.npm[id as usize];
            let mut done = false;
            while token1 != RANK_MAX {
                let l1 = v.tlen[token1 as usize] as usize;
                let token2 = b2id.get(&t[l1..]).copied().unwrap_or(RANK_MAX);
                if token2 != RANK_MAX
                    && token1 < id
                    && token2 < id
                    && v.is_valid_token_pair(token1, token2)
                {
                    v.pl_put(((token1 as u64) << 18) | token2 as u64, id);
                    v.split.push((token1, token2));
                    done = true;
                    break;
                }
                token1 = v.npm[token1 as usize];
            }
            if !done {
                v.split.push((id, id));
            }
        }
        // Construction-time is_valid calls used partial tables; reset the memo.
        for slot in &v.ivm {
            slot.store(0, Relaxed);
        }
        v
    }

    fn edge_build(&mut self, node: u32, b: u8) -> u32 {
        if node == 0 {
            if self.root_child[b as usize] != 0 {
                return self.root_child[b as usize];
            }
            let c = self.tnode_tok.len() as u32;
            self.tnode_tok.push(RANK_MAX);
            self.root_child[b as usize] = c;
            return c;
        }
        let k = ((node as u64) << 8) | b as u64;
        let key = (k + 1) as u32;
        let mut i = mul_shift(k, self.emask);
        loop {
            let s = self.etab[i as usize];
            if s == 0 {
                break;
            }
            if (s >> 32) as u32 == key {
                return s as u32;
            }
            i = (i + 1) & self.emask;
        }
        let child = self.tnode_tok.len() as u32;
        self.tnode_tok.push(RANK_MAX);
        self.etab[i as usize] = ((key as u64) << 32) | child as u64;
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(v: &Vocab, piece: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        v.encode(piece, &mut out);
        out
    }

    fn toy() -> Vocab {
        // Ranks must be contiguous 0..n-1 (as in tiktoken). bytes 0..256 = id b;
        // merges "ab"=256, "abc"=257.
        let mut pairs: Vec<(Vec<u8>, Rank)> =
            (0u16..256).map(|b| (vec![b as u8], b as Rank)).collect();
        pairs.push((b"ab".to_vec(), 256));
        pairs.push((b"abc".to_vec(), 257));
        Vocab::from_pairs(pairs)
    }

    #[test]
    fn single_byte() {
        assert_eq!(enc(&toy(), b"a"), vec![b'a' as u32]);
    }
    #[test]
    fn whole_token() {
        assert_eq!(enc(&toy(), b"abc"), vec![257]);
    }
    #[test]
    fn longest_match() {
        // "abx": greedy longest-match "ab"(256) then "x"
        assert_eq!(enc(&toy(), b"abx"), vec![256, b'x' as u32]);
    }
    #[test]
    fn no_merge() {
        assert_eq!(enc(&toy(), b"xy"), vec![b'x' as u32, b'y' as u32]);
    }
}
