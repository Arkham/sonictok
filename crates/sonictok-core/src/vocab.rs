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

// Packed e2 slot: [63]=used [62:38]=tag25 [37:18]=child20 [17:0]=best18.
const E2_USED: u64 = 1 << 63;
const E2_TAGMASK: u64 = (1 << 25) - 1;
const E2_CMP: u64 = E2_USED | (E2_TAGMASK << 38);
const E2_BEST_NONE: u32 = 0x3_FFFF;

/// Bijective 36-bit mixer (self-inverse xorshift / odd-multiply / xorshift), so
/// index bits + tag bits reconstruct the key exactly — a tagged slot never
/// aliases a different key, keeping the memo exact.
#[inline]
fn mix36(k: u64) -> u64 {
    // Pure odd-multiply by a 36-bit odd constant. Every caller reads only bits
    // [0,36) of the result (index from [e2tb,36) via >>+&mask, tag from [0,25)),
    // and those low bits are unaffected by the u64 wrap, so the final &2^36 mask
    // is dead — dropped to shave one op off the hash critical path. The multiply
    // is bijective mod 2^36 so index+tag still reconstruct the key exactly; the
    // e2 table is byte-identical to the masked version (construction unchanged).
    k.wrapping_mul((0x9E37_79B9_7F4A_7C15 & 0xF_FFFF_FFFF) | 1)
}

/// Exact op counters (feature `profile`) for the new BPE path.
#[cfg(feature = "profile")]
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static NEXT_MATCH: AtomicU64 = AtomicU64::new(0);
    pub static NM_STEPS: AtomicU64 = AtomicU64::new(0);
    pub static TOKENS: AtomicU64 = AtomicU64::new(0);
    pub static ISVALID: AtomicU64 = AtomicU64::new(0);
    pub static ISVALID_MISS: AtomicU64 = AtomicU64::new(0);
    pub static BACKTRACK: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub(super) fn inc(c: &AtomicU64) {
        c.fetch_add(1, Relaxed);
    }
    pub fn reset() {
        for c in [
            &NEXT_MATCH,
            &NM_STEPS,
            &TOKENS,
            &ISVALID,
            &ISVALID_MISS,
            &BACKTRACK,
        ] {
            c.store(0, Relaxed);
        }
    }
    pub fn snapshot() -> [(&'static str, u64); 6] {
        [
            ("next_match", NEXT_MATCH.load(Relaxed)),
            ("nm_steps", NM_STEPS.load(Relaxed)),
            ("tokens", TOKENS.load(Relaxed)),
            ("is_valid", ISVALID.load(Relaxed)),
            ("is_valid_miss", ISVALID_MISS.load(Relaxed)),
            ("backtrack", BACKTRACK.load(Relaxed)),
        ]
    }
}
#[cfg(feature = "profile")]
macro_rules! pf {
    ($c:ident) => {
        prof::inc(&prof::$c)
    };
}
#[cfg(not(feature = "profile"))]
macro_rules! pf {
    ($c:ident) => {{}};
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
    // 2-byte trie: ONE u64 slot load consumes 2 bytes (packed used+tag+child+best).
    // Keyed on node<<16|b1b2 via the bijective mix36; index+tag reconstruct the
    // key exactly (construction verifies no probe chain reaches a same-tag
    // stranger), so the table is exact.
    e2: Vec<u64>,
    e2mask: u32,
    e2tb: u32, // index shift = 36 - log2(slots)
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

thread_local! {
    /// (toks, bitfield) scratch for the rare backtracking path.
    static BT: std::cell::RefCell<(Vec<u32>, Vec<u64>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

#[inline]
fn mul_shift(k: u64, mask: u32) -> u32 {
    (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as u32 & mask
}

/// Longest run of consecutive occupied slots (treating the table as circular).
fn max_circular_run(slots: &[u64]) -> u64 {
    let mut run = 0u64;
    let mut maxrun = 0u64;
    let mut lead = 0u64;
    let mut open = true;
    for &s in slots {
        if s != 0 {
            run += 1;
            if run > maxrun {
                maxrun = run;
            }
        } else {
            if open {
                lead = run;
                open = false;
            }
            run = 0;
        }
    }
    if open {
        return slots.len() as u64; // fully occupied (can't happen at load < 1)
    }
    if run + lead > maxrun {
        maxrun = run + lead; // wraparound run
    }
    maxrun
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
        pf!(NEXT_MATCH);
        // SAFETY (hottest function, ~50% of encode): every index below is proven
        // in-bounds, so we elide bounds checks. idx = two u8s in [0,65535] and
        // r2node/r2best are exactly 65536 long; h is always masked with e2mask
        // and e2.len()==e2mask+1; the loop guard `i + 1 < len` keeps text[i] and
        // text[i+1] in range; the trailing access is guarded by `i < len`.
        let idx = ((text[0] as usize) << 8) | text[1] as usize;
        let mut node = unsafe { *self.r2node.get_unchecked(idx) };
        let mut best = unsafe { *self.r2best.get_unchecked(idx) };
        let mut i = 2;
        while node != 0 && i + 1 < len {
            pf!(NM_STEPS);
            let b0 = unsafe { *text.get_unchecked(i) };
            let b1 = unsafe { *text.get_unchecked(i + 1) };
            let k = ((node as u64) << 16) | ((b0 as u64) << 8) | b1 as u64;
            let m = mix36(k);
            let want = ((m & E2_TAGMASK) << 38) | E2_USED;
            let mut h = (m >> self.e2tb) as u32 & self.e2mask;
            let mut val = 0u64;
            loop {
                let s = unsafe { *self.e2.get_unchecked(h as usize) };
                if s == 0 {
                    break;
                }
                if (s & E2_CMP) == want {
                    val = s;
                    break;
                }
                h = (h + 1) & self.e2mask;
            }
            if val == 0 {
                // no 2-byte step: an odd-depth token may extend one byte
                let o = self.odd_lookup(node, b0);
                if o != RANK_MAX {
                    best = o;
                }
                return best;
            }
            let b18 = val as u32 & E2_BEST_NONE;
            if b18 != E2_BEST_NONE {
                best = b18;
            }
            node = (val >> 18) as u32 & 0xF_FFFF;
            i += 2;
        }
        if node != 0 && i < len {
            let o = self.odd_lookup(node, unsafe { *text.get_unchecked(i) });
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
            // SAFETY: h is always masked with omask and otab.len()==omask+1.
            let s = unsafe { *self.otab.get_unchecked(h as usize) };
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
        pf!(ISVALID);
        let m = mix36(((t1 as u64) << 18) | t2 as u64);
        let h = (m >> IV_TAGBITS) as usize & self.ivmask;
        let want = 0x8000_0000u32 | (((m as u32) & ((1 << IV_TAGBITS) - 1)) << 1);
        // SAFETY: h is masked with ivmask and ivm.len()==ivmask+1.
        let slot = unsafe { self.ivm.get_unchecked(h) };
        let s = slot.load(Relaxed);
        if (s & 0xFFFF_FFFE) == want {
            return (s & 1) != 0;
        }
        pf!(ISVALID_MISS);
        let res = self.ivtp_slow(t1, t2);
        slot.store(want | res as u32, Relaxed);
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
    #[inline]
    pub fn encode(&self, text: &[u8], out: &mut Vec<Rank>) {
        if text.is_empty() {
            return;
        }
        let first = self.next_match(text);
        self.encode_with_first(text, first, out);
    }

    #[inline]
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
                pf!(TOKENS);
                out.push(token);
                last = token;
                // SAFETY: token is a valid id (< n == tlen.len()); pos < len here.
                pos += unsafe { *self.tlen.get_unchecked(token as usize) } as usize;
                nt = if pos < len {
                    self.next_match(unsafe { text.get_unchecked(pos..) })
                } else {
                    RANK_MAX
                };
            }
            if ok {
                return;
            }
            pf!(BACKTRACK);
            out.truncate(out_start);
        }
        // Full backtracking (rare). Reuse thread-local scratch (no per-piece alloc).
        BT.with(|bt| {
            let mut bt = bt.borrow_mut();
            let (toks, bf) = &mut *bt;
            toks.clear();
            let words = (len + 1 + 63) >> 6;
            bf.clear();
            bf.resize(words, u64::MAX);
            let is_set = |bf: &[u64], b: usize| (bf[b >> 6] >> (b & 63)) & 1 != 0;

            let mut pos = 0usize;
            let mut next_token = first;
            while next_token != RANK_MAX {
                let mut token = next_token;
                let last = toks.last().copied().unwrap_or(RANK_MAX);
                loop {
                    let end = pos + self.tlen[token as usize] as usize;
                    if is_set(bf, end)
                        && (last == RANK_MAX || self.is_valid_token_pair(last, token))
                    {
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
            out.extend_from_slice(toks);
        });
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
            e2: Vec::new(),
            e2mask: 0,
            e2tb: 0,
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
            // pack e2 into the tagged single-u64 table (load factor ~0.5, floor
            // 2^17). Verify no circular run of occupied slots reaches the
            // same-tag separation bound 2^(bits-11) — else double and repack, so
            // a probe chain can never reach a same-tag stranger (exactness).
            let entries: Vec<(u64, u32, u32)> =
                e2map.into_iter().map(|(k, (c, b))| (k, c, b)).collect();
            // Lower load factor = bigger e2 = shorter probe chains. On M3 the
            // larger table still fits the (big) caches, so trading footprint for
            // fewer probe loads per next_match step is a net win.
            const E2_LOAD: f64 = 0.11;
            let mut want_cap = 1usize << 17;
            while (entries.len() as f64) / (want_cap as f64) > E2_LOAD {
                want_cap <<= 1;
            }
            loop {
                let bits = want_cap.trailing_zeros();
                let mask = (want_cap - 1) as u32;
                let tb = 36 - bits;
                let mut e2 = vec![0u64; want_cap];
                for &(k, child, best) in &entries {
                    debug_assert!(child < (1 << 20) && (best == RANK_MAX || best < (1 << 18)));
                    let m = mix36(k);
                    let mut h = (m >> tb) as u32 & mask;
                    while e2[h as usize] != 0 {
                        h = (h + 1) & mask;
                    }
                    let best18 = if best == RANK_MAX { E2_BEST_NONE } else { best };
                    e2[h as usize] =
                        E2_USED | ((m & E2_TAGMASK) << 38) | ((child as u64) << 18) | best18 as u64;
                }
                if max_circular_run(&e2) < (1u64 << (bits - 11)) {
                    v.e2 = e2;
                    v.e2mask = mask;
                    v.e2tb = tb;
                    break;
                }
                want_cap <<= 1; // never fires at load 0.5; exactness insurance
            }
            // pack otab.
            // Same big-cache insight as e2: lower load factor = shorter probe
            // chains in odd_lookup (hit at the end of most next_match walks).
            let mut ocap = 1024usize;
            while (otmap.len() as f64) / (ocap as f64) > 0.11 {
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
