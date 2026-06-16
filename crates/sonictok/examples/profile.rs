//! Deterministic profiling: pretok-vs-BPE time split + exact BPE op counts.
//! Run: cargo run --release --features profile --example profile
use std::time::Instant;

fn main() {
    let t = sonictok::get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/corpus.txt");
    let text = std::fs::read_to_string(path).expect("bench/corpus.txt");
    let mb = text.len() as f64 / 1e6;
    let reps = 50;

    // warm
    let _ = t.encode_ordinary(&text);

    // full encode
    let t0 = Instant::now();
    let mut ntok = 0;
    for _ in 0..reps {
        ntok = t.encode_ordinary(&text).len();
    }
    let full = t0.elapsed().as_secs_f64() / reps as f64;

    // pretokenize-only
    use sonictok_core::pretok::{Grammar, Pretokenizer, Scanner};
    let t0 = Instant::now();
    let mut npieces = 0usize;
    for _ in 0..reps {
        let mut s = Scanner::new(Grammar::Cl100k);
        npieces = 0;
        while let Some((_a, _z)) = s.next_piece(text.as_bytes()) {
            npieces += 1;
        }
    }
    let pre = t0.elapsed().as_secs_f64() / reps as f64;

    println!("corpus {mb:.2} MB, {ntok} tokens, {npieces} pieces\n");
    println!(
        "full encode : {:.3} ms  ({:.1} MB/s)",
        full * 1e3,
        mb / full
    );
    println!(
        "pretok only : {:.3} ms  ({:.1} MB/s, {:.0}%)",
        pre * 1e3,
        mb / pre,
        pre / full * 100.0
    );
    println!(
        "bpe (diff)  : {:.3} ms  ({:.0}%)\n",
        (full - pre) * 1e3,
        (full - pre) / full * 100.0
    );

    // exact BPE op counts (one pass)
    sonictok_core::bpe::prof::PIECES.store(0, std::sync::atomic::Ordering::Relaxed);
    // reset all
    for c in [
        &sonictok_core::bpe::prof::PIECES,
        &sonictok_core::bpe::prof::SINGLE_BYTE,
        &sonictok_core::bpe::prof::WHOLE_HIT,
        &sonictok_core::bpe::prof::MERGE_PIECES,
        &sonictok_core::bpe::prof::MERGE_ITERS,
        &sonictok_core::bpe::prof::LOOKUPS,
        &sonictok_core::bpe::prof::OUT_TOKENS,
    ] {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let _ = t.encode_ordinary(&text);
    println!("BPE op counts (one corpus pass):");
    for (name, v) in sonictok_core::bpe::prof::snapshot() {
        println!("  {name:14} {v}");
    }
    let s = sonictok_core::bpe::prof::snapshot();
    let pieces = s[0].1 as f64;
    let whole = s[2].1 as f64;
    let merge = s[3].1 as f64;
    println!(
        "\n  single-token pieces: {:.1}%   multi-token pieces: {:.1}%",
        whole / pieces * 100.0,
        merge / pieces * 100.0
    );
    println!("  lookups per token: {:.2}", s[5].1 as f64 / s[6].1 as f64);
}
