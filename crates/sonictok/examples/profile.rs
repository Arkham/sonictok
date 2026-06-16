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
    sonictok_core::vocab::prof::reset();
    let _ = t.encode_ordinary(&text);
    println!("new-BPE op counts (one corpus pass):");
    for (name, v) in sonictok_core::vocab::prof::snapshot() {
        println!("  {name:14} {v}");
    }
}
