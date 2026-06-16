use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sonictok::get_encoding;

fn bench_encode(c: &mut Criterion) {
    let t = get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/corpus.txt");
    let text = std::fs::read_to_string(path).expect("bench/corpus.txt");

    let mut g = c.benchmark_group("cl100k_base");
    g.throughput(Throughput::Bytes(text.len() as u64));
    g.bench_function("encode_ordinary", |b| {
        b.iter(|| std::hint::black_box(t.encode_ordinary(std::hint::black_box(&text))))
    });
    // Pretokenizer-only (no BPE) to isolate where time goes.
    g.bench_function("pretokenize_only", |b| {
        use sonictok_core::pretok::{Grammar, Pretokenizer, Scanner};
        b.iter(|| {
            let bytes = std::hint::black_box(text.as_bytes());
            let mut s = Scanner::new(Grammar::Cl100k);
            let mut n = 0usize;
            while let Some((a, z)) = s.next_piece(bytes) {
                n += z - a;
            }
            std::hint::black_box(n)
        })
    });
    // Parallel batch (rayon) — split into paragraphs like quicktok's bench.
    let docs: Vec<&str> = text.split("\n\n").filter(|s| !s.is_empty()).collect();
    g.bench_function("encode_batch", |b| {
        b.iter(|| std::hint::black_box(t.encode_batch(std::hint::black_box(&docs))))
    });
    g.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
