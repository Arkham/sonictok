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
    g.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
