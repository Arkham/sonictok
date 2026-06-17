//! Lightweight, fast, deterministic encode benchmark for the autoresearch loop.
//! Mirrors `cargo bench --bench encode` (cl100k/o200k encode_ordinary + batch)
//! but reports a stable median over many reps with far less wall time than
//! criterion's warmup+sampling machinery. Emits `METRIC name=value` lines.
use sonictok::get_encoding;
use std::time::Instant;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// Per-metric result: min, median, sample CV% (stdev/mean — a contention
/// detector: tight when the run got a clean P-core, blows up under scheduling
/// interference), and a checksum to defeat dead-code elimination.
struct Stat {
    min: f64,
    median: f64,
    cv_pct: f64,
    checksum: usize,
}

fn bench_single<F: FnMut() -> usize>(reps: usize, warmup: usize, mut f: F) -> Stat {
    let mut checksum = 0usize;
    for _ in 0..warmup {
        checksum = checksum.wrapping_add(f());
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let c = f();
        let dt = t0.elapsed().as_secs_f64() * 1e6; // µs
        checksum = checksum.wrapping_add(c);
        samples.push(dt);
    }
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let cv_pct = var.sqrt() / mean * 100.0;
    Stat {
        min,
        median: median(&mut samples),
        cv_pct,
        checksum,
    }
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::env::var("CORPUS_PATH")
        .unwrap_or_else(|_| format!("{manifest}/../../bench/corpus.txt"));
    let text = std::fs::read_to_string(&path).expect("corpus file");
    let nbytes = text.len();

    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // cl100k_base encode_ordinary (headline metric)
    let cl = get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let mut out = Vec::with_capacity(nbytes / 3 + 1);
    let cls = bench_single(reps, warmup, || {
        out.clear();
        cl.encode_ordinary_into(std::hint::black_box(&text), &mut out);
        std::hint::black_box(out.len())
    });
    let cl_tokens = {
        out.clear();
        cl.encode_ordinary_into(&text, &mut out);
        out.len()
    };

    // Secondary monitors (o200k + batch) only with BENCH_FULL=1 — the loop
    // measures cl100k alone for speed; run full periodically to catch regressions.
    let full = std::env::var("BENCH_FULL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let (o2s, bas) = if full {
        let o2 = get_encoding("o200k_base").expect("data/o200k_base.stb");
        let mut out2 = Vec::with_capacity(nbytes / 3 + 1);
        let o2s = bench_single(reps, warmup, || {
            out2.clear();
            o2.encode_ordinary_into(std::hint::black_box(&text), &mut out2);
            std::hint::black_box(out2.len())
        });
        let docs: Vec<&str> = text.split("\n\n").filter(|s| !s.is_empty()).collect();
        let bas = bench_single(reps.min(40), warmup.min(5), || {
            let b = cl.encode_batch(std::hint::black_box(&docs));
            std::hint::black_box(b.tokens.len())
        });
        (Some(o2s), Some(bas))
    } else {
        (None, None)
    };

    let mb = nbytes as f64 / 1_000_000.0;
    let mib = nbytes as f64 / (1024.0 * 1024.0);

    // Primary metric: cl100k single-thread microseconds (MEDIAN, lower is
    // better). median is the central estimator; *_cv_pct is the stability gate
    // (the loop rejects + retries runs whose cl100k_cv_pct is too high).
    println!("METRIC cl100k_us={:.1}", cls.median);
    println!("METRIC cl100k_min_us={:.1}", cls.min);
    println!("METRIC cl100k_cv_pct={:.2}", cls.cv_pct);
    println!("METRIC cl100k_mibs={:.2}", mib / (cls.median / 1e6));
    println!("METRIC cl100k_mbs={:.2}", mb / (cls.median / 1e6));
    println!("METRIC cl100k_min_mbs={:.2}", mb / (cls.min / 1e6));
    if let Some(o2s) = &o2s {
        println!("METRIC o200k_us={:.1}", o2s.median);
        println!("METRIC o200k_cv_pct={:.2}", o2s.cv_pct);
        println!("METRIC o200k_mibs={:.2}", mib / (o2s.median / 1e6));
    }
    if let Some(bas) = &bas {
        println!("METRIC batch_us={:.1}", bas.median);
        println!("METRIC batch_mibs={:.2}", mib / (bas.median / 1e6));
    }
    eprintln!(
        "cl100k: median {:.1}µs (min {:.1}, cv {:.2}%, {:.2} MB/s median / {:.2} min), tokens={cl_tokens}, sum={}",
        cls.median,
        cls.min,
        cls.cv_pct,
        mb / (cls.median / 1e6),
        mb / (cls.min / 1e6),
        cls.checksum
    );
    if let Some(o2s) = &o2s {
        eprintln!(
            "o200k:  median {:.1}µs (cv {:.2}%, {:.2} MB/s)",
            o2s.median,
            o2s.cv_pct,
            mb / (o2s.median / 1e6)
        );
    }
    if let Some(bas) = &bas {
        eprintln!(
            "batch:  median {:.1}µs ({:.2} MiB/s)",
            bas.median,
            mib / (bas.median / 1e6)
        );
    }
}
