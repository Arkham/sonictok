//! Project automation. Subcommands:
//!   build-data [enc]   pack data/<enc>.tiktoken + .special into data/<enc>.stb
//!   bench-compare      build+run local quicktok, then sonictok criterion bench
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build-data") => build_data(args.get(2).map(String::as_str).unwrap_or("cl100k_base")),
        Some("bench-compare") => bench_compare(),
        other => {
            eprintln!("usage: xtask <build-data [encoding]|bench-compare>; got {other:?}");
            exit(2);
        }
    }
}

fn build_data(enc: &str) {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD;

    let ranks_path = format!("data/{enc}.tiktoken");
    let special_path = format!("data/{enc}.special");
    let ranks_txt = std::fs::read_to_string(&ranks_path)
        .unwrap_or_else(|e| panic!("read {ranks_path}: {e} (run tools/export_{enc}.py first)"));
    let special_txt = std::fs::read_to_string(&special_path).unwrap_or_default();

    let mut ranks = Vec::new();
    let mut max_id = 0u32;
    for line in ranks_txt.lines() {
        let (b64tok, rank) = line.split_once(' ').expect("malformed rank line");
        let bytes = b64.decode(b64tok).expect("bad base64");
        let id: u32 = rank.parse().expect("bad rank");
        max_id = max_id.max(id);
        ranks.push((bytes, id));
    }
    let mut specials = Vec::new();
    for line in special_txt.lines() {
        let (name, id) = line.rsplit_once(' ').expect("malformed special line");
        let id: u32 = id.parse().expect("bad special id");
        max_id = max_id.max(id);
        specials.push((name.as_bytes().to_vec(), id));
    }

    let blob = sonictok_data::VocabBlob { name: enc.to_string(), max_id, ranks, specials };
    let bytes = blob.to_bytes();
    let out = format!("data/{enc}.stb");
    std::fs::write(&out, &bytes).expect("write blob");
    println!("wrote {out} ({} bytes, max_id {max_id})", bytes.len());
}

fn bench_compare() {
    use std::process::Command;
    let qdir = "bench/quicktok-ref";
    if !std::path::Path::new(qdir).exists() {
        eprintln!("cloning quicktok into {qdir} ...");
        run(Command::new("git").args([
            "clone",
            "--depth",
            "1",
            "https://github.com/dmatth1/quicktok",
            qdir,
        ]));
    }
    eprintln!("building + running quicktok native bench (its corpus == ours: Moby-Dick) ...");
    run(Command::new("make").arg("bench").current_dir(qdir));
    eprintln!("\n--- sonictok criterion bench (same bench/corpus.txt) ---");
    run(Command::new("cargo").args(["bench", "-p", "sonictok", "--bench", "encode"]));
    eprintln!("\nCompare sonictok MB/s above against quicktok's single-thread MB/s.");
    eprintln!("Target: beat quicktok native (see bench/BASELINE.md).");
}

fn run(cmd: &mut std::process::Command) {
    let status = cmd.status().expect("spawn");
    assert!(status.success(), "command failed: {cmd:?}");
}
