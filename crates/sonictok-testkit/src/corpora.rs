use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
pub fn corpus_text() -> String {
    std::fs::read_to_string(repo_root().join("bench/corpus.txt")).expect("bench/corpus.txt")
}
pub fn cl100k_blob() -> sonictok_data::VocabBlob {
    let bytes = std::fs::read(repo_root().join("data/cl100k_base.stb"))
        .expect("data/cl100k_base.stb (run: cargo run -p xtask -- build-data cl100k_base)");
    sonictok_data::VocabBlob::from_bytes(&bytes).unwrap()
}
