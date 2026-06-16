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
pub fn blob(encoding: &str) -> sonictok_data::VocabBlob {
    let path = repo_root().join(format!("data/{encoding}.stb"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("{path:?} (run: cargo run -p xtask -- build-data {encoding})"));
    sonictok_data::VocabBlob::from_bytes(&bytes).unwrap()
}
