#[test]
fn vendored_blobs_present() {
    for enc in ["cl100k_base", "o200k_base", "o200k_harmony"] {
        let p = format!("{}/../../data/{enc}.stb", env!("CARGO_MANIFEST_DIR"));
        assert!(
            std::path::Path::new(&p).exists(),
            "vendored data/{enc}.stb missing"
        );
    }
}
