#[test]
fn vendored_blob_present() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cl100k_base.stb");
    assert!(
        std::path::Path::new(p).exists(),
        "vendored data/cl100k_base.stb missing"
    );
}
