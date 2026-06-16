//! encode_batch must equal per-document encode_ordinary, byte-for-byte.
use sonictok::get_encoding;
use sonictok_testkit::corpora;

#[test]
fn batch_equals_per_doc() {
    let t = get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let text = corpora::corpus_text();
    let docs: Vec<&str> = text.split("\n\n").filter(|s| !s.is_empty()).collect();

    let batch = t.encode_batch(&docs);
    assert_eq!(batch.offsets.len(), docs.len() + 1);
    assert_eq!(batch.offsets[0], 0);
    assert_eq!(*batch.offsets.last().unwrap() as usize, batch.tokens.len());

    let counts = t.count_batch(&docs);
    for (i, doc) in docs.iter().enumerate() {
        let want = t.encode_ordinary(doc);
        let lo = batch.offsets[i] as usize;
        let hi = batch.offsets[i + 1] as usize;
        assert_eq!(&batch.tokens[lo..hi], &want[..], "doc {i} batch mismatch");
        assert_eq!(counts[i], want.len(), "doc {i} count mismatch");
    }
}

#[test]
fn batch_empty_and_singletons() {
    let t = get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let b = t.encode_batch(&[]);
    assert_eq!(b.offsets, vec![0]);
    assert!(b.tokens.is_empty());

    let b = t.encode_batch(&["", "a", ""]);
    assert_eq!(b.offsets.len(), 4);
    assert_eq!(b.tokens, t.encode_ordinary("a"));
}
