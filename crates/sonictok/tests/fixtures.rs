//! Hermetic exactness: diff sonictok against frozen tiktoken vectors.
use sonictok::get_encoding;

#[derive(serde::Deserialize)]
struct Record {
    mode: String,
    ids: Vec<u32>,
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    input_b64: Option<String>,
}

fn decode_input(r: &Record) -> Vec<u8> {
    if let Some(s) = &r.input {
        s.clone().into_bytes()
    } else {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(r.input_b64.as_ref().unwrap())
            .unwrap()
    }
}

#[test]
fn cl100k_fixtures_match() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/cl100k_base.json"
    );
    let json = std::fs::read_to_string(path).expect("run tools/gen_fixtures.py");
    let records: Vec<Record> = serde_json::from_str(&json).unwrap();
    let t = get_encoding("cl100k_base").expect("data/cl100k_base.stb");

    let mut failures = 0;
    for (i, r) in records.iter().enumerate() {
        let bytes = decode_input(r);
        let text = std::str::from_utf8(&bytes).expect("fixtures are valid utf8 in Plan 1");
        let got = match r.mode.as_str() {
            "ordinary" => t.encode_ordinary(text),
            "with_special" => t.encode_with_special(text),
            other => panic!("unknown mode {other}"),
        };
        if got != r.ids {
            failures += 1;
            eprintln!(
                "record {i} ({:?}) mismatch:\n  expected {:?}\n  got      {:?}",
                r.mode, r.ids, got
            );
        }
    }
    assert_eq!(failures, 0, "{failures} fixture mismatches");
}
