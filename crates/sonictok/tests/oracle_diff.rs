use sonictok::get_encoding;
use sonictok_testkit::{corpora, oracle::Oracle};

#[test]
fn corpus_matches_oracle() {
    let t = get_encoding("cl100k_base").expect("data/cl100k_base.stb");
    let oracle = Oracle::cl100k();
    let text = corpora::corpus_text();

    // chunk by lines to localize any divergence
    let mut mismatched_line = None;
    for (n, line) in text.lines().enumerate() {
        let got = t.encode_ordinary(line);
        let want = oracle.encode_ordinary(line);
        if got != want {
            mismatched_line = Some((n, line.to_string(), want, got));
            break;
        }
    }
    if let Some((n, line, want, got)) = mismatched_line {
        panic!("line {n} diverges from oracle:\n  text={line:?}\n  oracle={want:?}\n  sonictok={got:?}");
    }

    // whole-document check too (catches cross-line boundary issues)
    assert_eq!(t.encode_ordinary(&text), oracle.encode_ordinary(&text));
}
