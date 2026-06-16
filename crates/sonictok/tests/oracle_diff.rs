use sonictok::get_encoding;
use sonictok_testkit::{corpora, oracle::Oracle};

fn diff(encoding: &str, oracle: &Oracle) {
    let t = get_encoding(encoding).unwrap_or_else(|_| panic!("data/{encoding}.stb"));
    let text = corpora::corpus_text();

    // chunk by lines to localize any divergence
    for (n, line) in text.lines().enumerate() {
        let got = t.encode_ordinary(line);
        let want = oracle.encode_ordinary(line);
        if got != want {
            panic!(
                "[{encoding}] line {n} diverges from oracle:\n  text={line:?}\n  oracle={want:?}\n  sonictok={got:?}"
            );
        }
    }

    // whole-document check too (catches cross-line boundary issues)
    assert_eq!(t.encode_ordinary(&text), oracle.encode_ordinary(&text));
}

#[test]
fn cl100k_matches_oracle() {
    diff("cl100k_base", &Oracle::cl100k());
}

#[test]
fn o200k_matches_oracle() {
    diff("o200k_base", &Oracle::o200k_base());
}
