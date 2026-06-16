//! Python bindings for sonictok (tiktoken-style API). Built with maturin.
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use api::{Allowed, Tokenizer};
use std::collections::HashSet;

/// A loaded encoding. Mirrors tiktoken's `Encoding` surface.
#[pyclass(frozen)]
struct Encoding {
    inner: Tokenizer,
}

#[pymethods]
impl Encoding {
    /// Encode, ignoring special tokens (they tokenize as ordinary text).
    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.allow_threads(|| self.inner.encode_ordinary(text))
    }

    /// tiktoken-style `encode`: raises on a special token unless allowed.
    /// `allowed_special` is "all" or an iterable of special-token strings.
    #[pyo3(signature = (text, allowed_special=None))]
    fn encode(&self, text: &str, allowed_special: Option<Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
        let owned: Option<HashSet<String>> = match &allowed_special {
            None => None,
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    if s == "all" {
                        None // sentinel handled below as "all"
                    } else {
                        return Err(PyValueError::new_err("allowed_special must be 'all' or an iterable of strings"));
                    }
                } else {
                    Some(obj.extract::<HashSet<String>>()?)
                }
            }
        };
        let result = match (&allowed_special, &owned) {
            (Some(_), None) => self.inner.encode(text, Allowed::All), // "all"
            (None, _) => self.inner.encode(text, Allowed::None),
            (_, Some(set)) => {
                let refs: HashSet<&str> = set.iter().map(String::as_str).collect();
                self.inner.encode(text, Allowed::Set(&refs))
            }
        };
        result.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Encode recognizing all special tokens.
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.inner.encode_with_special(text)
    }

    /// Token count (encode_ordinary semantics).
    fn count(&self, py: Python<'_>, text: &str) -> usize {
        py.allow_threads(|| self.inner.count(text))
    }

    /// Decode token ids to text (lossy UTF-8 for invalid byte sequences).
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(|e| PyKeyError::new_err(e.to_string()))
    }

    /// Parallel batch encode (encode_ordinary semantics): list[list[int]].
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<u32>> {
        py.allow_threads(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let batch = self.inner.encode_batch(&refs);
            let mut out = Vec::with_capacity(texts.len());
            for w in batch.offsets.windows(2) {
                out.push(batch.tokens[w[0] as usize..w[1] as usize].to_vec());
            }
            out
        })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.encoding()
    }
    #[getter]
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn __repr__(&self) -> String {
        format!("<sonictok.Encoding {:?}>", self.inner.encoding())
    }
}

/// Load a bundled encoding: "cl100k_base", "o200k_base", or "o200k_harmony".
#[pyfunction]
fn get_encoding(name: &str) -> PyResult<Encoding> {
    api::get_encoding(name)
        .map(|inner| Encoding { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn sonictok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Encoding>()?;
    m.add_function(wrap_pyfunction!(get_encoding, m)?)?;
    m.add("__doc__", "Fast, exact BPE tokenizer — byte-identical to tiktoken.")?;
    Ok(())
}
