//! Stable C ABI for sonictok — FFI from any language.
//!
//! Ownership: `sonictok_get_encoding` returns an opaque handle freed with
//! `sonictok_free`. Encode/decode allocate output buffers the caller frees with
//! `sonictok_free_ids` / `sonictok_free_bytes`. Handles are `Send + Sync`, so a
//! single handle may be used from many threads concurrently.
#![allow(clippy::missing_safety_doc)]

use api::Tokenizer;
use std::ffi::{CStr, c_char};
use std::os::raw::c_int;

pub const ST_OK: c_int = 0;
pub const ST_ERR_NULL: c_int = 1;
pub const ST_ERR_ENCODING: c_int = 2;
pub const ST_ERR_UTF8: c_int = 3;
pub const ST_ERR_DECODE: c_int = 4;

/// Opaque tokenizer handle.
pub struct StTokenizer(Tokenizer);

/// Load a bundled encoding (e.g. "cl100k_base"). Returns NULL on failure.
/// # Safety: `name` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_get_encoding(name: *const c_char) -> *mut StTokenizer {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
        return std::ptr::null_mut();
    };
    match api::get_encoding(name) {
        Ok(t) => Box::into_raw(Box::new(StTokenizer(t))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a tokenizer handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_free(tok: *mut StTokenizer) {
    if !tok.is_null() {
        drop(unsafe { Box::from_raw(tok) });
    }
}

#[inline]
unsafe fn as_str<'a>(text: *const u8, len: usize) -> Option<&'a str> {
    if text.is_null() {
        return None;
    }
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(text, len) }).ok()
}

/// Encode (encode_ordinary semantics). On ST_OK, `*out_ids` points to a buffer of
/// `*out_len` u32 token ids that must be freed with `sonictok_free_ids`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_encode_ordinary(
    tok: *const StTokenizer,
    text: *const u8,
    text_len: usize,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
) -> c_int {
    if tok.is_null() || out_ids.is_null() || out_len.is_null() {
        return ST_ERR_NULL;
    }
    let tok = unsafe { &*tok };
    let Some(s) = (unsafe { as_str(text, text_len) }) else {
        return ST_ERR_UTF8;
    };
    let ids = tok.0.encode_ordinary(s).into_boxed_slice();
    let len = ids.len();
    let ptr = Box::into_raw(ids) as *mut u32;
    unsafe {
        *out_ids = ptr;
        *out_len = len;
    }
    ST_OK
}

/// Free an id buffer returned by `sonictok_encode_ordinary`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_free_ids(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ids, len)) });
    }
}

/// Token count (encode_ordinary semantics). Returns -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_count(
    tok: *const StTokenizer,
    text: *const u8,
    text_len: usize,
) -> isize {
    if tok.is_null() {
        return -1;
    }
    let tok = unsafe { &*tok };
    match unsafe { as_str(text, text_len) } {
        Some(s) => tok.0.count(s) as isize,
        None => -1,
    }
}

/// Decode ids to UTF-8 bytes. On ST_OK, `*out_bytes` points to a buffer of
/// `*out_len` bytes that must be freed with `sonictok_free_bytes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_decode(
    tok: *const StTokenizer,
    ids: *const u32,
    n: usize,
    out_bytes: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    if tok.is_null() || (ids.is_null() && n != 0) || out_bytes.is_null() || out_len.is_null() {
        return ST_ERR_NULL;
    }
    let tok = unsafe { &*tok };
    let ids = unsafe { std::slice::from_raw_parts(ids, n) };
    match tok.0.decode_bytes(ids) {
        Ok(bytes) => {
            let b = bytes.into_boxed_slice();
            let len = b.len();
            let ptr = Box::into_raw(b) as *mut u8;
            unsafe {
                *out_bytes = ptr;
                *out_len = len;
            }
            ST_OK
        }
        Err(_) => ST_ERR_DECODE,
    }
}

/// Free a byte buffer returned by `sonictok_decode`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_free_bytes(bytes: *mut u8, len: usize) {
    if !bytes.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(bytes, len)) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_n_vocab(tok: *const StTokenizer) -> usize {
    if tok.is_null() {
        return 0;
    }
    (unsafe { &*tok }).0.n_vocab()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sonictok_vocab_size(tok: *const StTokenizer) -> usize {
    if tok.is_null() {
        return 0;
    }
    (unsafe { &*tok }).0.vocab_size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    #[ignore = "requires data/cl100k_base.stb"]
    fn cabi_round_trip() {
        unsafe {
            let name = CString::new("cl100k_base").unwrap();
            let tok = sonictok_get_encoding(name.as_ptr());
            assert!(!tok.is_null());
            assert_eq!(sonictok_n_vocab(tok), 100277);

            let text = b"hello world";
            let mut ids: *mut u32 = std::ptr::null_mut();
            let mut len: usize = 0;
            assert_eq!(
                sonictok_encode_ordinary(tok, text.as_ptr(), text.len(), &mut ids, &mut len),
                ST_OK
            );
            assert_eq!(std::slice::from_raw_parts(ids, len), &[15339, 1917]);
            assert_eq!(sonictok_count(tok, text.as_ptr(), text.len()), 2);

            let mut bytes: *mut u8 = std::ptr::null_mut();
            let mut blen: usize = 0;
            assert_eq!(sonictok_decode(tok, ids, len, &mut bytes, &mut blen), ST_OK);
            assert_eq!(std::slice::from_raw_parts(bytes, blen), text);

            sonictok_free_bytes(bytes, blen);
            sonictok_free_ids(ids, len);
            sonictok_free(tok);
        }
    }
}
