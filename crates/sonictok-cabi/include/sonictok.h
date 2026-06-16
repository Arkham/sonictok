/* sonictok — stable C ABI. Byte-exact BPE tokenizer.
 *
 * A handle from sonictok_get_encoding() is freed with sonictok_free(). It is
 * thread-safe: one handle may be used from many threads concurrently. Output
 * buffers from encode/decode are owned by the caller and freed with the
 * matching sonictok_free_ids / sonictok_free_bytes.
 */
#ifndef SONICTOK_H
#define SONICTOK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StTokenizer StTokenizer;

/* status codes */
#define ST_OK 0
#define ST_ERR_NULL 1
#define ST_ERR_ENCODING 2
#define ST_ERR_UTF8 3
#define ST_ERR_DECODE 4

/* Load a bundled encoding ("cl100k_base", "o200k_base", "o200k_harmony").
 * Returns NULL on failure. Free with sonictok_free. */
StTokenizer *sonictok_get_encoding(const char *name);
void sonictok_free(StTokenizer *tok);

/* encode_ordinary: on ST_OK, *out_ids is a buffer of *out_len u32 ids,
 * freed with sonictok_free_ids. text must be valid UTF-8. */
int sonictok_encode_ordinary(const StTokenizer *tok, const uint8_t *text,
                             size_t text_len, uint32_t **out_ids, size_t *out_len);
void sonictok_free_ids(uint32_t *ids, size_t len);

/* token count (encode_ordinary semantics); -1 on error. */
ptrdiff_t sonictok_count(const StTokenizer *tok, const uint8_t *text, size_t text_len);

/* decode ids -> UTF-8 bytes; on ST_OK, *out_bytes has *out_len bytes,
 * freed with sonictok_free_bytes. */
int sonictok_decode(const StTokenizer *tok, const uint32_t *ids, size_t n,
                    uint8_t **out_bytes, size_t *out_len);
void sonictok_free_bytes(uint8_t *bytes, size_t len);

size_t sonictok_n_vocab(const StTokenizer *tok);
size_t sonictok_vocab_size(const StTokenizer *tok);

#ifdef __cplusplus
}
#endif
#endif /* SONICTOK_H */
