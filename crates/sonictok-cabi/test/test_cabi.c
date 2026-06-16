/* C smoke test for the sonictok C ABI. Built + run by `xtask test-cabi`. */
#include "sonictok.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    const char *data = getenv("SONICTOK_DATA");
    if (!data) {
        fprintf(stderr, "set SONICTOK_DATA to the repo data/ dir\n");
        return 2;
    }
    StTokenizer *tok = sonictok_get_encoding("cl100k_base");
    if (!tok) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    if (sonictok_n_vocab(tok) != 100277) {
        fprintf(stderr, "n_vocab mismatch: %zu\n", sonictok_n_vocab(tok));
        return 1;
    }

    const char *text = "hello world";
    uint32_t *ids = NULL;
    size_t n = 0;
    if (sonictok_encode_ordinary(tok, (const uint8_t *)text, strlen(text), &ids, &n) != ST_OK) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    if (n != 2 || ids[0] != 15339 || ids[1] != 1917) {
        fprintf(stderr, "ids mismatch: n=%zu [%u, %u]\n", n, n > 0 ? ids[0] : 0, n > 1 ? ids[1] : 0);
        return 1;
    }
    if (sonictok_count(tok, (const uint8_t *)text, strlen(text)) != 2) {
        fprintf(stderr, "count mismatch\n");
        return 1;
    }

    uint8_t *bytes = NULL;
    size_t blen = 0;
    if (sonictok_decode(tok, ids, n, &bytes, &blen) != ST_OK) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    if (blen != strlen(text) || memcmp(bytes, text, blen) != 0) {
        fprintf(stderr, "decode round-trip mismatch\n");
        return 1;
    }

    sonictok_free_bytes(bytes, blen);
    sonictok_free_ids(ids, n);
    sonictok_free(tok);
    printf("C ABI ok: encode/decode/count round-trip byte-exact\n");
    return 0;
}
