#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "rolling_hash.h"
#include <vector>
#include <cstring>

extern "C" {

BLT_API int64_t blt_roll_hash_single(
    const uint8_t* bytes,
    size_t window_len,
    uint64_t prime,
    int64_t vocab_size
) {
    if (!bytes || window_len == 0 || vocab_size <= 0) {
        return 0;
    }

    uint64_t hash_val = 0;
    uint64_t p_pow = 1;

    // Formula: sum_{j=1}^n b_{i-j+1} * a^{j-1} mod |E|
    // bytes[window_len - 1] is b_i (j=1, a^0)
    // bytes[0] is b_{i-n+1} (j=n, a^{n-1})
    for (size_t j = 0; j < window_len; ++j) {
        size_t idx = window_len - 1 - j;
        uint64_t byte_val = static_cast<uint64_t>(bytes[idx]);
        hash_val = (hash_val + (byte_val * p_pow) % static_cast<uint64_t>(vocab_size)) % static_cast<uint64_t>(vocab_size);
        p_pow = (p_pow * prime) % static_cast<uint64_t>(vocab_size);
    }

    return static_cast<int64_t>(hash_val);
}

BLT_API int32_t blt_compute_ngram_hashes(
    const uint8_t* bytes,
    size_t seq_len,
    const int32_t* ngram_sizes,
    const int64_t* vocab_sizes,
    size_t num_sizes,
    uint64_t prime,
    int64_t* out_hashes
) {
    if (!bytes || !ngram_sizes || !vocab_sizes || !out_hashes) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_len == 0 || num_sizes == 0) {
        return BLT_SUCCESS;
    }

    if (prime == 0) {
        prime = BLT_DEFAULT_HASH_PRIME;
    }

    for (size_t s = 0; s < num_sizes; ++s) {
        int32_t n = ngram_sizes[s];
        int64_t v = vocab_sizes[s];
        if (n <= 0 || v <= 0) {
            return BLT_ERROR_INVALID_ARGUMENT;
        }

        int64_t* out_row = out_hashes + (s * seq_len);
        uint64_t u_vocab = static_cast<uint64_t>(v);

        // Precompute powers of prime: p_pow[k] = prime^k mod v
        std::vector<uint64_t> p_pow(n, 1);
        for (int32_t k = 1; k < n; ++k) {
            p_pow[k] = (p_pow[k - 1] * prime) % u_vocab;
        }

        // For each position i in the byte sequence
        for (size_t i = 0; i < seq_len; ++i) {
            size_t available_len = (i + 1 < static_cast<size_t>(n)) ? (i + 1) : static_cast<size_t>(n);
            uint64_t h = 0;

            for (size_t j = 0; j < available_len; ++j) {
                size_t byte_idx = i - j;
                uint64_t b = static_cast<uint64_t>(bytes[byte_idx]);
                h = (h + (b * p_pow[j]) % u_vocab) % u_vocab;
            }

            out_row[i] = static_cast<int64_t>(h);
        }
    }

    return BLT_SUCCESS;
}

} // extern "C"
