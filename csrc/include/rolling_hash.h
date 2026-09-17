#ifndef BLT_ROLLING_HASH_H
#define BLT_ROLLING_HASH_H

#include "blt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLT_DEFAULT_HASH_PRIME 31337ULL

/**
 * Compute RollPolyHash for a single byte n-gram window ending at current position.
 *
 * @param bytes Pointer to byte buffer
 * @param window_len Number of bytes in the n-gram window (e.g. 3, 4, 5)
 * @param prime Polynomial base prime (e.g. 31337)
 * @param vocab_size Modulo bucket count for the embedding table
 * @return 64-bit integer hash bucket index in [0, vocab_size - 1]
 */
BLT_API int64_t blt_roll_hash_single(
    const uint8_t* bytes,
    size_t window_len,
    uint64_t prime,
    int64_t vocab_size
);

/**
 * Compute RollPolyHash across an entire sequence for multiple n-gram sizes.
 *
 * Output layout: out_hashes is a contiguous array of shape (num_sizes, seq_len)
 * stored in row-major order, so out_hashes[size_idx * seq_len + byte_idx] contains
 * the bucket index for ngram_sizes[size_idx] ending at byte_idx.
 *
 * @param bytes Input byte sequence of length seq_len
 * @param seq_len Sequence length
 * @param ngram_sizes Array of n-gram lengths (e.g. [3, 4, 5])
 * @param vocab_sizes Array of vocabulary table sizes corresponding to each n-gram size
 * @param num_sizes Number of n-gram sizes
 * @param prime Polynomial base prime
 * @param out_hashes Pre-allocated output buffer of size (num_sizes * seq_len)
 * @return BLT_SUCCESS or error code
 */
BLT_API int32_t blt_compute_ngram_hashes(
    const uint8_t* bytes,
    size_t seq_len,
    const int32_t* ngram_sizes,
    const int64_t* vocab_sizes,
    size_t num_sizes,
    uint64_t prime,
    int64_t* out_hashes
);

#ifdef __cplusplus
}
#endif

#endif // BLT_ROLLING_HASH_H
