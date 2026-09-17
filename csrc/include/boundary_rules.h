#ifndef BLT_BOUNDARY_RULES_H
#define BLT_BOUNDARY_RULES_H

#include "blt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute patch boundary mask using the monotonic entropy rule with context resets.
 *
 * A position i is marked as a boundary (1) if:
 * 1. i == 0 (start of sequence)
 * 2. bytes[i] == doc_boundary_token (start of new document)
 * 3. H(x_i) - H(x_{i-1}) > theta_r (entropy jump above threshold)
 * 4. Current patch length >= max_patch_size (hard ceiling to avoid runaway patches)
 *
 * Entropy delta state is reset whenever a document boundary or newline is encountered.
 *
 * @param entropy Float array of per-byte entropies of length seq_len
 * @param bytes uint8 array of byte tokens of length seq_len
 * @param seq_len Length of sequence
 * @param theta_r Monotonic entropy threshold
 * @param reset_on_newline If 1, resets previous entropy on '\n' (byte 10)
 * @param doc_boundary_token Value representing document boundary (default 256, or -1 if none)
 * @param max_patch_size Maximum allowed patch size (e.g. 32, or 0 if unlimited)
 * @param out_boundaries Output uint8 array of length seq_len (1 = patch start, 0 = continuation)
 * @param num_patches_out Optional output pointer to receive total number of patches formed
 * @return BLT_SUCCESS or error code
 */
BLT_API int32_t blt_monotonic_boundary_mask(
    const float* entropy,
    const uint8_t* bytes,
    size_t seq_len,
    float theta_r,
    int32_t reset_on_newline,
    int32_t doc_boundary_token,
    int32_t max_patch_size,
    uint8_t* out_boundaries,
    int64_t* num_patches_out
);

/**
 * Global threshold boundary rule: boundary if entropy[i] > theta_g.
 */
BLT_API int32_t blt_global_boundary_mask(
    const float* entropy,
    size_t seq_len,
    float theta_g,
    uint8_t* out_boundaries,
    int64_t* num_patches_out
);

#ifdef __cplusplus
}
#endif

#endif // BLT_BOUNDARY_RULES_H
