#ifndef BLT_BATCH_PACKING_H
#define BLT_BATCH_PACKING_H

#include "blt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Greedy patch-aware batch packing for variable-length sequences.
 *
 * Assigns each sequence an integer batch index in [0, num_batches_out - 1] such that
 * the sum of patches in any single batch does not exceed max_patches_per_batch.
 *
 * @param patch_counts Array of patch counts for each sequence (length num_seqs)
 * @param num_seqs Total number of sequences to pack
 * @param max_patches_per_batch Maximum total patches permitted per batch (e.g. 512)
 * @param out_batch_ids Pre-allocated int32 array of length num_seqs receiving batch assignments
 * @param num_batches_out Output pointer receiving the total number of batches created
 * @return BLT_SUCCESS or error code
 */
BLT_API int32_t blt_pack_patch_batches_c(
    const int32_t* patch_counts,
    size_t num_seqs,
    int32_t max_patches_per_batch,
    int32_t* out_batch_ids,
    int32_t* num_batches_out
);

#ifdef __cplusplus
}
#endif

#endif // BLT_BATCH_PACKING_H
