#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "batch_packing.h"

extern "C" {

BLT_API int32_t blt_pack_patch_batches_c(
    const int32_t* patch_counts,
    size_t num_seqs,
    int32_t max_patches_per_batch,
    int32_t* out_batch_ids,
    int32_t* num_batches_out
) {
    if (!patch_counts || !out_batch_ids || !num_batches_out) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }
    if (num_seqs == 0) {
        *num_batches_out = 0;
        return BLT_SUCCESS;
    }
    if (max_patches_per_batch <= 0) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }

    int32_t current_batch_id = 0;
    int32_t current_batch_patches = 0;

    for (size_t i = 0; i < num_seqs; ++i) {
        int32_t p = patch_counts[i];
        if (p <= 0) p = 1; // at least 1 patch per sequence

        // If adding p exceeds max_patches and current batch has at least one sequence
        if (current_batch_patches + p > max_patches_per_batch && current_batch_patches > 0) {
            current_batch_id++;
            current_batch_patches = 0;
        }

        out_batch_ids[i] = current_batch_id;
        current_batch_patches += p;
    }

    *num_batches_out = current_batch_id + 1;
    return BLT_SUCCESS;
}

} // extern "C"
