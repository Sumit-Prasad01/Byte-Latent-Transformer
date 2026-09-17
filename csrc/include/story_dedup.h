#ifndef BLT_STORY_DEDUP_H
#define BLT_STORY_DEDUP_H

#include "blt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute 64-bit FNV-1a hash over a byte buffer.
 */
BLT_API uint64_t blt_hash_bytes(const uint8_t* data, size_t length);

/**
 * Deduplicate an array of stories given their offsets in a corpus buffer.
 *
 * @param corpus Pointer to contiguous bytes containing all stories
 * @param story_offsets Array of start byte offsets for each story of length num_stories
 * @param story_lengths Array of byte lengths for each story of length num_stories
 * @param num_stories Number of stories
 * @param out_keep_mask Pre-allocated int8 array of size num_stories (1 = keep/unique, 0 = duplicate)
 * @param out_unique_count Optional output pointer to receive count of unique stories
 * @return BLT_SUCCESS or error code
 */
BLT_API int32_t blt_dedup_story_offsets(
    const uint8_t* corpus,
    const int64_t* story_offsets,
    const int64_t* story_lengths,
    size_t num_stories,
    int8_t* out_keep_mask,
    int64_t* out_unique_count
);

#ifdef __cplusplus
}
#endif

#endif // BLT_STORY_DEDUP_H
