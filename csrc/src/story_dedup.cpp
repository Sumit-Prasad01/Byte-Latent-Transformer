#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "story_dedup.h"
#include <unordered_set>

extern "C" {

BLT_API uint64_t blt_hash_bytes(const uint8_t* data, size_t length) {
    if (!data || length == 0) return 0;

    // 64-bit FNV-1a hash
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;

    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < length; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

BLT_API int32_t blt_dedup_story_offsets(
    const uint8_t* corpus,
    const int64_t* story_offsets,
    const int64_t* story_lengths,
    size_t num_stories,
    int8_t* out_keep_mask,
    int64_t* out_unique_count
) {
    if (!corpus || !story_offsets || !story_lengths || !out_keep_mask) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }
    if (num_stories == 0) {
        if (out_unique_count) *out_unique_count = 0;
        return BLT_SUCCESS;
    }

    std::unordered_set<uint64_t> seen_hashes;
    seen_hashes.reserve(num_stories);

    int64_t unique_count = 0;

    for (size_t i = 0; i < num_stories; ++i) {
        int64_t offset = story_offsets[i];
        int64_t length = story_lengths[i];

        if (offset < 0 || length <= 0) {
            out_keep_mask[i] = 0;
            continue;
        }

        const uint8_t* story_ptr = corpus + offset;
        uint64_t h = blt_hash_bytes(story_ptr, static_cast<size_t>(length));

        if (seen_hashes.find(h) == seen_hashes.end()) {
            seen_hashes.insert(h);
            out_keep_mask[i] = 1;
            unique_count++;
        } else {
            out_keep_mask[i] = 0;
        }
    }

    if (out_unique_count) {
        *out_unique_count = unique_count;
    }

    return BLT_SUCCESS;
}

} // extern "C"
