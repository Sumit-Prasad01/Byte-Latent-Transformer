#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "streaming_patcher.h"
#include <cstdlib>

struct BltStreamingPatcher {
    float theta_r;
    int32_t max_patch_size;
    int32_t reset_on_newline;
    int32_t doc_boundary_token;

    float prev_entropy;
    bool has_prev_entropy;
    int32_t current_patch_len;
    int64_t total_bytes_seen;
    int64_t total_patches_created;
};

extern "C" {

BLT_API BltStreamingPatcher* blt_streaming_patcher_create(
    float theta_r,
    int32_t max_patch_size,
    int32_t reset_on_newline,
    int32_t doc_boundary_token
) {
    BltStreamingPatcher* p = (BltStreamingPatcher*)malloc(sizeof(BltStreamingPatcher));
    if (!p) return NULL;

    p->theta_r = theta_r;
    p->max_patch_size = max_patch_size;
    p->reset_on_newline = reset_on_newline;
    p->doc_boundary_token = doc_boundary_token;

    p->prev_entropy = 0.0f;
    p->has_prev_entropy = false;
    p->current_patch_len = 0;
    p->total_bytes_seen = 0;
    p->total_patches_created = 0;

    return p;
}

BLT_API int32_t blt_streaming_patcher_feed(
    BltStreamingPatcher* patcher,
    int32_t byte_val,
    float entropy_val
) {
    if (!patcher) return BLT_ERROR_INVALID_ARGUMENT;

    bool is_boundary = false;

    // First byte ever seen is always the start of patch 0
    if (patcher->total_bytes_seen == 0) {
        is_boundary = true;
    }
    // Document boundary token forces a boundary
    else if (patcher->doc_boundary_token >= 0 && byte_val == patcher->doc_boundary_token) {
        is_boundary = true;
        patcher->has_prev_entropy = false;
    }
    // Hard ceiling on patch length
    else if (patcher->max_patch_size > 0 && patcher->current_patch_len >= patcher->max_patch_size) {
        is_boundary = true;
    }
    // Monotonicity threshold check
    else if (patcher->has_prev_entropy && (entropy_val - patcher->prev_entropy > patcher->theta_r)) {
        is_boundary = true;
    }

    if (is_boundary) {
        patcher->current_patch_len = 1;
        patcher->total_patches_created++;
    } else {
        patcher->current_patch_len++;
    }

    // Context reset rules
    if (patcher->reset_on_newline && byte_val == '\n') {
        patcher->has_prev_entropy = false;
    } else {
        patcher->prev_entropy = entropy_val;
        patcher->has_prev_entropy = true;
    }

    patcher->total_bytes_seen++;
    return is_boundary ? 1 : 0;
}

BLT_API void blt_streaming_patcher_reset(BltStreamingPatcher* patcher) {
    if (!patcher) return;
    patcher->prev_entropy = 0.0f;
    patcher->has_prev_entropy = false;
    patcher->current_patch_len = 0;
    patcher->total_bytes_seen = 0;
    patcher->total_patches_created = 0;
}

BLT_API int32_t blt_streaming_patcher_get_current_patch_len(const BltStreamingPatcher* patcher) {
    if (!patcher) return 0;
    return patcher->current_patch_len;
}

BLT_API void blt_streaming_patcher_destroy(BltStreamingPatcher* patcher) {
    if (patcher) {
        free(patcher);
    }
}

} // extern "C"
