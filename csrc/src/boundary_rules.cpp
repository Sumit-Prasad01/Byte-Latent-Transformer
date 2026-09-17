#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "boundary_rules.h"
#include <cmath>

extern "C" {

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
) {
    if (!entropy || !out_boundaries) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_len == 0) {
        if (num_patches_out) *num_patches_out = 0;
        return BLT_SUCCESS;
    }

    int64_t patch_count = 0;
    float prev_entropy = 0.0f;
    bool has_prev = false;
    int32_t current_patch_len = 0;

    for (size_t i = 0; i < seq_len; ++i) {
        bool is_boundary = false;
        uint8_t byte_val = bytes ? bytes[i] : 0;

        // Position 0 is always a boundary
        if (i == 0) {
            is_boundary = true;
        }
        // Document boundary token triggers a boundary
        else if (bytes && doc_boundary_token >= 0 && byte_val == static_cast<uint8_t>(doc_boundary_token)) {
            is_boundary = true;
            has_prev = false; // reset context
        }
        // Exceeded maximum patch size
        else if (max_patch_size > 0 && current_patch_len >= max_patch_size) {
            is_boundary = true;
        }
        // Monotonic condition: H(x_i) - H(x_{i-1}) > theta_r
        else if (has_prev && (entropy[i] - prev_entropy > theta_r)) {
            is_boundary = true;
        }

        if (is_boundary) {
            out_boundaries[i] = 1;
            patch_count++;
            current_patch_len = 1;
        } else {
            out_boundaries[i] = 0;
            current_patch_len++;
        }

        // Check for newline reset
        if (reset_on_newline && bytes && byte_val == '\n') {
            has_prev = false;
        } else {
            prev_entropy = entropy[i];
            has_prev = true;
        }
    }

    if (num_patches_out) {
        *num_patches_out = patch_count;
    }

    return BLT_SUCCESS;
}

BLT_API int32_t blt_global_boundary_mask(
    const float* entropy,
    size_t seq_len,
    float theta_g,
    uint8_t* out_boundaries,
    int64_t* num_patches_out
) {
    if (!entropy || !out_boundaries) {
        return BLT_ERROR_INVALID_ARGUMENT;
    }
    if (seq_len == 0) {
        if (num_patches_out) *num_patches_out = 0;
        return BLT_SUCCESS;
    }

    int64_t patch_count = 0;
    for (size_t i = 0; i < seq_len; ++i) {
        if (i == 0 || entropy[i] > theta_g) {
            out_boundaries[i] = 1;
            patch_count++;
        } else {
            out_boundaries[i] = 0;
        }
    }

    if (num_patches_out) {
        *num_patches_out = patch_count;
    }

    return BLT_SUCCESS;
}

} // extern "C"
