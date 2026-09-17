/**
 * Consolidated C API exports for Byte Latent Transformer native runtime.
 * Bundles all Tier 1 performance kernels into a single native dynamic library
 * (blt_native.dll) and static archive (libblt_native.a).
 */

#ifndef BLT_BUILD_DLL
#define BLT_BUILD_DLL
#endif
#include "blt_common.h"
#include "story_dedup.h"
#include "rolling_hash.h"
#include "boundary_rules.h"
#include "streaming_patcher.h"
#include "batch_packing.h"

extern "C" {

BLT_API const char* blt_get_version(void) {
    return "BLT-Native-0.1.0";
}

BLT_API int32_t blt_sanity_check(void) {
    return 42;
}

} // extern "C"
