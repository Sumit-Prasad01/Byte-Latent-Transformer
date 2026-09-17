#include "blt_common.h"
#include "story_dedup.h"
#include "rolling_hash.h"
#include "boundary_rules.h"
#include "streaming_patcher.h"
#include "batch_packing.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " Byte Latent Transformer (BLT) Native Engine CLI\n";
    std::cout << " Version: 0.1.0 (Meta FAIR BLT Architecture)\n";
    std::cout << " Targets: .dll (Dynamic), .a (Static), .exe (CLI)\n";
    std::cout << "====================================================\n";

    if (argc > 1 && std::string(argv[1]) == "--test-all") {
        std::cout << "[INFO] Running full native C++ test suite...\n";

        // 1. Test FNV-1a Hash & Story Dedup
        uint8_t sample_text[] = "Byte Latent Transformer";
        uint64_t h1 = blt_hash_bytes(sample_text, sizeof(sample_text) - 1);
        uint64_t h2 = blt_hash_bytes(sample_text, sizeof(sample_text) - 1);
        if (h1 == 0 || h1 != h2) {
            std::cerr << "[FAIL] Hash reproducibility test failed!\n";
            return 1;
        }
        std::cout << "  [✓] 64-bit Story Hashing: OK (Hash = " << h1 << ")\n";

        // 2. Test RollPolyHash
        uint8_t test_seq[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int32_t n_sizes[] = {3, 4};
        int64_t v_sizes[] = {1000, 1000};
        int64_t out_h[16] = {0};
        int32_t r_status = blt_compute_ngram_hashes(test_seq, 8, n_sizes, v_sizes, 2, BLT_DEFAULT_HASH_PRIME, out_h);
        if (r_status != BLT_SUCCESS) {
            std::cerr << "[FAIL] RollPolyHash failed!\n";
            return 1;
        }
        std::cout << "  [✓] RollPolyHash: OK\n";

        // 3. Test Boundary Rules
        float entropies[] = {1.0f, 1.2f, 3.0f, 1.1f};
        uint8_t bytes[] = {'a', 'b', 'c', 'd'};
        uint8_t bounds[4] = {0};
        int64_t patches = 0;
        int32_t b_status = blt_monotonic_boundary_mask(entropies, bytes, 4, 1.0f, 0, 256, 16, bounds, &patches);
        if (b_status != BLT_SUCCESS || bounds[0] != 1 || bounds[2] != 1) {
            std::cerr << "[FAIL] Boundary rules test failed!\n";
            return 1;
        }
        std::cout << "  [✓] Monotonic Boundary Rules: OK (Patches = " << patches << ")\n";

        // 4. Test Streaming Patcher
        BltStreamingPatcher* sp = blt_streaming_patcher_create(0.8f, 16, 0, 256);
        if (!sp) {
            std::cerr << "[FAIL] Streaming patcher creation failed!\n";
            return 1;
        }
        int32_t b0 = blt_streaming_patcher_feed(sp, 'a', 1.0f);
        int32_t b1 = blt_streaming_patcher_feed(sp, 'b', 1.1f);
        int32_t b2 = blt_streaming_patcher_feed(sp, 'c', 2.5f);
        blt_streaming_patcher_destroy(sp);
        if (b0 != 1 || b1 != 0 || b2 != 1) {
            std::cerr << "[FAIL] Streaming patcher feed behavior failed!\n";
            return 1;
        }
        std::cout << "  [✓] Stateful Streaming Patcher: OK\n";

        // 5. Test Batch Packing
        int32_t patch_counts[] = {100, 150, 300, 200};
        int32_t batch_ids[4] = {0};
        int32_t num_batches = 0;
        int32_t pack_status = blt_pack_patch_batches_c(patch_counts, 4, 300, batch_ids, &num_batches);
        if (pack_status != BLT_SUCCESS || num_batches != 3) {
            std::cerr << "[FAIL] Batch packing test failed! num_batches=" << num_batches << "\n";
            return 1;
        }
        std::cout << "  [✓] Patch-aware Batch Packing: OK (Batches = " << num_batches << ")\n";

        std::cout << "\n[SUCCESS] All 5 native C++ modules passed verification!\n";
        return 0;
    }

    std::cout << "Usage:\n";
    std::cout << "  blt_native.exe --test-all   Run comprehensive native self-test\n";
    std::cout << "  blt_native.exe --version    Print library version\n";
    return 0;
}
