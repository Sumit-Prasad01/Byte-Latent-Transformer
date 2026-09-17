#include "streaming_patcher.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << " BLT Streaming Patcher CLI Tool\n";
    std::cout << "========================================\n";

    float theta_r = 0.8f;
    int32_t max_patch = 16;
    int32_t reset_nl = 1;
    int32_t doc_token = 256;

    BltStreamingPatcher* patcher = blt_streaming_patcher_create(theta_r, max_patch, reset_nl, doc_token);
    if (!patcher) {
        std::cerr << "[FAIL] Failed to create streaming patcher.\n";
        return 1;
    }

    std::vector<int32_t> test_bytes = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!'};
    std::vector<float> test_entropies = {2.1f, 0.5f, 0.4f, 0.3f, 0.4f, 1.9f, 2.8f, 0.7f, 0.6f, 0.5f, 0.4f, 1.5f};

    std::cout << "[INFO] Feeding " << test_bytes.size() << " bytes sequentially:\n";
    int patches_count = 0;
    for (size_t i = 0; i < test_bytes.size(); ++i) {
        int32_t is_boundary = blt_streaming_patcher_feed(patcher, test_bytes[i], test_entropies[i]);
        if (is_boundary == 1) patches_count++;
        std::cout << "  step " << i << ": byte='" << (char)test_bytes[i]
                  << "' H=" << test_entropies[i]
                  << " -> is_boundary=" << is_boundary
                  << " (patch_len=" << blt_streaming_patcher_get_current_patch_len(patcher) << ")\n";
    }

    std::cout << "[INFO] Total patches created: " << patches_count << "\n";
    blt_streaming_patcher_destroy(patcher);

    if (patches_count >= 2) {
        std::cout << "[PASS] Streaming patcher execution verified.\n";
        return 0;
    } else {
        std::cerr << "[FAIL] Streaming patcher created fewer patches than expected.\n";
        return 1;
    }
}
