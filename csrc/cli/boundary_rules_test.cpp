#include "boundary_rules.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "========================================\n";
    std::cout << " BLT Boundary Rules Native Test\n";
    std::cout << "========================================\n";

    // Synthetic entropy curve
    std::vector<float> entropy = {1.0f, 1.2f, 2.5f, 2.6f, 4.0f, 1.1f};
    std::vector<uint8_t> bytes = {'a', 'b', 'c', '\n', 'd', 'e'};
    size_t seq_len = entropy.size();

    std::vector<uint8_t> out_boundaries(seq_len, 0);
    int64_t num_patches = 0;
    float theta_r = 1.0f;

    int32_t status = blt_monotonic_boundary_mask(
        entropy.data(),
        bytes.data(),
        seq_len,
        theta_r,
        1,      // reset on newline
        256,    // doc boundary token
        32,     // max patch size
        out_boundaries.data(),
        &num_patches
    );

    if (status != BLT_SUCCESS) {
        std::cerr << "[FAIL] Monotonic boundary mask computation returned error: " << status << "\n";
        return 1;
    }

    std::cout << "[INFO] Sequence length: " << seq_len << ", Patches formed: " << num_patches << "\n";
    for (size_t i = 0; i < seq_len; ++i) {
        std::cout << "  i=" << i << " byte='" << (char)bytes[i] << "' H=" << entropy[i]
                  << " boundary=" << (int)out_boundaries[i] << "\n";
    }

    // i=0 must be 1
    // i=2: 2.5 - 1.2 = 1.3 > 1.0 -> boundary=1
    // i=3 is '\n', reset context
    // i=4: after reset, H is not compared across newline -> boundary=0 unless jump from baseline
    if (out_boundaries[0] == 1 && out_boundaries[2] == 1) {
        std::cout << "[PASS] Monotonic boundary test passed.\n";
        return 0;
    } else {
        std::cerr << "[FAIL] Unexpected boundary pattern.\n";
        return 1;
    }
}
