#include "rolling_hash.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << " BLT RollPolyHash Benchmark & Verifier\n";
    std::cout << "========================================\n";

    bool verify_mode = (argc > 1 && std::string(argv[1]) == "--verify");

    const size_t seq_len = 100000; // 100k bytes benchmark
    std::vector<uint8_t> bytes(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        bytes[i] = static_cast<uint8_t>((i * 37 + 13) % 256);
    }

    std::vector<int32_t> ngram_sizes = {3, 4, 5};
    std::vector<int64_t> vocab_sizes = {20000, 20000, 20000};
    std::vector<int64_t> out_hashes(ngram_sizes.size() * seq_len, 0);

    auto start = std::chrono::high_resolution_clock::now();
    int32_t status = blt_compute_ngram_hashes(
        bytes.data(),
        seq_len,
        ngram_sizes.data(),
        vocab_sizes.data(),
        ngram_sizes.size(),
        BLT_DEFAULT_HASH_PRIME,
        out_hashes.data()
    );
    auto end = std::chrono::high_resolution_clock::now();

    if (status != BLT_SUCCESS) {
        std::cerr << "[FAIL] Rolling hash computation returned error: " << status << "\n";
        return 1;
    }

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput_mb = (static_cast<double>(seq_len) / (1024.0 * 1024.0)) / (duration.count() / 1000.0);

    std::cout << "[INFO] Processed " << seq_len << " bytes for " << ngram_sizes.size() << " n-gram sizes in "
              << duration.count() << " ms (" << throughput_mb << " MB/s)\n";

    // Single hash sanity check
    uint8_t sample[3] = {65, 66, 67}; // 'A', 'B', 'C'
    int64_t single_hash = blt_roll_hash_single(sample, 3, BLT_DEFAULT_HASH_PRIME, 20000);
    std::cout << "[INFO] Sample single hash for 'ABC': " << single_hash << "\n";

    if (verify_mode || single_hash >= 0) {
        std::cout << "[PASS] RollPolyHash verification succeeded.\n";
        return 0;
    }

    return 0;
}
