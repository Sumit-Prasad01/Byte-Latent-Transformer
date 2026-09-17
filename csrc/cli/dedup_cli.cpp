#include "story_dedup.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << " BLT Story Deduplication CLI Tool\n";
    std::cout << "========================================\n";

    if (argc > 1 && std::string(argv[1]) == "--test") {
        std::cout << "[Test Mode] Running internal test suite...\n";

        std::string story1 = "Once upon a time there was a puppy named Spot.";
        std::string story2 = "Once upon a time there was a cat named Whiskers.";
        std::string story3 = "Once upon a time there was a puppy named Spot."; // duplicate of 1

        std::string corpus = story1 + story2 + story3;
        int64_t offsets[3] = {0, static_cast<int64_t>(story1.length()), static_cast<int64_t>(story1.length() + story2.length())};
        int64_t lengths[3] = {static_cast<int64_t>(story1.length()), static_cast<int64_t>(story2.length()), static_cast<int64_t>(story3.length())};
        int8_t keep_mask[3] = {0, 0, 0};
        int64_t unique_count = 0;

        int32_t status = blt_dedup_story_offsets(
            reinterpret_cast<const uint8_t*>(corpus.data()),
            offsets,
            lengths,
            3,
            keep_mask,
            &unique_count
        );

        if (status == BLT_SUCCESS && unique_count == 2 && keep_mask[0] == 1 && keep_mask[1] == 1 && keep_mask[2] == 0) {
            std::cout << "[PASS] Deduplication test passed! Unique: " << unique_count << " / 3\n";
            return 0;
        } else {
            std::cerr << "[FAIL] Deduplication test failed! Unique: " << unique_count << ", status: " << status << "\n";
            return 1;
        }
    }

    std::cout << "Usage: blt_dedup.exe --test\n";
    return 0;
}
