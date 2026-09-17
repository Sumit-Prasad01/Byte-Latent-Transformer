cmake_minimum_required(VERSION 3.15)
project(blt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Core source files
set(CORE_SOURCES
    src/story_dedup.cpp
    src/rolling_hash.cpp
    src/boundary_rules.cpp
    src/streaming_patcher.cpp
    src/batch_packing.cpp
    src/blt_c_api.cpp
)

# 1. Static Library (.a / .lib)
add_library(blt_native_static STATIC ${CORE_SOURCES})
set_target_properties(blt_native_static PROPERTIES OUTPUT_NAME "blt_native")

# 2. Dynamic Shared Library (.dll / .so)
add_library(blt_native_shared SHARED ${CORE_SOURCES})
target_compile_definitions(blt_native_shared PRIVATE BLT_BUILD_DLL)
set_target_properties(blt_native_shared PROPERTIES OUTPUT_NAME "blt_native")

# 3. Standalone CLI Executables (.exe)
add_executable(blt_native cli/blt_native_cli.cpp)
target_link_libraries(blt_native PRIVATE blt_native_static)

add_executable(blt_dedup cli/dedup_cli.cpp)
target_link_libraries(blt_dedup PRIVATE blt_native_static)

add_executable(blt_rolling_hash_bench cli/rolling_hash_bench.cpp)
target_link_libraries(blt_rolling_hash_bench PRIVATE blt_native_static)

add_executable(blt_boundary_rules_test cli/boundary_rules_test.cpp)
target_link_libraries(blt_boundary_rules_test PRIVATE blt_native_static)

add_executable(blt_streaming_patcher_cli cli/streaming_patcher_cli.cpp)
target_link_libraries(blt_streaming_patcher_cli PRIVATE blt_native_static)
