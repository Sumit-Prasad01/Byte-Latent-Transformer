#ifndef BLT_COMMON_H
#define BLT_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(BLT_BUILD_DLL)
        #define BLT_API __declspec(dllexport)
    #elif defined(BLT_USE_DLL)
        #define BLT_API __declspec(dllimport)
    #else
        #define BLT_API
    #endif
#else
    #if defined(__GNUC__) && __GNUC__ >= 4
        #define BLT_API __attribute__((visibility("default")))
    #else
        #define BLT_API
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status and error codes
typedef enum {
    BLT_SUCCESS = 0,
    BLT_ERROR_INVALID_ARGUMENT = -1,
    BLT_ERROR_OUT_OF_MEMORY = -2,
    BLT_ERROR_BUFFER_TOO_SMALL = -3,
    BLT_ERROR_INVALID_STATE = -4
} BltStatusCode;

// Special token constants
#define BLT_BYTE_VOCAB_SIZE 256
#define BLT_DOC_BOUNDARY_TOKEN 256
#define BLT_BOS_TOKEN 257
#define BLT_EOS_TOKEN 258
#define BLT_PAD_TOKEN 259
#define BLT_TOTAL_VOCAB_SIZE 260

#ifdef __cplusplus
}
#endif

#endif // BLT_COMMON_H
