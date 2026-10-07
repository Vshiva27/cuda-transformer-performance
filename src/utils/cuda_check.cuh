#pragma once
// =============================================================================
// cuda_check.cuh — CUDA error checking.
//
// Almost every CUDA runtime function returns a cudaError_t. If we ignore it,
// a failure (out of memory, bad pointer, invalid launch) goes unnoticed and
// the program silently produces garbage. These macros check the value and
// stop the program with a clear message that names the file and line.
// Explained in docs/02_cuda_fundamentals.md, section "Error handling".
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Called by the CUDA_CHECK macro. Not meant to be called directly.
inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "\nCUDA ERROR: %s (%s)\n"
                     "  failed call : %s\n"
                     "  location    : %s:%d\n",
                     cudaGetErrorName(err), cudaGetErrorString(err), expr, file, line);
        std::exit(EXIT_FAILURE);
    }
}

// Wrap any CUDA runtime call:  CUDA_CHECK(cudaMalloc(&ptr, bytes));
// #call turns the code text into a string so the message can show it.
#define CUDA_CHECK(call) cuda_check_impl((call), #call, __FILE__, __LINE__)

// Put this right after a kernel launch: kernel<<<grid, block>>>(...);
//
// A kernel launch returns nothing, so we ask "did the last launch fail?" with
// cudaGetLastError(). This only catches LAUNCH errors (for example a block size
// above 1024). Errors that happen WHILE the kernel runs (for example reading
// outside an array) are reported later, by whatever CUDA call synchronizes next.
// Build with -DCTP_DEBUG_SYNC=ON to wait for every kernel and catch those
// errors at the exact launch that caused them.
#ifdef CTP_DEBUG_SYNC
#define CUDA_CHECK_KERNEL()                       \
    do {                                          \
        CUDA_CHECK(cudaGetLastError());           \
        CUDA_CHECK(cudaDeviceSynchronize());      \
    } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif
