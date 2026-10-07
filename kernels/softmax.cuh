#pragma once
// =============================================================================
// softmax.cuh — host-side launch functions for all softmax versions.
//
// Every version computes, for each row r of a row-major matrix x (rows x cols):
//     y[r][c] = exp(x[r][c] - max_r) / sum_j exp(x[r][j] - max_r)
// (pointers are GPU memory; functions only queue the work).
// Explained in docs/05_softmax.md.
// =============================================================================

#include <vector>

namespace gpu {

// v1 — one THREAD per row, three sequential passes. Uncoalesced.
void softmax_naive(const float* d_x, float* d_y, int rows, int cols);

// v2 — one BLOCK per row; threads split the row; tree reduction in shared memory.
// block_size: 32, 64, 128, 256, 512 or 1024.
void softmax_block(const float* d_x, float* d_y, int rows, int cols, int block_size = 256);

// v3 — one WARP per row; reductions with warp shuffles (no shared memory).
void softmax_warp(const float* d_x, float* d_y, int rows, int cols);

// v4 — one warp per row, ONLINE softmax: max and sum in a single pass over x.
void softmax_online(const float* d_x, float* d_y, int rows, int cols);

using SoftmaxLaunchFn = void (*)(const float*, float*, int, int);

struct SoftmaxVersion {
    const char* name;
    SoftmaxLaunchFn launch;
};

inline const std::vector<SoftmaxVersion>& softmax_versions() {
    static const std::vector<SoftmaxVersion> versions = {
        {"v1 thread/row", [](const float* x, float* y, int r, int c) { softmax_naive(x, y, r, c); }},
        {"v2 block/row", [](const float* x, float* y, int r, int c) { softmax_block(x, y, r, c); }},
        {"v3 warp/row", [](const float* x, float* y, int r, int c) { softmax_warp(x, y, r, c); }},
        {"v4 warp online", [](const float* x, float* y, int r, int c) { softmax_online(x, y, r, c); }},
    };
    return versions;
}

}  // namespace gpu
