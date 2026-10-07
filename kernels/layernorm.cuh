#pragma once
// =============================================================================
// layernorm.cuh — host-side launch functions for LayerNorm and fused
// residual-add + LayerNorm.
//
// For each row r of a row-major matrix x (rows x cols):
//     mean  = (1/cols) * sum_c x[r][c]
//     var   = (1/cols) * sum_c (x[r][c] - mean)^2
//     y[r][c] = (x[r][c] - mean) / sqrt(var + eps) * gamma[c] + beta[c]
// gamma and beta have `cols` elements (one per feature, shared by all rows).
// Explained in docs/06_layernorm.md.
// =============================================================================

#include <vector>

namespace gpu {

constexpr float LAYERNORM_EPS = 1e-5f;  // PyTorch's default

// v1 — one thread per row.
void layernorm_naive(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows,
                     int cols, float eps = LAYERNORM_EPS);

// v2 — one warp per row, warp-shuffle reductions, x read three times.
void layernorm_warp(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows,
                    int cols, float eps = LAYERNORM_EPS);

// v3 — one block per row; the row is held in registers, so x is read from
// global memory exactly once. Supports cols <= 16384.
void layernorm_block(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows,
                     int cols, float eps = LAYERNORM_EPS);

// Fused: h = x + residual;  y = LayerNorm(h). Writes BOTH h (the new residual
// stream, needed by the next layer) and y, in a single kernel. cols <= 16384.
void add_layernorm(const float* d_x, const float* d_residual, const float* d_gamma, const float* d_beta,
                   float* d_h, float* d_y, int rows, int cols, float eps = LAYERNORM_EPS);

using LayerNormLaunchFn = void (*)(const float*, const float*, const float*, float*, int, int);

struct LayerNormVersion {
    const char* name;
    LayerNormLaunchFn launch;
};

inline const std::vector<LayerNormVersion>& layernorm_versions() {
    static const std::vector<LayerNormVersion> versions = {
        {"v1 thread/row", [](const float* x, const float* g, const float* b, float* y, int r, int c) {
             layernorm_naive(x, g, b, y, r, c);
         }},
        {"v2 warp/row", [](const float* x, const float* g, const float* b, float* y, int r, int c) {
             layernorm_warp(x, g, b, y, r, c);
         }},
        {"v3 block/row regs", [](const float* x, const float* g, const float* b, float* y, int r, int c) {
             layernorm_block(x, g, b, y, r, c);
         }},
    };
    return versions;
}

}  // namespace gpu
