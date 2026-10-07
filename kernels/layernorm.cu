// =============================================================================
// layernorm.cu — LayerNorm kernels and the fused residual-add + LayerNorm.
//
// LayerNorm normalizes each row (one token's hidden vector) to mean 0 and
// variance 1, then applies a learned per-feature scale (gamma) and shift
// (beta). Every Transformer block uses it twice. Like softmax it is a row-wise
// reduction (two of them: mean and variance) followed by an elementwise step.
//
// Variance is computed with the TWO-PASS method (first the mean, then the mean
// of squared differences), which is numerically stable. See docs/06_layernorm.md
// section 2 for why the one-pass formula E[x^2] - mean^2 is dangerous.
// =============================================================================

#include <cstdio>
#include <cstdlib>

#include "layernorm.cuh"
#include "utils/cuda_check.cuh"
#include "warp_reduce.cuh"

namespace {
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
}  // namespace

// -----------------------------------------------------------------------------
// v1: one thread per row, sequential loops.
// -----------------------------------------------------------------------------
__global__ void layernorm_naive_kernel(const float* x, const float* gamma, const float* beta, float* y, int rows,
                                       int cols, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += x_row[c];
    const float mean = sum / cols;

    float sq_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        const float d = x_row[c] - mean;
        sq_sum += d * d;
    }
    const float inv_std = rsqrtf(sq_sum / cols + eps);  // 1 / sqrt(var + eps)

    for (int c = 0; c < cols; ++c) y_row[c] = (x_row[c] - mean) * inv_std * gamma[c] + beta[c];
}

// -----------------------------------------------------------------------------
// v2: one warp per row (same structure as softmax v3).
// -----------------------------------------------------------------------------
__global__ void layernorm_warp_kernel(const float* x, const float* gamma, const float* beta, float* y, int rows,
                                      int cols, float eps) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;  // whole warp exits together
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) sum += x_row[c];
    const float mean = warp_reduce_sum(sum) / cols;

    float sq_sum = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        const float d = x_row[c] - mean;
        sq_sum += d * d;
    }
    const float inv_std = rsqrtf(warp_reduce_sum(sq_sum) / cols + eps);

    for (int c = lane; c < cols; c += WARP_SIZE) y_row[c] = (x_row[c] - mean) * inv_std * gamma[c] + beta[c];
}

// -----------------------------------------------------------------------------
// Shared by v3 and the fused kernel: the row is already in registers
// (`vals`); compute mean, variance and write y.
//
// `float (&vals)[PER_THREAD]` means "a reference to an array of PER_THREAD
// floats": the function works on the caller's array itself (no copy), and the
// size is part of the type, so loops over it can be fully unrolled and the
// array stays in registers.
// -----------------------------------------------------------------------------
template <int BLOCK, int PER_THREAD>
__device__ __forceinline__ void normalize_row_from_registers(float (&vals)[PER_THREAD], const float* gamma,
                                                             const float* beta, float* y_row, int cols, float eps,
                                                             float* shared) {
    const int tid = threadIdx.x;

    // Mean. Slots beyond the end of the row hold 0, so they add nothing.
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) sum += vals[i];
    const float mean = block_reduce_sum<BLOCK>(sum, shared) / cols;

    // Variance (second pass — over REGISTERS, not global memory).
    float sq_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int c = tid + i * BLOCK;
        if (c < cols) {
            const float d = vals[i] - mean;
            sq_sum += d * d;
        }
    }
    const float inv_std = rsqrtf(block_reduce_sum<BLOCK>(sq_sum, shared) / cols + eps);

    // Normalize, scale, shift, write.
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int c = tid + i * BLOCK;
        if (c < cols) y_row[c] = (vals[i] - mean) * inv_std * gamma[c] + beta[c];
    }
}

// -----------------------------------------------------------------------------
// v3: one block per row, row cached in registers (x read once).
// Thread tid holds columns tid, tid + BLOCK, tid + 2*BLOCK, ... (coalesced).
// -----------------------------------------------------------------------------
template <int BLOCK, int PER_THREAD>
__global__ void layernorm_block_kernel(const float* x, const float* gamma, const float* beta, float* y, int rows,
                                       int cols, float eps) {
    __shared__ float shared[BLOCK / WARP_SIZE];
    const int row = blockIdx.x;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    float vals[PER_THREAD];
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int c = threadIdx.x + i * BLOCK;
        vals[i] = (c < cols) ? x_row[c] : 0.0f;
    }
    normalize_row_from_registers<BLOCK, PER_THREAD>(vals, gamma, beta, y_row, cols, eps, shared);
}

// -----------------------------------------------------------------------------
// Fused residual add + LayerNorm. Identical to v3 except the load step:
// h = x + residual is computed in registers, written out once (the next layer
// needs it), and normalized without ever being read back from memory.
// -----------------------------------------------------------------------------
template <int BLOCK, int PER_THREAD>
__global__ void add_layernorm_kernel(const float* x, const float* residual, const float* gamma, const float* beta,
                                     float* h, float* y, int rows, int cols, float eps) {
    __shared__ float shared[BLOCK / WARP_SIZE];
    const int row = blockIdx.x;
    const size_t offset = static_cast<size_t>(row) * cols;

    float vals[PER_THREAD];
#pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
        const int c = threadIdx.x + i * BLOCK;
        if (c < cols) {
            vals[i] = x[offset + c] + residual[offset + c];
            h[offset + c] = vals[i];
        } else {
            vals[i] = 0.0f;
        }
    }
    normalize_row_from_registers<BLOCK, PER_THREAD>(vals, gamma, beta, y + offset, cols, eps, shared);
}

// -----------------------------------------------------------------------------
// Host-side launch functions
// -----------------------------------------------------------------------------
namespace gpu {

void layernorm_naive(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows, int cols,
                     float eps) {
    if (rows <= 0 || cols <= 0) return;
    const int block = 256;
    layernorm_naive_kernel<<<(rows + block - 1) / block, block>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    CUDA_CHECK_KERNEL();
}

void layernorm_warp(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows, int cols,
                    float eps) {
    if (rows <= 0 || cols <= 0) return;
    const int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    layernorm_warp_kernel<<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    CUDA_CHECK_KERNEL();
}

// Register capacity = BLOCK * PER_THREAD columns. We pick the smallest
// configuration that fits the row, so few register slots are wasted:
//   cols <=  1024: 256 threads x 4   (GPT-2 small: 768, BERT: 768/1024)
//   cols <=  4096: 512 threads x 8   (LLaMA-7B: 4096)
//   cols <= 16384: 1024 threads x 16
static void layernorm_capacity_error(int cols) {
    std::fprintf(stderr, "layernorm_block/add_layernorm: cols = %d exceeds 16384 (use layernorm_warp)\n", cols);
    std::exit(EXIT_FAILURE);
}

void layernorm_block(const float* d_x, const float* d_gamma, const float* d_beta, float* d_y, int rows, int cols,
                     float eps) {
    if (rows <= 0 || cols <= 0) return;
    if (cols <= 1024) {
        layernorm_block_kernel<256, 4><<<rows, 256>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    } else if (cols <= 4096) {
        layernorm_block_kernel<512, 8><<<rows, 512>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    } else if (cols <= 16384) {
        layernorm_block_kernel<1024, 16><<<rows, 1024>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    } else {
        layernorm_capacity_error(cols);
    }
    CUDA_CHECK_KERNEL();
}

void add_layernorm(const float* d_x, const float* d_residual, const float* d_gamma, const float* d_beta, float* d_h,
                   float* d_y, int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    if (cols <= 1024) {
        add_layernorm_kernel<256, 4><<<rows, 256>>>(d_x, d_residual, d_gamma, d_beta, d_h, d_y, rows, cols, eps);
    } else if (cols <= 4096) {
        add_layernorm_kernel<512, 8><<<rows, 512>>>(d_x, d_residual, d_gamma, d_beta, d_h, d_y, rows, cols, eps);
    } else if (cols <= 16384) {
        add_layernorm_kernel<1024, 16><<<rows, 1024>>>(d_x, d_residual, d_gamma, d_beta, d_h, d_y, rows, cols, eps);
    } else {
        layernorm_capacity_error(cols);
    }
    CUDA_CHECK_KERNEL();
}

}  // namespace gpu
