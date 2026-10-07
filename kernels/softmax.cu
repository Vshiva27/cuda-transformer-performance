// =============================================================================
// softmax.cu — four row-wise softmax kernels.
//
// Softmax turns a row of arbitrary numbers ("scores") into probabilities:
// every output is in (0, 1] and each row sums to 1. In a Transformer it is
// applied to every row of the attention score matrix QK^T / sqrt(d), and to
// the final logits over the vocabulary.
//
// All versions use the numerically stable form (subtract the row maximum
// first, see docs/05_softmax.md section 2). They differ in how the work of one
// row is split across threads and how the row's max and sum are combined.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "softmax.cuh"
#include "utils/cuda_check.cuh"
#include "warp_reduce.cuh"

namespace {
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;  // v3/v4: 8 rows per 256-thread block
}  // namespace

// -----------------------------------------------------------------------------
// v1: one thread handles a whole row, alone.
// -----------------------------------------------------------------------------
__global__ void softmax_naive_kernel(const float* x, float* y, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;  // safe: this kernel has no __syncthreads()

    const float* x_row = x + static_cast<size_t>(row) * cols;  // pointer to x[row][0]
    float* y_row = y + static_cast<size_t>(row) * cols;

    // Pass 1: row maximum.
    float row_max = -INFINITY;
    for (int c = 0; c < cols; ++c) row_max = fmaxf(row_max, x_row[c]);

    // Pass 2: sum of exp(x - max).
    float row_sum = 0.0f;
    for (int c = 0; c < cols; ++c) row_sum += expf(x_row[c] - row_max);

    // Pass 3: normalize.
    const float inv_sum = 1.0f / row_sum;
    for (int c = 0; c < cols; ++c) y_row[c] = expf(x_row[c] - row_max) * inv_sum;
}

// -----------------------------------------------------------------------------
// v2: one block per row. BLOCK threads stride through the row (coalesced),
// then combine their partial results with a tree reduction in shared memory.
// -----------------------------------------------------------------------------
template <int BLOCK>
__global__ void softmax_block_kernel(const float* x, float* y, int rows, int cols) {
    __shared__ float scratch[BLOCK];  // one slot per thread, reused for max and for sum

    const int row = blockIdx.x;  // grid has exactly `rows` blocks
    const int tid = threadIdx.x;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    // ---- Pass 1: maximum. Each thread scans columns tid, tid+BLOCK, ...
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += BLOCK) local_max = fmaxf(local_max, x_row[c]);
    scratch[tid] = local_max;
    __syncthreads();
    // Tree reduction: halve the number of active threads each step.
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();  // outside the if: every thread must reach it
    }
    const float row_max = scratch[0];
    __syncthreads();  // everyone has read scratch[0] before we overwrite scratch

    // ---- Pass 2: sum of exp(x - max), same pattern.
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK) local_sum += expf(x_row[c] - row_max);
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    // ---- Pass 3: normalize and write.
    for (int c = tid; c < cols; c += BLOCK) y_row[c] = expf(x_row[c] - row_max) * inv_sum;
}

// -----------------------------------------------------------------------------
// v3: one warp per row. The 32 lanes stride through the row (coalesced) and
// combine partial results with register shuffles — no shared memory, no
// __syncthreads().
// -----------------------------------------------------------------------------
__global__ void softmax_warp_kernel(const float* x, float* y, int rows, int cols) {
    const int warp_id = threadIdx.x / WARP_SIZE;  // which warp inside the block (0..7)
    const int lane = threadIdx.x % WARP_SIZE;     // my position inside the warp (0..31)
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    // All 32 lanes of a warp share the same `row`, so the whole warp exits
    // together and the shuffles below always have all 32 lanes present.
    if (row >= rows) return;

    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    float local_max = -INFINITY;
    for (int c = lane; c < cols; c += WARP_SIZE) local_max = fmaxf(local_max, x_row[c]);
    const float row_max = warp_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) local_sum += expf(x_row[c] - row_max);
    const float inv_sum = 1.0f / warp_reduce_sum(local_sum);

    for (int c = lane; c < cols; c += WARP_SIZE) y_row[c] = expf(x_row[c] - row_max) * inv_sum;
}

// -----------------------------------------------------------------------------
// v4: online softmax. Each lane keeps a running (max m, sum s) where s is the
// sum of exp(x - m) of the values seen so far. When a new maximum appears, the
// old sum is RESCALED by exp(old_m - new_m). One pass over x gives both the
// max and the sum, so x is read twice in total instead of three times.
// -----------------------------------------------------------------------------

// Merge two partial results (m, s) and (m_other, s_other) into (m, s).
__device__ __forceinline__ void online_merge(float& m, float& s, float m_other, float s_other) {
    const float m_new = fmaxf(m, m_other);
    if (m_new == -INFINITY) return;  // both parts empty (lane saw no elements): nothing to do
    s = s * expf(m - m_new) + s_other * expf(m_other - m_new);
    m = m_new;
}

__global__ void softmax_online_kernel(const float* x, float* y, int rows, int cols) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;

    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;

    // ---- Pass 1: running max and running sum, in ONE read of the row.
    float m = -INFINITY;
    float s = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        const float v = x_row[c];
        if (v > m) {
            s = s * expf(m - v) + 1.0f;  // rescale old sum to the new max; exp(v - v) = 1
            m = v;
        } else if (v > -INFINITY) {
            s += expf(v - m);
        }
        // else: v = -inf (a masked score, e.g. causal attention). exp(-inf) = 0
        // contributes nothing, and skipping it avoids exp(-inf - (-inf)) = NaN
        // when this lane has not seen a finite value yet.
    }

    // ---- Combine the 32 lanes' (m, s) pairs with a shuffle tree.
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const float m_other = __shfl_down_sync(FULL_WARP_MASK, m, offset);
        const float s_other = __shfl_down_sync(FULL_WARP_MASK, s, offset);
        online_merge(m, s, m_other, s_other);
    }
    const float row_max = __shfl_sync(FULL_WARP_MASK, m, 0);
    const float inv_sum = 1.0f / __shfl_sync(FULL_WARP_MASK, s, 0);

    // ---- Pass 2: normalize and write.
    for (int c = lane; c < cols; c += WARP_SIZE) y_row[c] = expf(x_row[c] - row_max) * inv_sum;
}

// -----------------------------------------------------------------------------
// Host-side launch functions
// -----------------------------------------------------------------------------
namespace gpu {

void softmax_naive(const float* d_x, float* d_y, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    const int block = 256;
    const int grid = (rows + block - 1) / block;  // one thread per row
    softmax_naive_kernel<<<grid, block>>>(d_x, d_y, rows, cols);
    CUDA_CHECK_KERNEL();
}

template <int BLOCK>
static void launch_block(const float* d_x, float* d_y, int rows, int cols) {
    softmax_block_kernel<BLOCK><<<rows, BLOCK>>>(d_x, d_y, rows, cols);  // one block per row
    CUDA_CHECK_KERNEL();
}

void softmax_block(const float* d_x, float* d_y, int rows, int cols, int block_size) {
    if (rows <= 0 || cols <= 0) return;
    switch (block_size) {
        case 32: launch_block<32>(d_x, d_y, rows, cols); break;
        case 64: launch_block<64>(d_x, d_y, rows, cols); break;
        case 128: launch_block<128>(d_x, d_y, rows, cols); break;
        case 256: launch_block<256>(d_x, d_y, rows, cols); break;
        case 512: launch_block<512>(d_x, d_y, rows, cols); break;
        case 1024: launch_block<1024>(d_x, d_y, rows, cols); break;
        default:
            std::fprintf(stderr, "softmax_block: unsupported block size %d\n", block_size);
            std::exit(EXIT_FAILURE);
    }
}

void softmax_warp(const float* d_x, float* d_y, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    const int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;  // one warp per row
    softmax_warp_kernel<<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(d_x, d_y, rows, cols);
    CUDA_CHECK_KERNEL();
}

void softmax_online(const float* d_x, float* d_y, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    const int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    softmax_online_kernel<<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(d_x, d_y, rows, cols);
    CUDA_CHECK_KERNEL();
}

}  // namespace gpu
