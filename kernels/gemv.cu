// =============================================================================
// gemv.cu — matrix-vector product for decoding, with FP32, FP16 or INT8 weights.
//
// One WARP computes one output y[n]: its 32 lanes walk along row n of W
// together (consecutive lanes read consecutive addresses, so every load is
// coalesced), each lane keeps a partial sum, and a warp shuffle reduction
// (warp_reduce.cuh) adds the 32 partial sums.
//
// Each lane loads 16 bytes per instruction (one uint4): 4 FP32 weights,
// 8 FP16 weights or 16 INT8 weights. A warp therefore reads 512 contiguous
// bytes per step, whatever the weight type: the weight type only changes how
// many steps a row takes. Rows whose length in bytes is not a multiple of 16
// (or a W that is not 16-byte aligned) use a one-weight-per-load fallback.
//
// gemv_multirow_kernel (INT8 only, so far) gives each warp several rows and
// reads x once for all of them; see the comment above it.
// Explained in docs/08_precision.md §9.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "quantization.cuh"
#include "utils/cuda_check.cuh"
#include "warp_reduce.cuh"

// Weight -> float. INT8 -> float is exact (every integer up to 2^24 is a float).
__device__ __forceinline__ float weight_to_float(float w) { return w; }
__device__ __forceinline__ float weight_to_float(__half w) { return __half2float(w); }
__device__ __forceinline__ float weight_to_float(int8_t w) { return static_cast<float>(w); }

// `scale` may be nullptr (FP32 / FP16 weights): then the sum is stored as is.
template <typename WT, bool VECTORIZED>
__global__ void gemv_kernel(const WT* __restrict__ W, const float* __restrict__ x, const float* __restrict__ scale,
                            float* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;  // one row per warp
    // `row` is the same for all 32 lanes of a warp, so a warp exits as a whole
    // and the full-warp shuffles below are safe.
    if (row >= N) return;

    const WT* w_row = W + static_cast<size_t>(row) * K;
    float acc = 0.0f;

    if (VECTORIZED) {
        constexpr int VEC = 16 / sizeof(WT);  // weights per 16-byte load: 4, 8 or 16
        const uint4* w_vec = reinterpret_cast<const uint4*>(w_row);
        const int num_vec = K / VEC;
        for (int v = lane; v < num_vec; v += 32) {
            const uint4 packed = w_vec[v];  // one 16-byte load
            const WT* w = reinterpret_cast<const WT*>(&packed);
            const float4* x4 = reinterpret_cast<const float4*>(x + v * VEC);
#pragma unroll
            for (int i = 0; i < VEC / 4; ++i) {
                const float4 xi = x4[i];  // x is small (K floats) and stays in cache
                acc = fmaf(xi.x, weight_to_float(w[4 * i + 0]), acc);
                acc = fmaf(xi.y, weight_to_float(w[4 * i + 1]), acc);
                acc = fmaf(xi.z, weight_to_float(w[4 * i + 2]), acc);
                acc = fmaf(xi.w, weight_to_float(w[4 * i + 3]), acc);
            }
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            acc = fmaf(x[k], weight_to_float(w_row[k]), acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = scale != nullptr ? acc * scale[row] : acc;
    }
}

// -----------------------------------------------------------------------------
// Multi-row version: one warp computes ROWS consecutive outputs.
//
// gemv_kernel reads x once per row, so every weight costs sizeof(float) = 4 bytes
// of x traffic through L1. With INT8 weights (1 byte) that x traffic is 4x the
// weight traffic and L1, not DRAM, becomes the limit. Here each step loads one
// chunk of x into registers ONCE and multiplies it with the matching chunk of
// all ROWS rows, so x is read once per ROWS rows. All ROWS weight loads are
// issued before any of them is used, so more bytes are in flight per warp.
// Rows past N (last warp) are skipped; the check is the same for all 32 lanes.
// -----------------------------------------------------------------------------
template <typename WT, bool VECTORIZED, int ROWS>
__global__ void gemv_multirow_kernel(const WT* __restrict__ W, const float* __restrict__ x,
                                     const float* __restrict__ scale, float* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * ROWS;
    if (row0 >= N) return;  // whole warp exits together

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    if (VECTORIZED) {
        constexpr int VEC = 16 / sizeof(WT);
        const int num_vec = K / VEC;
        for (int v = lane; v < num_vec; v += 32) {
            uint4 packed[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {  // issue every weight load first
                packed[r] = row0 + r < N
                                ? reinterpret_cast<const uint4*>(W + static_cast<size_t>(row0 + r) * K)[v]
                                : make_uint4(0, 0, 0, 0);
            }
            float xs[VEC];  // this chunk of x, read once for all ROWS rows
            const float4* x4 = reinterpret_cast<const float4*>(x + v * VEC);
#pragma unroll
            for (int i = 0; i < VEC / 4; ++i) {
                const float4 xi = x4[i];
                xs[4 * i + 0] = xi.x;
                xs[4 * i + 1] = xi.y;
                xs[4 * i + 2] = xi.z;
                xs[4 * i + 3] = xi.w;
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const WT* w = reinterpret_cast<const WT*>(&packed[r]);
#pragma unroll
                for (int i = 0; i < VEC; ++i) acc[r] = fmaf(xs[i], weight_to_float(w[i]), acc[r]);
            }
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            const float xk = x[k];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (row0 + r < N) acc[r] = fmaf(xk, weight_to_float(W[static_cast<size_t>(row0 + r) * K + k]), acc[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float sum = warp_reduce_sum(acc[r]);  // all 32 lanes take part, also for rows >= N
        if (lane == 0 && row0 + r < N) {
            y[row0 + r] = scale != nullptr ? sum * scale[row0 + r] : sum;
        }
    }
}

template <typename WT, int ROWS>
static void launch_gemv_multirow(const WT* d_W, const float* d_x, const float* d_scale, float* d_y, int N, int K) {
    constexpr int BLOCK = 256;  // 8 warps = 8 * ROWS output rows per block
    constexpr int ROWS_PER_BLOCK = (BLOCK / 32) * ROWS;
    const int grid = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const bool aligned = (static_cast<size_t>(K) * sizeof(WT)) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(d_W) % 16 == 0 && reinterpret_cast<uintptr_t>(d_x) % 16 == 0;
    if (aligned) {
        gemv_multirow_kernel<WT, true, ROWS><<<grid, BLOCK>>>(d_W, d_x, d_scale, d_y, N, K);
    } else {
        gemv_multirow_kernel<WT, false, ROWS><<<grid, BLOCK>>>(d_W, d_x, d_scale, d_y, N, K);
    }
    CUDA_CHECK_KERNEL();
}

template <typename WT>
static void launch_gemv(const WT* d_W, const float* d_x, const float* d_scale, float* d_y, int N, int K) {
    if (N <= 0) return;
    constexpr int BLOCK = 256;  // 8 warps = 8 output rows per block
    const int grid = (N + BLOCK / 32 - 1) / (BLOCK / 32);
    // The 16-byte path needs every row (and x) to start on a 16-byte boundary.
    const bool aligned = (static_cast<size_t>(K) * sizeof(WT)) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(d_W) % 16 == 0 && reinterpret_cast<uintptr_t>(d_x) % 16 == 0;
    if (aligned) {
        gemv_kernel<WT, true><<<grid, BLOCK>>>(d_W, d_x, d_scale, d_y, N, K);
    } else {
        gemv_kernel<WT, false><<<grid, BLOCK>>>(d_W, d_x, d_scale, d_y, N, K);
    }
    CUDA_CHECK_KERNEL();
}

namespace gpu {

void gemv_fp32(const float* d_W, const float* d_x, float* d_y, int N, int K) {
    launch_gemv<float>(d_W, d_x, nullptr, d_y, N, K);
}

void gemv_fp16(const __half* d_W, const float* d_x, float* d_y, int N, int K) {
    launch_gemv<__half>(d_W, d_x, nullptr, d_y, N, K);
}

void gemv_int8(const int8_t* d_q, const float* d_scale, const float* d_x, float* d_y, int N, int K) {
    launch_gemv<int8_t>(d_q, d_x, d_scale, d_y, N, K);
}

void gemv_int8_multirow(const int8_t* d_q, const float* d_scale, const float* d_x, float* d_y, int N, int K,
                        int rows_per_warp) {
    if (N <= 0) return;
    switch (rows_per_warp) {
        case 1: launch_gemv_multirow<int8_t, 1>(d_q, d_x, d_scale, d_y, N, K); break;
        case 2: launch_gemv_multirow<int8_t, 2>(d_q, d_x, d_scale, d_y, N, K); break;
        case 4: launch_gemv_multirow<int8_t, 4>(d_q, d_x, d_scale, d_y, N, K); break;
        case 8: launch_gemv_multirow<int8_t, 8>(d_q, d_x, d_scale, d_y, N, K); break;
        default:
            std::fprintf(stderr, "gemv_int8_multirow: rows_per_warp must be 1, 2, 4 or 8 (got %d)\n", rows_per_warp);
            std::exit(EXIT_FAILURE);
    }
}

}  // namespace gpu
