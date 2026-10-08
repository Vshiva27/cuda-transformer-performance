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
// Explained in docs/08_precision.md §9.
// =============================================================================

#include <cstdint>

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

}  // namespace gpu
