// =============================================================================
// matmul_fp16.cu — FP16 conversion kernels and an FP16-input tiled GEMM whose
// accumulator type is a template parameter (float or __half).
//
// The GEMM is GEMM v3 (docs/04 Part 2) with two changes:
//   1. A, B and the shared-memory tiles are __half (2 bytes instead of 4);
//   2. the running sum has type AccT.
// Keeping everything else identical lets the benchmark isolate the effect of
// the accumulator precision. Explained in docs/08_precision.md.
// =============================================================================

#include "precision.cuh"
#include "utils/cuda_check.cuh"

// -----------------------------------------------------------------------------
// Conversions (grid-stride loops, docs/02 section 8)
// -----------------------------------------------------------------------------
__global__ void float_to_half_kernel(const float* in, __half* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = __float2half(in[i]);  // round to the nearest representable half (ties to even)
    }
}

__global__ void half_to_float_kernel(const __half* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = __half2float(in[i]);  // exact: every half value is representable as a float
    }
}

// -----------------------------------------------------------------------------
// Multiply-accumulate, one overload per accumulator type.
// -----------------------------------------------------------------------------
// float accumulator: convert both halves to float (exact), FMA in FP32.
// The product of two halves (11 significant bits each) has at most 22
// significant bits, so it fits in a float (24) exactly: only the ADD rounds.
__device__ __forceinline__ void multiply_add(float& acc, __half a, __half b) {
    acc = fmaf(__half2float(a), __half2float(b), acc);
}
// __half accumulator: __hfma computes a*b + acc and rounds the result to FP16.
__device__ __forceinline__ void multiply_add(__half& acc, __half a, __half b) {
    acc = __hfma(a, b, acc);
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <int TILE, typename AccT>
__global__ void matmul_tiled_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    const __half zero = __float2half(0.0f);

    AccT sum = AccT(0.0f);  // works for float and for __half (it has a float constructor)

    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : zero;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : zero;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            multiply_add(sum, As[ty][k], Bs[k][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = to_float(sum);
    }
}

namespace gpu {

void float_to_half(const float* d_in, __half* d_out, int n) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block < 4096 ? (n + block - 1) / block : 4096;
    float_to_half_kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

void half_to_float(const __half* d_in, float* d_out, int n) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block < 4096 ? (n + block - 1) / block : 4096;
    half_to_float_kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

template <typename AccT>
static void launch_tiled_fp16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_fp16_kernel<TILE, AccT><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void matmul_tiled_fp16_acc32(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    launch_tiled_fp16<float>(d_A, d_B, d_C, M, N, K);
}

void matmul_tiled_fp16_acc16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    launch_tiled_fp16<__half>(d_A, d_B, d_C, M, N, K);
}

}  // namespace gpu
