// =============================================================================
// matmul_wmma.cu — GEMM v5: warp-level matrix multiply on Tensor Cores.
//
// Tensor Cores are dedicated matrix-multiply units inside each SM (compute
// capability 7.0+: V100, T4, A100, L4, H100, ...). One Tensor Core operation
// multiplies small matrix tiles in a single step instead of one FMA per
// thread per element.
//
// We use them through WMMA ("Warp Matrix Multiply-Accumulate", <mma.h>):
// all 32 threads of a warp TOGETHER
//   load a 16x16 tile of A  (FP16) into a "fragment",
//   load a 16x16 tile of B  (FP16) into a fragment,
//   compute  C_tile += A_tile x B_tile  with FP32 accumulation,
// and finally store the 16x16 FP32 result. A fragment is spread across the
// registers of the 32 threads in a hardware-defined way that we never need to
// know — that is what makes this a warp-level operation.
//
// This is a deliberately simple version: each warp reads its tiles directly
// from global memory (through L1/L2), without shared-memory staging.
// Explained line by line in docs/08_precision.md.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <mma.h>

#include "precision.cuh"
#include "utils/cuda_check.cuh"

using namespace nvcuda;  // WMMA lives in the namespace nvcuda::wmma

namespace {
constexpr int WMMA_M = 16;  // tile shape supported for FP16 inputs: 16 x 16 x 16
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_X = 2;  // a block has 2 x 2 warps -> covers a 32 x 32 tile of C
constexpr int WARPS_Y = 2;
constexpr int THREADS_PER_BLOCK = WARPS_X * WARPS_Y * 32;  // 128
}  // namespace

__global__ void matmul_wmma_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700  // WMMA instructions only exist on sm_70+
    // Which 16 x 16 tile of C does my warp compute?
    const int warp_id = threadIdx.x / 32;     // 0..3
    const int warp_row = warp_id / WARPS_X;   // 0..1
    const int warp_col = warp_id % WARPS_X;   // 0..1
    const int tile_row = (blockIdx.y * WARPS_Y + warp_row) * WMMA_M;  // first row of the tile
    const int tile_col = (blockIdx.x * WARPS_X + warp_col) * WMMA_N;  // first column of the tile
    if (tile_row >= M || tile_col >= N) return;  // whole warp exits together

    // Fragments: opaque per-warp containers held in registers.
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);  // C_tile = 0

    // Walk along K, 16 at a time — the same structure as the tiled GEMM.
    for (int k = 0; k < K; k += WMMA_K) {
        // Pointer to the top-left element of the tile, plus the row length
        // ("leading dimension") so WMMA knows how far apart rows are.
        wmma::load_matrix_sync(a_frag, A + tile_row * K + k, K);   // A[tile_row..+15][k..k+15]
        wmma::load_matrix_sync(b_frag, B + k * N + tile_col, N);   // B[k..k+15][tile_col..+15]
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);            // C_tile += A_tile x B_tile
    }

    wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
#endif
}

namespace gpu {

bool wmma_supported() {
    static int major = -1;  // queried once, then remembered
    if (major < 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    }
    return major >= 7;
}

void matmul_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    if (!wmma_supported()) {
        std::fprintf(stderr, "matmul_wmma: this GPU has no Tensor Cores (needs compute capability >= 7.0)\n");
        std::exit(EXIT_FAILURE);
    }
    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        std::fprintf(stderr, "matmul_wmma: M, N, K must be multiples of 16 (got %d, %d, %d)\n", M, N, K);
        std::exit(EXIT_FAILURE);
    }
    if (M == 0 || N == 0) return;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + WARPS_X * WMMA_N - 1) / (WARPS_X * WMMA_N), (M + WARPS_Y * WMMA_M - 1) / (WARPS_Y * WMMA_M));
    matmul_wmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

}  // namespace gpu
