// =============================================================================
// matmul_basic.cu — GEMM versions 1 and 2.
//
// Both kernels assign ONE thread to ONE element C[row][col] and compute
//     C[row][col] = sum over k = 0..K-1 of  A[row][k] * B[k][col]
// They differ in ONE thing only: which thread coordinate (x or y) is mapped
// to the row and which to the column. That alone changes the memory access
// pattern of every warp, and therefore the speed.
// Line-by-line explanation: docs/04_matrix_multiplication.md.
// =============================================================================

#include "matmul.cuh"
#include "utils/cuda_check.cuh"

// -----------------------------------------------------------------------------
// Version 1: naive. x -> row, y -> col.
// -----------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// Version 2: coalesced. x -> col, y -> row.
// -----------------------------------------------------------------------------
__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C, int M, int N,
                                        int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// Host-side launch functions
// -----------------------------------------------------------------------------
namespace gpu {

void matmul_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
                  int block_x, int block_y) {
    if (M <= 0 || N <= 0) return;
    dim3 block(block_x, block_y);
    // x covers rows (M), y covers columns (N) — matching the kernel's mapping.
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void matmul_coalesced(const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
                      int block_x, int block_y) {
    if (M <= 0 || N <= 0) return;
    dim3 block(block_x, block_y);
    // x covers columns (N), y covers rows (M).
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_coalesced_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

}  // namespace gpu
