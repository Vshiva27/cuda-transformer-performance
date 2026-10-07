// =============================================================================
// matmul_register.cu — GEMM version 4: register tiling (2D thread tile).
//
// Problem with v3: for every multiply-add, a thread performs TWO shared-memory
// loads (As[ty][k] and Bs[k][tx]). Shared memory is fast, but not as fast as
// the FMA units, so the shared-memory pipe becomes the bottleneck.
//
// Idea: let each thread compute a TM x TN = 4 x 4 group of C elements instead
// of one. At each k it loads TM values of A and TN values of B into REGISTERS
// (8 loads) and combines them into TM*TN = 16 multiply-adds (an "outer
// product"). Loads per FMA drop from 2 to 0.5, and each register value is
// reused 4 times.
//
// Tile hierarchy:
//   block tile   BM x BN = 64 x 64 elements of C   (one thread block, 256 threads)
//   K step       BK = 8                             (A tile 64x8, B tile 8x64 in shared memory)
//   thread tile  TM x TN = 4 x 4                    (16 accumulators in registers)
//
// Explanation with diagrams: docs/04_matrix_multiplication.md, Part 2.
// =============================================================================

#include "matmul.cuh"
#include "utils/cuda_check.cuh"

namespace {

constexpr int BM = 64;  // rows of C per block
constexpr int BN = 64;  // columns of C per block
constexpr int BK = 8;   // K-depth of one shared-memory tile
constexpr int TM = 4;   // rows of C per thread
constexpr int TN = 4;   // columns of C per thread

constexpr int THREADS_X = BN / TN;                  // 16 threads across the columns
constexpr int THREADS_Y = BM / TM;                  // 16 threads down the rows
constexpr int NUM_THREADS = THREADS_X * THREADS_Y;  // 256 threads per block

}  // namespace

__global__ void matmul_register_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];  // 64 x 8 piece of A
    __shared__ float Bs[BK][BN];  // 8 x 64 piece of B

    // 1D block of 256 threads; we split the thread id into a 16 x 16 grid.
    const int tid = threadIdx.x;
    const int thread_col = tid % THREADS_X;  // 0..15
    const int thread_row = tid / THREADS_X;  // 0..15

    // Top-left corner of this block's 64 x 64 tile of C.
    const int block_row0 = blockIdx.y * BM;
    const int block_col0 = blockIdx.x * BN;

    // This thread owns C elements at
    //   rows    block_row0 + thread_row + i * THREADS_Y   (i = 0..3)
    //   columns block_col0 + thread_col + j * THREADS_X   (j = 0..3)
    // i.e. the same position inside each of the 4 x 4 sub-tiles of 16 x 16.
    // (Why spread out and not 4 x 4 neighbours: see docs, "bank conflicts".)
    float acc[TM][TN] = {};  // 16 accumulators, zero-initialized, held in registers
    float a_reg[TM];
    float b_reg[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Cooperative load of the A tile (64 x 8 = 512 values, 2 per thread).
        // Consecutive i -> consecutive c (column) -> consecutive global addresses.
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            const int r = i / BK;
            const int c = i % BK;
            const int g_row = block_row0 + r;
            const int g_col = k0 + c;
            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }
        // ---- Cooperative load of the B tile (8 x 64 = 512 values, 2 per thread).
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            const int r = i / BN;
            const int c = i % BN;
            const int g_row = k0 + r;
            const int g_col = block_col0 + c;
            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }
        __syncthreads();

        // ---- Compute: for each k, an outer product of 4 A-values x 4 B-values.
        // #pragma unroll makes every loop index a compile-time constant, which
        // is REQUIRED for acc/a_reg/b_reg to stay in registers (registers
        // cannot be indexed by a run-time variable).
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[thread_row + i * THREADS_Y][k];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[k][thread_col + j * THREADS_X];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- Write the 16 results.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = block_row0 + thread_row + i * THREADS_Y;
            const int col = block_col0 + thread_col + j * THREADS_X;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

namespace gpu {

void matmul_register(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    dim3 block(NUM_THREADS);  // 1D block: 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_register_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

}  // namespace gpu
