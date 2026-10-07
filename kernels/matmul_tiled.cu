// =============================================================================
// matmul_tiled.cu — GEMM version 3: shared-memory tiling.
//
// Problem with v2: every thread reads its whole row of A and column of B from
// global memory (through the caches). Neighbouring threads read the SAME
// values again and again.
//
// Idea: a block computes a TILE x TILE square of C. It walks along K in steps
// of TILE. At each step the block's threads cooperatively copy one TILE x TILE
// piece of A and one of B into SHARED memory (each element loaded once), wait
// for each other, and then every thread does TILE multiply-adds reading only
// shared memory. Each value loaded from global memory is now used TILE times.
//
// Line-by-line explanation and a step-by-step 4x4 trace:
// docs/04_matrix_multiplication.md, Part 2.
// =============================================================================

#include <cstdio>
#include <cstdlib>

#include "matmul.cuh"
#include "utils/cuda_check.cuh"

// TILE is a template parameter: a compile-time constant. Shared memory array
// sizes must be known at compile time, and constant loop bounds let the
// compiler unroll the inner loop.
template <int TILE>
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory: one copy per BLOCK, visible to all its threads.
    __shared__ float As[TILE][TILE];  // current tile of A: rows of this block, TILE columns of K
    __shared__ float Bs[TILE][TILE];  // current tile of B: TILE rows of K, columns of this block

    const int tx = threadIdx.x;  // column inside the tile
    const int ty = threadIdx.y;  // row inside the tile

    // The C element this thread is responsible for (same mapping as v2).
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // Number of tiles along K, rounded up so a partial last tile is included.
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        // ---- Phase 1: cooperative load. Each thread copies ONE element of
        //      each tile. Outside the matrix we store 0, which adds nothing.
        const int a_col = t * TILE + tx;  // which column of A (= which k)
        const int b_row = t * TILE + ty;  // which row of B    (= which k)
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Wait until EVERY thread of the block has written its element.
        // Without this, a fast thread would read tile entries that a slower
        // thread has not stored yet.
        __syncthreads();

        // ---- Phase 2: compute. TILE multiply-adds from shared memory.
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Wait until EVERY thread has finished reading this tile before the
        // next iteration overwrites As/Bs.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

namespace gpu {

template <int TILE>
static void launch_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();
}

void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int tile) {
    if (M <= 0 || N <= 0) return;
    // Each TILE value is a separately compiled kernel (template instantiation).
    switch (tile) {
        case 8: launch_tiled<8>(d_A, d_B, d_C, M, N, K); break;
        case 16: launch_tiled<16>(d_A, d_B, d_C, M, N, K); break;
        case 32: launch_tiled<32>(d_A, d_B, d_C, M, N, K); break;
        default:
            std::fprintf(stderr, "matmul_tiled: unsupported tile size %d (use 8, 16 or 32)\n", tile);
            std::exit(EXIT_FAILURE);
    }
}

}  // namespace gpu
