#pragma once
// =============================================================================
// matmul.cuh — host-side launch functions for all GEMM (matrix multiply) versions.
//
// Every version computes C = A x B with row-major FP32 matrices:
//   A: M x K,  B: K x N,  C: M x N      (all pointers are GPU memory)
// Every function only queues the work and returns immediately.
// Explained in docs/04_matrix_multiplication.md.
// =============================================================================

#include <vector>

namespace gpu {

// Version 1 — naive. One thread per C element; threadIdx.x selects the ROW.
// Neighbouring threads in a warp work on different rows -> uncoalesced access.
void matmul_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
                  int block_x = 32, int block_y = 8);

// Version 2 — coalesced. Same work per thread, but threadIdx.x selects the COLUMN.
// Neighbouring threads read neighbouring elements of B and write neighbouring
// elements of C -> coalesced access.
void matmul_coalesced(const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
                      int block_x = 32, int block_y = 8);

// Version 3 — shared-memory tiling. A block of tile x tile threads computes a
// tile x tile square of C, staging tiles of A and B in shared memory.
// Supported tile sizes: 8, 16, 32.
void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
                  int tile = 32);

// Version 4 — register tiling. 256-thread blocks compute 64 x 64 tiles of C;
// each thread computes a 4 x 4 group of outputs held in registers.
void matmul_register(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);

// ---------------------------------------------------------------------------
// A list of all versions in their default configuration, so tests and
// benchmarks can loop over them. Adding a new version = adding one line here.
// ---------------------------------------------------------------------------
using GemmLaunchFn = void (*)(const float*, const float*, float*, int, int, int);

struct GemmVersion {
    const char* name;
    GemmLaunchFn launch;
};

inline const std::vector<GemmVersion>& gemm_versions() {
    // Captureless lambdas convert automatically to plain function pointers.
    static const std::vector<GemmVersion> versions = {
        {"v1 naive", [](const float* A, const float* B, float* C, int M, int N, int K) {
             matmul_naive(A, B, C, M, N, K);
         }},
        {"v2 coalesced", [](const float* A, const float* B, float* C, int M, int N, int K) {
             matmul_coalesced(A, B, C, M, N, K);
         }},
        {"v3 tiled-32", [](const float* A, const float* B, float* C, int M, int N, int K) {
             matmul_tiled(A, B, C, M, N, K, 32);
         }},
        {"v4 register 4x4", [](const float* A, const float* B, float* C, int M, int N, int K) {
             matmul_register(A, B, C, M, N, K);
         }},
    };
    return versions;
}

}  // namespace gpu
