// =============================================================================
// vector_add.cu — the first CUDA kernels: C = A + B.
//
// Vector addition is the "hello world" of CUDA. It is too simple to be useful
// on its own, but it teaches the thread-index formula used by EVERY kernel in
// this project, and it is the purest example of a memory-bound workload.
// In a Transformer the same pattern appears as the residual connection:
//     x = x + attention(x)
// Line-by-line explanation: docs/02_cuda_fundamentals.md.
// =============================================================================

#include "vector_add.cuh"
#include "utils/cuda_check.cuh"

// -----------------------------------------------------------------------------
// Version 1: one thread computes exactly one output element.
// -----------------------------------------------------------------------------
__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c, int n) {
    // Global index of this thread: which element it is responsible for.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The last block may have more threads than elements left; those threads
    // must do nothing, otherwise they would write past the end of c.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// -----------------------------------------------------------------------------
// Version 2: grid-stride loop. The grid is smaller than n, so each thread
// processes element i, then i + stride, then i + 2*stride, ...
// -----------------------------------------------------------------------------
__global__ void vector_add_grid_stride_kernel(const float* a, const float* b, float* c, int n) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // total number of threads in the grid

    for (int i = start; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// -----------------------------------------------------------------------------
// Host-side launch functions
// -----------------------------------------------------------------------------
namespace gpu {

void vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
    if (n <= 0) return;  // a grid with 0 blocks is an invalid launch
    // Ceiling division: enough blocks so that grid_size * block_size >= n.
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_naive_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();
}

void vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n,
                            int block_size, int grid_size) {
    if (n <= 0) return;
    vector_add_grid_stride_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();
}

int vector_add_grid_stride_default_grid(int block_size) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    // Ask CUDA: "with this kernel and this block size, how many blocks can
    // live on one SM at the same time?" (limited by threads, registers and
    // shared memory per SM). This is the occupancy calculator.
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, vector_add_grid_stride_kernel, block_size, /*dynamicSMemSize=*/0));

    return sm_count * blocks_per_sm;
}

}  // namespace gpu
