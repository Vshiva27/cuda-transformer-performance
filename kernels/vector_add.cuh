#pragma once
// =============================================================================
// vector_add.cuh — host-side functions that launch the vector-add kernels.
//
// Pointers passed here must point to GPU memory (e.g. DeviceBuffer::data()).
// Each function only QUEUES the work on the GPU and returns immediately.
// =============================================================================

namespace gpu {

// Version 1: one thread per element. Launches ceil(n / block_size) blocks.
void vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size);

// Version 2: grid-stride loop. A fixed number of blocks (grid_size) covers any n;
// each thread handles several elements.
void vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n,
                            int block_size, int grid_size);

// Suggested grid size for the grid-stride kernel: just enough blocks to fill
// every SM of this GPU as much as possible (number of SMs x blocks per SM).
int vector_add_grid_stride_default_grid(int block_size);

}  // namespace gpu
