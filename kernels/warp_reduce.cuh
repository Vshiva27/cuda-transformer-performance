#pragma once
// =============================================================================
// warp_reduce.cuh — reductions inside one warp using shuffle instructions.
//
// A "reduction" combines many values into one (sum, max, ...). Inside a warp,
// threads can read each other's REGISTERS directly with __shfl_*_sync, with no
// shared memory and no __syncthreads(). Used by softmax (Phase 5) and
// LayerNorm (Phase 6). Explained in docs/05_softmax.md, section "Warp shuffles".
//
// Requirement: all 32 lanes of the warp must call these functions together
// (the mask 0xffffffff promises the hardware that all 32 lanes participate).
// =============================================================================

#include <cuda_runtime.h>

constexpr unsigned FULL_WARP_MASK = 0xffffffffu;  // one bit per lane, all 32 set

// Sum of `v` over the 32 lanes of the warp; every lane receives the result.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Tree reduction: after offset 16 lanes 0..15 hold pair sums, after 8
    // lanes 0..7 hold sums of 4, ... after 1 lane 0 holds the total.
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset);  // read v from lane (my_lane + offset)
    }
    // Broadcast lane 0's total to every lane.
    return __shfl_sync(FULL_WARP_MASK, v, 0);
}

// Maximum of `v` over the 32 lanes of the warp; every lane receives the result.
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset));
    }
    return __shfl_sync(FULL_WARP_MASK, v, 0);
}

// Sum of `v` over ALL threads of a block of BLOCK threads; every thread
// receives the result. Two levels (docs/06_layernorm.md, section 6):
//   1. each warp sums its 32 values with shuffles;
//   2. each warp's total goes to shared memory; then every warp reads those
//      BLOCK/32 totals and sums them with shuffles again.
// `shared` must hold BLOCK/32 floats. Contains __syncthreads(): every thread
// of the block must call this function.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32, at most 1024");
    constexpr int NUM_WARPS = BLOCK / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    v = warp_reduce_sum(v);           // level 1: total of my warp (in every lane)
    if (lane == 0) shared[warp] = v;  // one slot per warp
    __syncthreads();                  // all warp totals are written

    // Level 2: every warp sums the NUM_WARPS totals, so every thread ends up
    // with the block total (no extra broadcast step needed).
    v = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    v = warp_reduce_sum(v);
    __syncthreads();  // everyone has read `shared` before the next call overwrites it
    return v;
}
