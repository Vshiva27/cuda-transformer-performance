#pragma once
// =============================================================================
// bench_utils.cuh — shared timing helpers for every benchmark.
//
// Rules every measurement follows:
//  1. Warm-up runs first. The first launches pay one-time costs (loading the
//     kernel onto the GPU, raising GPU clocks, filling caches) that do not
//     represent steady-state inference.
//  2. Many timed iterations, averaged, because a single run is noisy.
//  3. GPU work is timed with CUDA events; CPU work with a wall-clock timer.
// =============================================================================

#include "utils/cuda_check.cuh"
#include "utils/timer.cuh"

// Average GPU time (ms) of one call to `fn`. `fn` should only LAUNCH GPU work.
// All iterations are bracketed by a single pair of events: the GPU runs the
// launches back-to-back, so we measure GPU execution time, not CPU overhead.
template <typename Fn>
float time_gpu_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());  // warm-up must finish before we start timing

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    float total_ms = timer.stop_ms();
    return total_ms / iters;
}

// Average wall-clock time (ms) of one call to `fn`. If `fn` launches GPU work,
// it must synchronize itself, otherwise we only measure the launch.
template <typename Fn>
double time_cpu_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();

    CpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    return timer.stop_ms() / iters;
}

// Achieved memory bandwidth in GB/s (1 GB = 1e9 bytes).
inline double bandwidth_gbs(double bytes_moved, double ms) {
    return bytes_moved / (ms * 1e-3) / 1e9;
}
