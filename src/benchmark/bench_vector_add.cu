// =============================================================================
// bench_vector_add.cu — Experiment 1: CPU vs CUDA on a memory-bound operation.
//
// What it measures, for several vector lengths n:
//   CPU        : cpu::vector_add, wall-clock
//   naive      : GPU kernel time of version 1 (CUDA events)
//   gridstride : GPU kernel time of version 2 (CUDA events)
//   end-to-end : copy A,B to GPU + kernel + copy C back (wall-clock)
//   no-sync    : WRONG timing on purpose (wall-clock around a launch, no sync)
//   bandwidth  : bytes moved by the naive kernel / its time, vs GPU peak
// Then a block-size sweep at the largest size.
//
// Usage:  ./bench_vector_add
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "vector_add.cuh"

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "vector_add", info.name);

    const int warmup = 5;
    const int gpu_iters = 100;
    const int cpu_iters = 10;
    const int block_size = 256;

    const std::vector<int> sizes = {1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26};

    std::printf("Vector add C = A + B, FP32, block size %d (times are averages, ms)\n\n", block_size);
    std::printf("%10s | %9s | %9s | %10s | %10s | %9s | %9s | %7s | %8s\n", "n", "CPU", "naive",
                "gridstride", "end-to-end", "no-sync", "GB/s", "% peak", "CPU/naive");
    std::printf("-----------+-----------+-----------+------------+------------+-----------+-----------+---------+---------\n");

    int largest_n = 0;
    for (int n : sizes) {
        // Three arrays of n floats must live on the GPU. Only use half the
        // free memory, to stay safe on small GPUs or shared Colab machines.
        std::size_t bytes_needed = 3ull * n * sizeof(float);
        if (bytes_needed > info.free_mem_bytes / 2) {
            std::printf("%10d | skipped: needs %zu MB of GPU memory\n", n, bytes_needed >> 20);
            continue;
        }
        largest_n = n;

        // ---- Inputs on the CPU ----
        std::vector<float> h_a(n), h_b(n), h_ref(n), h_out(n);
        fill_random(h_a, 1);
        fill_random(h_b, 2);

        // ---- CPU baseline ----
        double cpu_ms = time_cpu_ms([&] { cpu::vector_add(h_a.data(), h_b.data(), h_ref.data(), n); },
                                    1, cpu_iters);

        // ---- Copy inputs to the GPU ----
        DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
        d_a.copy_from_host(h_a);
        d_b.copy_from_host(h_b);

        // ---- Correctness first: never report the speed of a wrong answer ----
        gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), n, block_size);
        d_c.copy_to_host(h_out);
        CompareResult cmp = compare_arrays(h_ref, h_out, 0.0, 0.0);
        if (!cmp.ok) {
            std::fprintf(stderr, "naive kernel gave a WRONG result for n=%d\n", n);
            print_compare_failure(cmp, h_ref, h_out);
            return EXIT_FAILURE;
        }

        // ---- GPU kernel time ----
        float naive_ms = time_gpu_ms(
            [&] { gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), n, block_size); },
            warmup, gpu_iters);

        int gs_grid = gpu::vector_add_grid_stride_default_grid(block_size);
        float gs_ms = time_gpu_ms(
            [&] {
                gpu::vector_add_grid_stride(d_a.data(), d_b.data(), d_c.data(), n, block_size, gs_grid);
            },
            warmup, gpu_iters);

        // ---- End-to-end: what a caller holding CPU data actually waits for ----
        double e2e_ms = time_cpu_ms(
            [&] {
                d_a.copy_from_host(h_a);
                d_b.copy_from_host(h_b);
                gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), n, block_size);
                d_c.copy_to_host(h_out);
                CUDA_CHECK(cudaDeviceSynchronize());
            },
            1, cpu_iters);

        // ---- Deliberately WRONG timing: no synchronization ----
        // The CPU timer stops as soon as the launch is queued, before the GPU
        // has done the work. This shows why CUDA events (or a sync) are needed.
        CpuTimer wrong_timer;
        wrong_timer.start();
        gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), n, block_size);
        double no_sync_ms = wrong_timer.stop_ms();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Bytes moved by the kernel: read A, read B, write C.
        double bytes = 3.0 * n * sizeof(float);
        double gbs = bandwidth_gbs(bytes, naive_ms);

        std::printf("%10d | %9.4f | %9.4f | %10.4f | %10.4f | %9.4f | %9.1f | %6.1f%% | %8.1fx\n", n,
                    cpu_ms, naive_ms, gs_ms, e2e_ms, no_sync_ms, gbs,
                    100.0 * gbs / info.peak_bandwidth_gbs, cpu_ms / naive_ms);

        const std::string shape = "n=" + std::to_string(n);
        log.add("A sizes", "cpu", shape, "fp32", cpu_ms, -1, bandwidth_gbs(bytes, cpu_ms));
        log.add("A sizes", "naive", shape, "fp32", naive_ms, -1, gbs);
        log.add("A sizes", "grid-stride", shape, "fp32", gs_ms, -1, bandwidth_gbs(bytes, gs_ms));
        log.add("A sizes", "end-to-end (copies + naive)", shape, "fp32", e2e_ms);
        log.add("A sizes", "no-sync (WRONG timing)", shape, "fp32", no_sync_ms);
    }

    if (largest_n == 0) return EXIT_SUCCESS;

    // ---- Experiment: block size sweep ----
    std::printf("\nBlock-size sweep, naive kernel, n = %d\n\n", largest_n);
    std::printf("%10s | %10s | %9s | %9s\n", "block size", "grid size", "ms", "GB/s");
    std::printf("-----------+------------+-----------+----------\n");

    DeviceBuffer<float> d_a(largest_n), d_b(largest_n), d_c(largest_n);
    std::vector<float> h_a(largest_n), h_b(largest_n);
    fill_random(h_a, 1);
    fill_random(h_b, 2);
    d_a.copy_from_host(h_a);
    d_b.copy_from_host(h_b);

    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        if (bs > info.max_threads_per_block) continue;
        float ms = time_gpu_ms(
            [&] { gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), largest_n, bs); },
            warmup, gpu_iters);
        int grid = (largest_n + bs - 1) / bs;
        std::printf("%10d | %10d | %9.4f | %9.1f\n", bs, grid, ms,
                    bandwidth_gbs(3.0 * largest_n * sizeof(float), ms));
        log.add("B block size", "naive block=" + std::to_string(bs), "n=" + std::to_string(largest_n), "fp32", ms, -1,
                bandwidth_gbs(3.0 * largest_n * sizeof(float), ms));
    }

    std::printf("\nGrid-stride kernel used %d blocks (= %d SMs x %d blocks/SM) at block size %d.\n",
                gpu::vector_add_grid_stride_default_grid(block_size), info.sm_count,
                gpu::vector_add_grid_stride_default_grid(block_size) / info.sm_count, block_size);
    return EXIT_SUCCESS;
}
