// =============================================================================
// bench_matmul.cu — GEMM experiments.
//
// Experiment A: square matrices 32 .. 2048 — CPU vs every GPU version (v1..v4).
// Experiment B: Transformer-shaped GEMMs (GPT-2-small sizes) — what real
//               inference layers look like, including M = 1 (token generation).
// Experiment C: block shapes for v2 (coalesced).
// Experiment D: tile sizes for v3 (shared-memory tiling).
//
// Metric: GFLOP/s = 2*M*N*K / time. GEMM does M*N*K multiply-adds, and each
// multiply-add counts as 2 floating-point operations.
//
// Usage:  ./bench_matmul
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "matmul.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

static double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

// Holds the inputs/outputs of one GEMM problem on both CPU and GPU.
struct GemmProblem {
    int M, N, K;
    std::vector<float> h_A, h_B, h_C;
    DeviceBuffer<float> d_A, d_B, d_C;

    GemmProblem(int M_, int N_, int K_)
        : M(M_), N(N_), K(K_),
          h_A(static_cast<std::size_t>(M_) * K_), h_B(static_cast<std::size_t>(K_) * N_),
          h_C(static_cast<std::size_t>(M_) * N_),
          d_A(h_A.size()), d_B(h_B.size()), d_C(h_C.size()) {
        fill_random(h_A, 1);
        fill_random(h_B, 2);
        d_A.copy_from_host(h_A);
        d_B.copy_from_host(h_B);
    }

    void run(gpu::GemmLaunchFn launch) { launch(d_A.data(), d_B.data(), d_C.data(), M, N, K); }

    std::vector<float> download() {
        d_C.copy_to_host(h_C);
        return h_C;
    }
};

// Stop the program if a GPU result does not match the reference.
static void verify(const char* what, const std::vector<float>& ref, const std::vector<float>& out, int K) {
    double tol = gemm_tolerance(K);
    CompareResult r = compare_arrays(ref, out, tol, tol);
    if (!r.ok) {
        std::fprintf(stderr, "%s gave a WRONG result\n", what);
        print_compare_failure(r, ref, out);
        std::exit(EXIT_FAILURE);
    }
}

// Few iterations for slow cases, more for fast ones.
static int iters_for(double flops) {
    if (flops > 4e9) return 10;
    if (flops > 1e8) return 50;
    return 200;
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "matmul", info.name);
    const double peak = info.peak_fp32_gflops;
    const std::vector<gpu::GemmVersion>& versions = gpu::gemm_versions();

    // ------------------------------------------------------------------------
    // Experiment A: square matrices, every version
    // ------------------------------------------------------------------------
    std::printf("Experiment A: square GEMM, n x n x n, FP32, default configuration of each version\n");
    std::printf("(CPU skipped above n = 1024; there the verified v1 result is the reference.\n");
    std::printf(" v1 skipped above n = 2048, where it takes seconds per launch; there v2 is the\n");
    std::printf(" reference: it adds the products in exactly the same order as v1.\n");
    std::printf(" n = 4096 is included so that large GPUs such as A100/H100 are filled.)\n");

    for (int n : {32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        if (3ull * n * n * sizeof(float) > info.free_mem_bytes / 2) {
            std::printf("\nn = %d skipped: not enough GPU memory\n", n);
            continue;
        }
        GemmProblem p(n, n, n);
        const int iters = iters_for(2.0 * n * n * n);
        const bool skip_v1 = n > 2048;

        std::vector<float> ref(p.h_C.size());
        double cpu_ms = -1.0;
        if (n <= 1024) {
            cpu_ms = time_cpu_ms([&] { cpu::matmul(p.h_A.data(), p.h_B.data(), ref.data(), n, n, n); }, 0,
                                 n <= 256 ? 5 : 1);
        } else {
            p.run(versions[skip_v1 ? 1 : 0].launch);
            ref = p.download();
        }

        const std::string shape = "square " + dims(n, n, n);
        if (cpu_ms >= 0) {
            std::printf("\nn = %d   CPU: %.3f ms (%.2f GFLOP/s)\n", n, cpu_ms, gflops(n, n, n, cpu_ms));
            log.add("A square", "cpu (1 thread)", shape, "fp32", cpu_ms, gflops(n, n, n, cpu_ms));
        } else {
            std::printf("\nn = %d   CPU: skipped\n", n);
        }
        std::printf("  %-18s | %10s | %10s | %7s | %8s | %8s | %9s\n", "version", "ms", "GFLOP/s", "% peak",
                    "vs prev", "vs v1", "vs CPU");

        double v1_ms = 0.0, prev_ms = 0.0;
        for (size_t vi = 0; vi < versions.size(); ++vi) {
            const gpu::GemmVersion& v = versions[vi];
            if (vi == 0 && skip_v1) {
                std::printf("  %-18s | skipped (too slow at this size)\n", v.name);
                continue;
            }
            p.run(v.launch);
            verify(v.name, ref, p.download(), n);

            float ms = time_gpu_ms([&] { p.run(v.launch); }, 2, iters);
            if (vi == 0) v1_ms = ms;
            double g = gflops(n, n, n, ms);
            char vs_cpu[32] = "-";
            char vs_prev[32] = "-";
            char vs_v1[32] = "-";
            if (cpu_ms >= 0) std::snprintf(vs_cpu, sizeof(vs_cpu), "%.1fx", cpu_ms / ms);
            if (prev_ms > 0) std::snprintf(vs_prev, sizeof(vs_prev), "%.2fx", prev_ms / ms);
            if (v1_ms > 0) std::snprintf(vs_v1, sizeof(vs_v1), "%.2fx", v1_ms / ms);
            std::printf("  %-18s | %10.4f | %10.1f | %6.1f%% | %8s | %8s | %9s\n", v.name, ms, g,
                        peak > 0 ? 100.0 * g / peak : 0.0, vs_prev, vs_v1, vs_cpu);
            log.add("A square", v.name, shape, "fp32", ms, g);
            prev_ms = ms;
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: shapes from a real Transformer layer (GPT-2 small:
    // hidden size 768, MLP size 3072). M = number of tokens processed at once.
    // ------------------------------------------------------------------------
    struct NamedShape {
        const char* name;
        int M, N, K;
    };
    const std::vector<NamedShape> shapes = {
        {"QKV projection, prefill 512 tokens", 512, 3 * 768, 768},
        {"MLP up-projection, prefill 512 tokens", 512, 3072, 768},
        {"MLP down-projection, prefill 512 tokens", 512, 768, 3072},
        {"QKV projection, decode 1 token", 1, 3 * 768, 768},
        {"MLP up-projection, decode 1 token", 1, 3072, 768},
    };

    std::printf("\n\nExperiment B: Transformer-shaped GEMMs (GPT-2 small), FP32\n");
    std::printf("GB/s = minimum bytes (read A and B once, write C once) / time\n");
    for (const NamedShape& s : shapes) {
        GemmProblem p(s.M, s.N, s.K);
        std::vector<float> ref(p.h_C.size());
        cpu::matmul(p.h_A.data(), p.h_B.data(), ref.data(), s.M, s.N, s.K);
        const int iters = iters_for(2.0 * s.M * s.N * s.K);
        const double min_bytes = 4.0 * (static_cast<double>(s.M) * s.K + static_cast<double>(s.K) * s.N +
                                        static_cast<double>(s.M) * s.N);

        std::printf("\n%s  (M=%d, N=%d, K=%d)\n", s.name, s.M, s.N, s.K);
        std::printf("  %-18s | %10s | %10s | %10s\n", "version", "ms", "GFLOP/s", "GB/s");
        for (const gpu::GemmVersion& v : versions) {
            p.run(v.launch);
            verify(v.name, ref, p.download(), s.K);
            float ms = time_gpu_ms([&] { p.run(v.launch); }, 2, iters);
            std::printf("  %-18s | %10.4f | %10.1f | %10.1f\n", v.name, ms, gflops(s.M, s.N, s.K, ms),
                        bandwidth_gbs(min_bytes, ms));
            log.add("B transformer", v.name, std::string(s.name) + " " + dims(s.M, s.N, s.K), "fp32", ms,
                    gflops(s.M, s.N, s.K, ms), bandwidth_gbs(min_bytes, ms));
        }
    }

    const int n = 1024;
    GemmProblem p(n, n, n);

    // ------------------------------------------------------------------------
    // Experiment C: block shape sweep, v2 coalesced, n = 1024
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment C: block shapes, v2 coalesced, n = %d\n\n", n);
    std::printf("%8s | %8s | %8s | %10s | %10s\n", "block.x", "block.y", "threads", "ms", "GFLOP/s");
    std::printf("---------+----------+----------+------------+-----------\n");
    const std::vector<std::pair<int, int>> blocks = {{32, 1},  {32, 4},  {32, 8}, {32, 16},
                                                     {32, 32}, {16, 16}, {8, 8},  {8, 32}};
    for (const std::pair<int, int>& b : blocks) {
        const int bx = b.first;
        const int by = b.second;
        float ms = time_gpu_ms(
            [&] { gpu::matmul_coalesced(p.d_A.data(), p.d_B.data(), p.d_C.data(), n, n, n, bx, by); }, 2,
            10);
        std::printf("%8d | %8d | %8d | %10.4f | %10.1f\n", bx, by, bx * by, ms, gflops(n, n, n, ms));
        log.add("C block shape", "v2 coalesced block " + dims(bx, by), "square " + dims(n, n, n), "fp32", ms,
                gflops(n, n, n, ms));
    }

    // ------------------------------------------------------------------------
    // Experiment D: tile size sweep, v3 tiled, n = 1024
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment D: tile sizes, v3 tiled, n = %d\n\n", n);
    std::printf("%6s | %8s | %14s | %10s | %10s\n", "tile", "threads", "shared bytes", "ms", "GFLOP/s");
    std::printf("-------+----------+----------------+------------+-----------\n");
    for (int tile : {8, 16, 32}) {
        float ms = time_gpu_ms(
            [&] { gpu::matmul_tiled(p.d_A.data(), p.d_B.data(), p.d_C.data(), n, n, n, tile); }, 2, 10);
        std::printf("%6d | %8d | %14d | %10.4f | %10.1f\n", tile, tile * tile,
                    2 * tile * tile * static_cast<int>(sizeof(float)), ms, gflops(n, n, n, ms));
        log.add("D tile size", "v3 tiled-" + std::to_string(tile), "square " + dims(n, n, n), "fp32", ms,
                gflops(n, n, n, ms));
    }
    return EXIT_SUCCESS;
}
