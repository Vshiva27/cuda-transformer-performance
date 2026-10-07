// =============================================================================
// bench_layernorm.cu — LayerNorm and kernel-fusion experiments.
//
// Experiment A: LayerNorm v1..v3 on the shared shapes (same as
//               python/benchmark.py), plus the CPU reference.
// Experiment B: residual add + LayerNorm, UNFUSED (vector_add kernel, then
//               LayerNorm v3: 2 launches, h written then read back) versus
//               FUSED (one kernel, h stays in registers).
//
// Metric: GB/s from the MINIMUM bytes each task needs (gamma/beta are tiny
// and ignored):
//   LayerNorm:            read x + write y                 =  8 bytes/element
//   add + LayerNorm task: read x, residual + write h, y    = 16 bytes/element
// The unfused pipeline actually moves 20 bytes/element (it also reads h
// back), which is exactly the waste fusion removes.
//
// Usage:  ./bench_layernorm
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "layernorm.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "vector_add.cuh"

static void verify(const char* what, const std::vector<float>& ref, const std::vector<float>& out) {
    CompareResult r = compare_arrays(ref, out, 5e-5, 1e-5);
    if (!r.ok) {
        std::fprintf(stderr, "%s gave a WRONG result\n", what);
        print_compare_failure(r, ref, out);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "layernorm", info.name);
    const auto& versions = gpu::layernorm_versions();
    const float eps = gpu::LAYERNORM_EPS;

    struct Shape {
        int rows, cols;
        const char* note;
    };
    const std::vector<Shape> shapes = {{512, 768, "GPT-2 small, 512 tokens"},
                                       {2048, 768, "GPT-2 small, 2048 tokens"},
                                       {4096, 1024, "BERT-large/GPT-2 medium, 4096 tokens"},
                                       {8192, 4096, "LLaMA-7B hidden size, 8192 tokens"}};

    // ------------------------------------------------------------------------
    // Experiment A
    // ------------------------------------------------------------------------
    std::printf("Experiment A: LayerNorm over the hidden dimension, FP32\n");
    for (const Shape& s : shapes) {
        const size_t n = static_cast<size_t>(s.rows) * s.cols;
        std::vector<float> x(n), gamma(s.cols), beta(s.cols), ref(n), out(n);
        fill_random(x, 1, -2.0f, 2.0f);
        fill_random(gamma, 2, 0.5f, 1.5f);
        fill_random(beta, 3, -0.5f, 0.5f);
        DeviceBuffer<float> d_x(n), d_g(s.cols), d_b(s.cols), d_y(n);
        d_x.copy_from_host(x);
        d_g.copy_from_host(gamma);
        d_b.copy_from_host(beta);

        double cpu_ms = time_cpu_ms(
            [&] { cpu::layernorm(x.data(), gamma.data(), beta.data(), ref.data(), s.rows, s.cols, eps); }, 0, 3);
        const double min_bytes = 8.0 * n;

        std::printf("\n%d x %d (%s)   CPU (double-precision reference): %.3f ms\n", s.rows, s.cols, s.note, cpu_ms);
        log.add("A shapes", "cpu (double, 1 thread)", dims(s.rows, s.cols), "fp32", cpu_ms);
        std::printf("  %-18s | %10s | %9s | %7s | %8s | %8s\n", "version", "ms", "GB/s", "% peak", "vs v1", "vs CPU");
        double v1_ms = 0.0;
        for (const auto& v : versions) {
            v.launch(d_x.data(), d_g.data(), d_b.data(), d_y.data(), s.rows, s.cols);
            d_y.copy_to_host(out);
            verify(v.name, ref, out);
            const int iters = n > (1 << 24) ? 20 : 100;
            float ms = time_gpu_ms([&] { v.launch(d_x.data(), d_g.data(), d_b.data(), d_y.data(), s.rows, s.cols); },
                                   3, iters);
            if (v1_ms == 0.0) v1_ms = ms;
            double gbs = bandwidth_gbs(min_bytes, ms);
            std::printf("  %-18s | %10.4f | %9.1f | %6.1f%% | %7.2fx | %7.1fx\n", v.name, ms, gbs,
                        100.0 * gbs / info.peak_bandwidth_gbs, v1_ms / ms, cpu_ms / ms);
            log.add("A shapes", v.name, dims(s.rows, s.cols), "fp32", ms, -1, gbs);
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: kernel fusion
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment B: residual add + LayerNorm, unfused (2 kernels) vs fused (1 kernel)\n");
    std::printf("GB/s uses the task's minimum traffic of 16 bytes/element for both.\n\n");
    std::printf("  %-12s | %12s | %12s | %10s | %10s | %8s\n", "shape", "unfused ms", "fused ms", "unfused GB/s",
                "fused GB/s", "speedup");
    for (const Shape& s : shapes) {
        const size_t n = static_cast<size_t>(s.rows) * s.cols;
        std::vector<float> x(n), res(n), gamma(s.cols), beta(s.cols), h_ref(n), y_ref(n), h(n), y(n);
        fill_random(x, 4, -2.0f, 2.0f);
        fill_random(res, 5, -2.0f, 2.0f);
        fill_random(gamma, 6, 0.5f, 1.5f);
        fill_random(beta, 7, -0.5f, 0.5f);
        cpu::add_layernorm(x.data(), res.data(), gamma.data(), beta.data(), h_ref.data(), y_ref.data(), s.rows,
                           s.cols, eps);

        DeviceBuffer<float> d_x(n), d_r(n), d_g(s.cols), d_b(s.cols), d_h(n), d_y(n);
        d_x.copy_from_host(x);
        d_r.copy_from_host(res);
        d_g.copy_from_host(gamma);
        d_b.copy_from_host(beta);

        auto unfused = [&] {
            gpu::vector_add_naive(d_x.data(), d_r.data(), d_h.data(), static_cast<int>(n), 256);
            gpu::layernorm_block(d_h.data(), d_g.data(), d_b.data(), d_y.data(), s.rows, s.cols);
        };
        auto fused = [&] {
            gpu::add_layernorm(d_x.data(), d_r.data(), d_g.data(), d_b.data(), d_h.data(), d_y.data(), s.rows,
                               s.cols);
        };

        // Verify both pipelines (h and y) before timing.
        unfused();
        d_h.copy_to_host(h);
        d_y.copy_to_host(y);
        verify("unfused h", h_ref, h);
        verify("unfused y", y_ref, y);
        fused();
        d_h.copy_to_host(h);
        d_y.copy_to_host(y);
        verify("fused h", h_ref, h);
        verify("fused y", y_ref, y);

        const int iters = n > (1 << 24) ? 20 : 100;
        float unfused_ms = time_gpu_ms(unfused, 3, iters);
        float fused_ms = time_gpu_ms(fused, 3, iters);
        const double task_bytes = 16.0 * n;
        char label[32];
        std::snprintf(label, sizeof(label), "%dx%d", s.rows, s.cols);
        std::printf("  %-12s | %12.4f | %12.4f | %12.1f | %10.1f | %7.2fx\n", label, unfused_ms, fused_ms,
                    bandwidth_gbs(task_bytes, unfused_ms), bandwidth_gbs(task_bytes, fused_ms),
                    unfused_ms / fused_ms);
        log.add("B fusion", "unfused (vector_add + layernorm v3)", dims(s.rows, s.cols), "fp32", unfused_ms, -1,
                bandwidth_gbs(task_bytes, unfused_ms), "2 launches");
        log.add("B fusion", "fused add_layernorm", dims(s.rows, s.cols), "fp32", fused_ms, -1,
                bandwidth_gbs(task_bytes, fused_ms), "1 launch");
    }
    return EXIT_SUCCESS;
}
