// =============================================================================
// bench_softmax.cu — softmax experiments.
//
// Experiment A: the shared shape list (same as python/benchmark.py) —
//               CPU vs v1..v4.
// Experiment B: row length sensitivity. The total size is fixed at 16M
//               elements while rows get longer and fewer. Shows when
//               "one warp per row" vs "one block per row" wins, including
//               an LLM vocabulary-sized row (50257 = GPT-2 vocabulary).
// Experiment C: block size for v2.
//
// Metric: softmax is memory-bound, so we report GB/s using the MINIMUM
// traffic: read x once + write y once = 8 bytes per element. Versions that
// read x 2-3 times (relying on caches) still get the same byte count, so a
// lower GB/s directly shows wasted traffic or poor parallelism.
//
// Usage:  ./bench_softmax
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "softmax.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

struct SoftmaxProblem {
    int rows, cols;
    std::vector<float> h_x, h_y;
    DeviceBuffer<float> d_x, d_y;

    SoftmaxProblem(int r, int c)
        : rows(r), cols(c), h_x(static_cast<size_t>(r) * c), h_y(h_x.size()), d_x(h_x.size()), d_y(h_x.size()) {
        fill_random(h_x, 7, -3.0f, 3.0f);
        d_x.copy_from_host(h_x);
    }
    void run(gpu::SoftmaxLaunchFn fn) { fn(d_x.data(), d_y.data(), rows, cols); }
    double min_bytes() const { return 8.0 * rows * cols; }
};

static void verify(const char* name, SoftmaxProblem& p, const std::vector<float>& ref) {
    p.d_y.copy_to_host(p.h_y);
    CompareResult r = compare_arrays(ref, p.h_y, 1e-7, 1e-5 + 1e-7 * p.cols);
    if (!r.ok) {
        std::fprintf(stderr, "%s gave a WRONG result for %dx%d\n", name, p.rows, p.cols);
        print_compare_failure(r, ref, p.h_y);
        std::exit(EXIT_FAILURE);
    }
}

static int iters_for(double bytes) {
    return bytes > 5e8 ? 20 : (bytes > 5e7 ? 50 : 200);
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "softmax", info.name);
    const auto& versions = gpu::softmax_versions();

    // ------------------------------------------------------------------------
    // Experiment A
    // ------------------------------------------------------------------------
    std::printf("Experiment A: softmax over rows, FP32 (shapes shared with python/benchmark.py)\n");
    struct Shape {
        int rows, cols;
    };
    for (Shape s : {Shape{1024, 128}, Shape{4096, 512}, Shape{4096, 1024}, Shape{12 * 1024, 1024}}) {
        SoftmaxProblem p(s.rows, s.cols);
        std::vector<float> ref(p.h_x.size());
        double cpu_ms = time_cpu_ms([&] { cpu::softmax(p.h_x.data(), ref.data(), s.rows, s.cols); }, 0, 3);

        std::printf("\n%d x %d   CPU (double-precision reference): %.3f ms\n", s.rows, s.cols, cpu_ms);
        log.add("A shapes", "cpu (double, 1 thread)", dims(s.rows, s.cols), "fp32", cpu_ms);
        std::printf("  %-16s | %10s | %9s | %7s | %8s | %8s\n", "version", "ms", "GB/s", "% peak", "vs v1",
                    "vs CPU");
        double v1_ms = 0.0;
        for (const auto& v : versions) {
            p.run(v.launch);
            verify(v.name, p, ref);
            float ms = time_gpu_ms([&] { p.run(v.launch); }, 3, iters_for(p.min_bytes()));
            if (v1_ms == 0.0) v1_ms = ms;
            double gbs = bandwidth_gbs(p.min_bytes(), ms);
            std::printf("  %-16s | %10.4f | %9.1f | %6.1f%% | %7.2fx | %7.1fx\n", v.name, ms, gbs,
                        100.0 * gbs / info.peak_bandwidth_gbs, v1_ms / ms, cpu_ms / ms);
            log.add("A shapes", v.name, dims(s.rows, s.cols), "fp32", ms, -1, gbs);
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: same total size, different row lengths
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment B: row length sensitivity (~16M elements each), GB/s per version\n\n");
    std::printf("  %8s x %-8s |", "rows", "cols");
    for (const auto& v : versions) std::printf(" %15s |", v.name);
    std::printf("\n");
    for (Shape s : {Shape{1 << 19, 32}, Shape{1 << 16, 256}, Shape{1 << 14, 1024}, Shape{1 << 12, 4096},
                    Shape{1 << 10, 16384}, Shape{1 << 8, 65536}, Shape{334, 50257}, Shape{1, 50257}}) {
        SoftmaxProblem p(s.rows, s.cols);
        std::vector<float> ref(p.h_x.size());
        cpu::softmax(p.h_x.data(), ref.data(), s.rows, s.cols);
        std::printf("  %8d x %-8d |", s.rows, s.cols);
        for (const auto& v : versions) {
            p.run(v.launch);
            verify(v.name, p, ref);
            float ms = time_gpu_ms([&] { p.run(v.launch); }, 2, 20);
            std::printf(" %15.1f |", bandwidth_gbs(p.min_bytes(), ms));
            log.add("B row length", v.name, dims(s.rows, s.cols), "fp32", ms, -1, bandwidth_gbs(p.min_bytes(), ms));
        }
        std::printf("\n");
    }

    // ------------------------------------------------------------------------
    // Experiment C: block size for v2
    // ------------------------------------------------------------------------
    for (Shape s : {Shape{4096, 1024}, Shape{1 << 8, 65536}}) {
        SoftmaxProblem p(s.rows, s.cols);
        std::printf("\n\nExperiment C: v2 block size, %d x %d\n\n", s.rows, s.cols);
        std::printf("  %10s | %10s | %9s\n", "block size", "ms", "GB/s");
        for (int bs : {32, 64, 128, 256, 512, 1024}) {
            float ms = time_gpu_ms([&] { gpu::softmax_block(p.d_x.data(), p.d_y.data(), s.rows, s.cols, bs); }, 2,
                                   50);
            std::printf("  %10d | %10.4f | %9.1f\n", bs, ms, bandwidth_gbs(p.min_bytes(), ms));
            log.add("C block size", "v2 block/row bs=" + std::to_string(bs), dims(s.rows, s.cols), "fp32", ms, -1,
                    bandwidth_gbs(p.min_bytes(), ms));
        }
    }
    return EXIT_SUCCESS;
}
