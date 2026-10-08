// =============================================================================
// bench_quantization.cu — INT8 weight-only quantization for decoding.
//
// Experiment A (speed): one decoding step, y = W x (M = 1 token), with FP32,
//   FP16 and INT8 weights, all through the same GEMV kernel (kernels/gemv.cu).
//   Decode is memory-bound, so the prediction is time ~ bytes of W:
//   FP16 ~2x and INT8 ~4x faster than FP32. Shapes: GPT-2 small (weights
//   2-9 MB in FP32) and a 7B-class model (LLaMA-7B sizes, 64-180 MB), whose
//   weights are larger than the L2 cache of any current GPU, so they measure
//   DRAM, not L2 (docs/09 §21 item 2). Rows whose weights fit in L2 are marked *.
//   GB/s = minimum bytes (W once, scales, x, y) / time.
// Experiment B (accuracy): error of y against the exact product of the ORIGINAL
//   FP32 weights, i.e. the error caused by storing the weights in fewer bits.
//   Uniform weights, then the same weights with 16 outliers (|w| = 50, vs 1 for
//   all others), to compare one scale per row with one scale for the tensor.
//   rel. RMS error = sqrt(sum (y - exact)^2 / sum exact^2).
// Experiment C (rows per warp): the INT8 multi-row kernel with R = 1, 2, 4, 8
//   rows per warp on the 7B-class shapes. Nsight Compute showed the one-row
//   INT8 kernel is limited by L1 (x is re-read for every row), not DRAM
//   (docs/12 §9.1); reading x once per R rows should move it back towards DRAM.
//   Experiment A also includes R = 4 as "int8 weights, 4 rows/warp".
//
// Every kernel is verified against an exact (double) product of the weights it
// actually reads before it is timed.
// Usage:  ./bench_quantization
// Explained in docs/08_precision.md §9.
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "precision.cuh"
#include "quantization.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "utils/precision_utils.cuh"

// INT8_R4: INT8 weights, multi-row kernel with 4 rows per warp (x read once per 4 rows).
enum class Format { FP32, FP16, INT8, INT8_R4 };

// One weight matrix, stored on the GPU in all three formats.
struct DecodeLayer {
    int N, K;
    std::vector<float> h_W, h_x;
    std::vector<std::int8_t> h_q;
    std::vector<float> h_scale;
    DeviceBuffer<float> d_W, d_x, d_y;
    DeviceBuffer<__half> d_Wh;
    DeviceBuffer<std::int8_t> d_q;
    DeviceBuffer<float> d_scale;

    // `W` is N x K, row-major (one row per output).
    DecodeLayer(std::vector<float> W, int N_, int K_, bool per_tensor = false)
        : N(N_), K(K_), h_W(std::move(W)), h_x(K_), h_q(h_W.size()), h_scale(N_), d_W(h_W.size()), d_x(K_), d_y(N_),
          d_Wh(h_W.size()), d_q(h_W.size()), d_scale(N_) {
        fill_random(h_x, 21);
        cpu::quantize_int8(h_W.data(), h_q.data(), h_scale.data(), N, K, per_tensor);
        d_W.copy_from_host(h_W);
        d_x.copy_from_host(h_x);
        gpu::float_to_half(d_W.data(), d_Wh.data(), static_cast<int>(h_W.size()));
        d_q.copy_from_host(h_q);
        d_scale.copy_from_host(h_scale);
    }

    void run(Format f) {
        switch (f) {
            case Format::FP32: gpu::gemv_fp32(d_W.data(), d_x.data(), d_y.data(), N, K); break;
            case Format::FP16: gpu::gemv_fp16(d_Wh.data(), d_x.data(), d_y.data(), N, K); break;
            case Format::INT8: gpu::gemv_int8(d_q.data(), d_scale.data(), d_x.data(), d_y.data(), N, K); break;
            case Format::INT8_R4:
                gpu::gemv_int8_multirow(d_q.data(), d_scale.data(), d_x.data(), d_y.data(), N, K, 4);
                break;
        }
    }

    std::vector<float> download() {
        std::vector<float> y(N);
        d_y.copy_to_host(y);
        return y;
    }

    // Exact product of the weights the kernel for `f` actually reads.
    std::vector<double> exact_for(Format f) const {
        std::vector<double> y(N);
        if (f == Format::INT8 || f == Format::INT8_R4) {
            cpu::gemv_int8_f64(h_q.data(), h_scale.data(), h_x.data(), y.data(), N, K);
        } else if (f == Format::FP16) {
            const std::vector<float> W16 = round_to_half_on_gpu(h_W);
            cpu::gemv_f64(W16.data(), h_x.data(), y.data(), N, K);
        } else {
            cpu::gemv_f64(h_W.data(), h_x.data(), y.data(), N, K);
        }
        return y;
    }

    std::vector<double> exact_fp32_weights() const { return exact_for(Format::FP32); }

    // Bytes a GEMV must move at least: W once, the scales, x, y.
    double min_bytes(Format f) const {
        const double weights = static_cast<double>(N) * K;
        const double io = 4.0 * (static_cast<double>(K) + N);
        switch (f) {
            case Format::FP32: return 4.0 * weights + io;
            case Format::FP16: return 2.0 * weights + io;
            case Format::INT8:
            case Format::INT8_R4: return 1.0 * weights + 4.0 * N + io;
        }
        return 0.0;
    }
    double weight_bytes(Format f) const {
        const double weights = static_cast<double>(N) * K;
        return f == Format::FP32 ? 4.0 * weights : f == Format::FP16 ? 2.0 * weights : weights + 4.0 * N;
    }
};

struct FormatInfo {
    Format f;
    const char* name;
    const char* dtype;
};
static const FormatInfo kFormats[] = {
    {Format::FP32, "fp32 weights", "fp32"},
    {Format::FP16, "fp16 weights", "fp16"},
    {Format::INT8, "int8 weights, per-row scale", "int8"},
    {Format::INT8_R4, "int8 weights, 4 rows/warp", "int8"},
};

static double rel_rms_error(const std::vector<double>& exact, const std::vector<float>& out) {
    double err2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < exact.size(); ++i) {
        const double d = out[i] - exact[i];
        err2 += d * d;
        ref2 += exact[i] * exact[i];
    }
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "quantization", info.name);
    const double peak = info.peak_bandwidth_gbs;

    // ------------------------------------------------------------------------
    // Experiment A: decode speed
    // ------------------------------------------------------------------------
    struct NamedShape {
        const char* name;
        int N, K;  // N outputs, K inputs
    };
    const std::vector<NamedShape> shapes = {
        {"GPT-2 small QKV projection", 3 * 768, 768},
        {"GPT-2 small MLP up", 3072, 768},
        {"GPT-2 small MLP down", 768, 3072},
        {"7B-class QKV projection", 3 * 4096, 4096},
        {"7B-class MLP up", 11008, 4096},
        {"7B-class MLP down", 4096, 11008},
    };

    std::printf("Experiment A: one decoding step y = W x (1 token), FP32 activations and accumulation\n");
    std::printf("GB/s = minimum bytes (W once + scales + x + y) / time; * = weights fit in the %.0f MB L2 cache,\n",
                info.l2_cache_bytes / 1e6);
    std::printf("so that row may be measuring L2 rather than DRAM bandwidth.\n");
    for (const NamedShape& s : shapes) {
        const size_t n = static_cast<size_t>(s.N) * s.K;
        if (7.0 * n > info.free_mem_bytes / 2.0) {  // 4 + 2 + 1 bytes per weight
            std::printf("\n%s skipped: not enough GPU memory\n", s.name);
            continue;
        }
        std::vector<float> W(n);
        fill_random(W, 31);
        DecodeLayer L(std::move(W), s.N, s.K);
        const std::string shape = std::string(s.name) + " " + dims(s.N, s.K);

        std::printf("\n%s  (N=%d outputs, K=%d inputs)\n", s.name, s.N, s.K);
        std::printf("  %-28s | %9s | %10s | %9s | %7s | %8s\n", "weights", "MB", "ms", "GB/s", "% peak", "vs fp32");
        double fp32_ms = 0.0;
        for (const FormatInfo& fi : kFormats) {
            L.run(fi.f);
            const ErrorStats e = measure_error(L.exact_for(fi.f), L.download());
            if (e.non_finite > 0 || e.scaled_err > 1e-4) {
                std::fprintf(stderr, "%s gave a WRONG result for %s (scaled error %.2e)\n", fi.name, s.name,
                             e.scaled_err);
                return EXIT_FAILURE;
            }
            const int iters = n > 8000000 ? 50 : 200;
            const float ms = time_gpu_ms([&] { L.run(fi.f); }, 5, iters);
            if (fi.f == Format::FP32) fp32_ms = ms;
            const double gbs = bandwidth_gbs(L.min_bytes(fi.f), ms);
            const bool in_l2 = L.weight_bytes(fi.f) <= info.l2_cache_bytes;
            char vs[32] = "-";
            if (fp32_ms > 0) std::snprintf(vs, sizeof(vs), "%.2fx", fp32_ms / ms);
            std::printf("  %-28s | %9.2f | %10.4f | %8.1f%s | %6.1f%% | %8s\n", fi.name, L.weight_bytes(fi.f) / 1e6, ms,
                        gbs, in_l2 ? "*" : " ", peak > 0 ? 100.0 * gbs / peak : 0.0, vs);
            log.add("A decode gemv", fi.name, shape, fi.dtype, ms, 2.0 * s.N * s.K / (ms * 1e-3) / 1e9, gbs,
                    in_l2 ? "weights fit in L2" : "");
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: accuracy of the stored weights
    // ------------------------------------------------------------------------
    const int N = 4096, K = 4096;
    std::printf("\n\nExperiment B: error caused by storing W in fewer bits (N = K = %d, W and x uniform in [-1, 1])\n", N);
    std::printf("vs the exact product of the original FP32 weights; rel. RMS = sqrt(sum err^2 / sum exact^2),\n");
    std::printf("max = max |err| / max |exact|\n\n");
    std::printf("  %-22s | %-28s | %12s | %12s\n", "weights", "storage", "rel. RMS err", "max err");

    for (bool outliers : {false, true}) {
        std::vector<float> W(static_cast<size_t>(N) * K);
        fill_random(W, 41);
        if (outliers) {
            // 16 weights at fixed, spread-out positions, 50x larger than any other weight.
            for (int i = 0; i < 16; ++i) {
                W[(static_cast<size_t>(i) * 1048573u + 12345u) % W.size()] = (i % 2 == 0) ? 50.0f : -50.0f;
            }
        }
        const char* case_name = outliers ? "uniform + 16 outliers" : "uniform";
        for (bool per_tensor : {false, true}) {
            DecodeLayer L(W, N, K, per_tensor);
            const std::vector<double> exact = L.exact_fp32_weights();
            struct Row {
                Format f;
                const char* storage;
                const char* dtype;
            };
            std::vector<Row> rows;
            if (!per_tensor) {
                rows.push_back({Format::FP16, "fp16", "fp16"});
                rows.push_back({Format::INT8, "int8, per-row scale", "int8"});
            } else {
                rows.push_back({Format::INT8, "int8, per-tensor scale", "int8"});
            }
            for (const Row& r : rows) {
                L.run(r.f);
                const std::vector<float> y = L.download();
                const double rms = rel_rms_error(exact, y);
                const ErrorStats e = measure_error(exact, y);
                std::printf("  %-22s | %-28s | %12.3e | %12.3e\n", case_name, r.storage, rms, e.scaled_err);
                char note[96];
                std::snprintf(note, sizeof(note), "rel_rms_err=%.3e; max_err=%.3e", rms, e.scaled_err);
                log.add(std::string("B accuracy, ") + case_name, r.storage, dims(N, K), r.dtype, -1, -1, -1, note);
            }
        }
    }
    std::printf("\nOne outlier sets the scale of everything that shares it: with a per-tensor scale every\n");
    std::printf("weight in the matrix gets a step of 50/127 ~= 0.39; with per-row scales only the rows\n");
    std::printf("that contain an outlier do.\n");

    // ------------------------------------------------------------------------
    // Experiment C: rows per warp, INT8 multi-row kernel (7B-class shapes)
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment C: INT8 weights, rows per warp (x read once per R rows), 7B-class shapes\n");
    std::printf("R = 1 is the multi-row kernel with one row; 'one row/warp' is the original kernel (Experiment A)\n");
    for (const NamedShape& s : shapes) {
        if (s.K < 4096) continue;  // only the 7B-class shapes: weights larger than L2
        const size_t n = static_cast<size_t>(s.N) * s.K;
        if (7.0 * n > info.free_mem_bytes / 2.0) continue;
        std::vector<float> W(n);
        fill_random(W, 31);
        DecodeLayer L(std::move(W), s.N, s.K);
        const std::vector<double> exact = L.exact_for(Format::INT8);
        const std::string shape = std::string(s.name) + " " + dims(s.N, s.K);

        std::printf("\n%s  (N=%d, K=%d)\n", s.name, s.N, s.K);
        std::printf("  %-16s | %10s | %9s | %7s | %13s\n", "rows per warp", "ms", "GB/s", "% peak", "vs one row/warp");
        const float base_ms = time_gpu_ms([&] { L.run(Format::INT8); }, 5, 50);
        std::printf("  %-16s | %10.4f | %9.1f | %6.1f%% | %13s\n", "one row/warp", base_ms,
                    bandwidth_gbs(L.min_bytes(Format::INT8), base_ms),
                    peak > 0 ? 100.0 * bandwidth_gbs(L.min_bytes(Format::INT8), base_ms) / peak : 0.0, "1.00x");
        for (int rows : {1, 2, 4, 8}) {
            auto launch = [&] {
                gpu::gemv_int8_multirow(L.d_q.data(), L.d_scale.data(), L.d_x.data(), L.d_y.data(), L.N, L.K, rows);
            };
            launch();
            const ErrorStats e = measure_error(exact, L.download());
            if (e.non_finite > 0 || e.scaled_err > 1e-4) {
                std::fprintf(stderr, "int8 multirow R=%d gave a WRONG result for %s (scaled error %.2e)\n", rows,
                             s.name, e.scaled_err);
                return EXIT_FAILURE;
            }
            const float ms = time_gpu_ms(launch, 5, 50);
            const double gbs = bandwidth_gbs(L.min_bytes(Format::INT8), ms);
            char label[32], vs[32];
            std::snprintf(label, sizeof(label), "R = %d", rows);
            std::snprintf(vs, sizeof(vs), "%.2fx", base_ms / ms);
            std::printf("  %-16s | %10.4f | %9.1f | %6.1f%% | %13s\n", label, ms, gbs,
                        peak > 0 ? 100.0 * gbs / peak : 0.0, vs);
            log.add("C rows per warp", "int8 multirow R=" + std::to_string(rows), shape, "int8", ms,
                    2.0 * s.N * s.K / (ms * 1e-3) / 1e9, gbs);
        }
    }
    return EXIT_SUCCESS;
}
