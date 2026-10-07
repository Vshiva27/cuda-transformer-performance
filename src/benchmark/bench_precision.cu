// =============================================================================
// bench_precision.cu — FP32 vs FP16 vs FP16-in/FP32-accumulate, and Tensor Cores.
//
// Experiment A (speed): square GEMMs, n = 256 .. 2048 (multiples of 16 for WMMA)
//   FP32  : v3 tiled-32, v4 register (from Phase 3)
//   FP16  : tiled fp16-in fp32-acc, tiled fp16-in fp16-acc, v5 WMMA (Tensor Cores)
// Experiment B (accuracy vs K): M = N = 64, K = 64 .. 16384. Error of each
//   version against two exact (double) references:
//     "total"  : exact product of the ORIGINAL FP32 inputs
//                -> includes the error of rounding the inputs to FP16
//     "arith"  : exact product of the FP16-ROUNDED inputs
//                -> only the error made inside the kernel (accumulation)
//   Errors are reported as a fraction of max |reference|.
// Experiment C (overflow): positive inputs in [0, 8], K = 8192: the true sums
//   (~130,000) exceed FP16's maximum (65,504).
// Experiment D (Transformer shapes, prefill): FP32 v4 vs FP16 WMMA.
//
// Usage:  ./bench_precision
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <string>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "matmul.cuh"
#include "precision.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "utils/precision_utils.cuh"

static double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

// One problem in both precisions on the GPU.
struct MixedProblem {
    int M, N, K;
    std::vector<float> h_A, h_B;
    DeviceBuffer<float> d_Af, d_Bf, d_C;
    DeviceBuffer<__half> d_Ah, d_Bh;

    MixedProblem(int M_, int N_, int K_, float lo = -1.0f, float hi = 1.0f)
        : M(M_), N(N_), K(K_), h_A(static_cast<size_t>(M_) * K_), h_B(static_cast<size_t>(K_) * N_),
          d_Af(h_A.size()), d_Bf(h_B.size()), d_C(static_cast<size_t>(M_) * N_), d_Ah(h_A.size()), d_Bh(h_B.size()) {
        fill_random(h_A, 11, lo, hi);
        fill_random(h_B, 12, lo, hi);
        d_Af.copy_from_host(h_A);
        d_Bf.copy_from_host(h_B);
        gpu::float_to_half(d_Af.data(), d_Ah.data(), static_cast<int>(h_A.size()));
        gpu::float_to_half(d_Bf.data(), d_Bh.data(), static_cast<int>(h_B.size()));
    }
    std::vector<float> download() {
        std::vector<float> C(static_cast<size_t>(M) * N);
        d_C.copy_to_host(C);
        return C;
    }
};

struct Variant {
    const char* name;
    bool fp16;
    gpu::GemmLaunchFn fp32_fn;
    void (*fp16_fn)(const __half*, const __half*, float*, int, int, int);
    bool needs_tensor_cores;
    double tol;  // allowed scaled error when verifying (Experiment A)
};

static void run(const Variant& v, MixedProblem& p) {
    if (v.fp16) {
        v.fp16_fn(p.d_Ah.data(), p.d_Bh.data(), p.d_C.data(), p.M, p.N, p.K);
    } else {
        v.fp32_fn(p.d_Af.data(), p.d_Bf.data(), p.d_C.data(), p.M, p.N, p.K);
    }
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "precision", info.name);
    const bool tc = gpu::wmma_supported();
    if (!tc) std::printf("No Tensor Cores (compute capability < 7.0): WMMA rows are skipped.\n\n");

    // tol = allowed error as a fraction of max |exact| (same values as tests/test_precision.cu;
    // FP32 variants are checked against the exact product of the FP32 inputs, FP16 variants
    // against the exact product of the FP16-rounded inputs).
    const std::vector<Variant> variants = {
        {"fp32 v3 tiled-32", false,
         [](const float* A, const float* B, float* C, int M, int N, int K) { gpu::matmul_tiled(A, B, C, M, N, K, 32); },
         nullptr, false, 2e-5},
        {"fp32 v4 register", false, gpu::matmul_register, nullptr, false, 2e-5},
        {"fp16 tiled, fp32 acc", true, nullptr, gpu::matmul_tiled_fp16_acc32, false, 2e-5},
        {"fp16 tiled, fp16 acc", true, nullptr, gpu::matmul_tiled_fp16_acc16, false, 2e-2},
        {"v5 WMMA, fp32 acc", true, nullptr, gpu::matmul_wmma, true, 1e-4},
    };

    // ------------------------------------------------------------------------
    // Experiment A: speed
    // ------------------------------------------------------------------------
    std::printf("Experiment A: square GEMM speed (A, B in the stated precision; C in FP32)\n");
    std::printf("(each variant is verified against an exact double-precision product first, n <= 1024)\n");
    for (int n : {256, 512, 1024, 2048}) {
        MixedProblem p(n, n, n);
        std::vector<double> exact, exact16;
        if (n <= 1024) {
            exact.resize(static_cast<size_t>(n) * n);
            exact16.resize(exact.size());
            cpu::matmul_f64(p.h_A.data(), p.h_B.data(), exact.data(), n, n, n);
            const std::vector<float> A16 = round_to_half_on_gpu(p.h_A), B16 = round_to_half_on_gpu(p.h_B);
            cpu::matmul_f64(A16.data(), B16.data(), exact16.data(), n, n, n);
        }
        std::printf("\nn = %d\n  %-22s | %10s | %10s | %12s | %12s\n", n, "variant", "ms", "GFLOP/s", "vs fp32 v4",
                    "scaled err");
        double v4_ms = 0.0;
        for (const Variant& v : variants) {
            if (v.needs_tensor_cores && !tc) continue;
            char err_col[32] = "-";
            if (n <= 1024) {
                run(v, p);
                const ErrorStats e = measure_error(v.fp16 ? exact16 : exact, p.download());
                if (e.non_finite > 0 || e.scaled_err > v.tol) {
                    std::fprintf(stderr, "%s gave a WRONG result at n = %d (scaled error %.2e > %.0e)\n", v.name, n,
                                 e.scaled_err, v.tol);
                    return EXIT_FAILURE;
                }
                std::snprintf(err_col, sizeof(err_col), "%.2e", e.scaled_err);
            }
            const int iters = n >= 2048 ? 10 : 50;
            float ms = time_gpu_ms([&] { run(v, p); }, 2, iters);
            if (v.fp32_fn == gpu::matmul_register) v4_ms = ms;
            char vs[32] = "-";
            if (v4_ms > 0) std::snprintf(vs, sizeof(vs), "%.2fx", v4_ms / ms);
            std::printf("  %-22s | %10.4f | %10.1f | %12s | %12s\n", v.name, ms, gflops(n, n, n, ms), vs, err_col);
            log.add("A speed", v.name, "square " + dims(n, n, n), v.fp16 ? "fp16" : "fp32", ms, gflops(n, n, n, ms),
                    -1, std::string("scaled_err=") + err_col);
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: accuracy vs K
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment B: error vs K (M = N = 64, inputs uniform in [-1, 1])\n");
    std::printf("error = max |out - exact| / max |exact|\n");
    std::printf("  total = vs exact product of original FP32 inputs;  arith = vs exact product of the FP16-rounded inputs\n\n");
    std::printf("  %6s |", "K");
    for (const Variant& v : variants) {
        if (v.needs_tensor_cores && !tc) continue;
        std::printf(" %21s |", v.name);
    }
    std::printf("\n  %6s |", "");
    for (const Variant& v : variants) {
        if (v.needs_tensor_cores && !tc) continue;
        std::printf(" %10s %10s |", "total", "arith");
    }
    std::printf("\n");
    for (int K : {64, 256, 1024, 4096, 16384}) {
        MixedProblem p(64, 64, K);
        std::vector<double> exact(64 * 64), exact16(64 * 64);
        cpu::matmul_f64(p.h_A.data(), p.h_B.data(), exact.data(), 64, 64, K);
        const std::vector<float> A16 = round_to_half_on_gpu(p.h_A), B16 = round_to_half_on_gpu(p.h_B);
        cpu::matmul_f64(A16.data(), B16.data(), exact16.data(), 64, 64, K);

        std::printf("  %6d |", K);
        for (const Variant& v : variants) {
            if (v.needs_tensor_cores && !tc) continue;
            run(v, p);
            const std::vector<float> out = p.download();
            const ErrorStats total = measure_error(exact, out);
            // For FP32 variants the inputs were not rounded, so "arith" = "total".
            const ErrorStats arith = measure_error(v.fp16 ? exact16 : exact, out);
            std::printf(" %10.2e %10.2e |", total.scaled_err, arith.scaled_err);
            char note[96];
            std::snprintf(note, sizeof(note), "total_err=%.3e; arith_err=%.3e", total.scaled_err, arith.scaled_err);
            log.add("B accuracy vs K", v.name, dims(64, 64, K), v.fp16 ? "fp16" : "fp32", -1, -1, -1, note);
        }
        std::printf("\n");
    }

    // ------------------------------------------------------------------------
    // Experiment C: overflow of an FP16 accumulator
    // ------------------------------------------------------------------------
    {
        const int M = 16, N = 16, K = 8192;
        MixedProblem p(M, N, K, 0.0f, 8.0f);
        std::vector<double> exact(M * N);
        cpu::matmul_f64(p.h_A.data(), p.h_B.data(), exact.data(), M, N, K);
        std::printf("\n\nExperiment C: overflow. Inputs in [0, 8], K = %d; FP16 maximum = 65504\n", K);
        std::printf("  largest exact output: %.0f\n", measure_error(exact, std::vector<float>(M * N, 0.0f)).max_abs_ref);
        for (const Variant& v : variants) {
            if (v.needs_tensor_cores && !tc) continue;
            run(v, p);
            const ErrorStats e = measure_error(exact, p.download());
            std::printf("  %-22s : %3d of %d outputs are inf/NaN, scaled error of the rest %.2e\n", v.name, e.non_finite,
                        M * N, e.scaled_err);
            log.add("C overflow", v.name, dims(M, N, K), v.fp16 ? "fp16" : "fp32", -1, -1, -1,
                    "non_finite=" + std::to_string(e.non_finite) + "/" + std::to_string(M * N));
        }
    }

    // ------------------------------------------------------------------------
    // Experiment D: Transformer shapes (prefill, GPT-2 small)
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment D: Transformer GEMMs, prefill of 512 tokens (GPT-2 small)\n");
    struct NamedShape {
        const char* name;
        int M, N, K;
    };
    for (const NamedShape& s : {NamedShape{"QKV projection", 512, 2304, 768}, NamedShape{"MLP up", 512, 3072, 768},
                                NamedShape{"MLP down", 512, 768, 3072}}) {
        MixedProblem p(s.M, s.N, s.K);
        std::printf("\n%s (%d x %d x %d)\n", s.name, s.M, s.N, s.K);
        for (const Variant& v : variants) {
            if (v.needs_tensor_cores && !tc) continue;
            float ms = time_gpu_ms([&] { run(v, p); }, 2, 50);
            std::printf("  %-22s | %10.4f ms | %10.1f GFLOP/s\n", v.name, ms, gflops(s.M, s.N, s.K, ms));
            log.add("D transformer", v.name, std::string(s.name) + " " + dims(s.M, s.N, s.K), v.fp16 ? "fp16" : "fp32",
                    ms, gflops(s.M, s.N, s.K, ms));
        }
    }
    std::printf("\n(Decode shapes with M = 1 are not run with WMMA: it needs M to be a multiple of 16. See docs/08.)\n");
    return EXIT_SUCCESS;
}
