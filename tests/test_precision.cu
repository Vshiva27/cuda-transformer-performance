// =============================================================================
// test_precision.cu — FP16 conversion and FP16 GEMMs.
//
// 1. Conversion: known FP16 roundings (values checked with a CPU emulation of
//    IEEE half rounding, docs/08 section 2): 0.1 -> 0.0999755859375,
//    3.14159 -> 3.140625, 2049 -> 2048 (tie to even), 65504 stays,
//    70000 -> inf (overflow), 1e-5 -> 1.0013580322265625e-05 (subnormal),
//    1e-8 -> 0 (underflow).
// 2. FP16 GEMMs against an exact (double) product of the FP16-ROUNDED inputs,
//    so only the arithmetic inside the kernel is measured. Tolerances are a
//    fraction of the output scale (max |reference|), from the emulation in
//    docs/08 section 6:
//      FP32 accumulation (tiled) : 2e-5   (measured <= 1e-6 for K <= 1024)
//      FP32 accumulation (WMMA)  : 1e-4   (Tensor Core adders may round differently)
//      FP16 accumulation         : 2e-2   (measured <= 6e-3 for K <= 1024)
//    An indexing bug produces errors of the same size as the outputs (~1.0).
// 3. WMMA: the 4x4 hand example from docs/04 zero-padded into 16x16
//    (must be exact), and shapes that are multiples of 16.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "precision.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "utils/precision_utils.cuh"

static int g_failures = 0;
static int g_checks = 0;

static void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

using HalfGemmFn = void (*)(const __half*, const __half*, float*, int, int, int);

// Upload FP32 inputs, convert to FP16 on the GPU, run the GEMM, return C.
static std::vector<float> run_half_gemm(HalfGemmFn fn, const std::vector<float>& A, const std::vector<float>& B, int M,
                                        int N, int K) {
    DeviceBuffer<float> d_Af(A.size()), d_Bf(B.size()), d_C(static_cast<size_t>(M) * N);
    DeviceBuffer<__half> d_A(A.size()), d_B(B.size());
    d_Af.copy_from_host(A);
    d_Bf.copy_from_host(B);
    gpu::float_to_half(d_Af.data(), d_A.data(), static_cast<int>(A.size()));
    gpu::float_to_half(d_Bf.data(), d_B.data(), static_cast<int>(B.size()));
    std::vector<float> C(static_cast<size_t>(M) * N, -999.0f);
    d_C.copy_from_host(C);
    fn(d_A.data(), d_B.data(), d_C.data(), M, N, K);
    d_C.copy_to_host(C);
    return C;
}

static void test_conversion() {
    const std::vector<float> in = {1.0f, 0.1f, 3.14159f, 2049.0f, 65504.0f, 70000.0f, 1e-5f, 1e-8f, -2.5f};
    const std::vector<double> expected = {1.0, 0.0999755859375, 3.140625, 2048.0, 65504.0, INFINITY,
                                          1.0013580322265625e-05, 0.0, -2.5};
    const std::vector<float> out = round_to_half_on_gpu(in);
    for (size_t i = 0; i < in.size(); ++i) {
        const bool ok = std::isinf(expected[i]) ? std::isinf(out[i]) : static_cast<double>(out[i]) == expected[i];
        char msg[160];
        std::snprintf(msg, sizeof(msg), "half(%g) = %.17g, expected %.17g", in[i], out[i], expected[i]);
        expect(ok, msg);
        std::printf("  %s\n", msg);
    }
}

static void test_wmma_hand_example() {
    // 4x4 example from docs/04 (C = [[4,6,2,7],[12,14,10,15],[20,22,18,23],[28,30,26,31]]),
    // placed in the top-left corner of 16x16 zero matrices.
    const int n = 16;
    const float a4[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const float b4[16] = {1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1};
    const float c4[16] = {4, 6, 2, 7, 12, 14, 10, 15, 20, 22, 18, 23, 28, 30, 26, 31};
    std::vector<float> A(n * n, 0.0f), B(n * n, 0.0f), expected(n * n, 0.0f);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            A[r * n + c] = a4[r * 4 + c];
            B[r * n + c] = b4[r * 4 + c];
            expected[r * n + c] = c4[r * 4 + c];
        }
    }
    const std::vector<float> C = run_half_gemm(gpu::matmul_wmma, A, B, n, n, n);
    CompareResult r = compare_arrays(expected, C, 0.0, 0.0);
    if (!r.ok) print_compare_failure(r, expected, C);
    expect(r.ok, "wmma 4x4 hand example padded to 16x16 (exact)");
}

struct HalfVersion {
    const char* name;
    HalfGemmFn fn;
    double tol;  // allowed error as a fraction of max |reference|
    bool needs_16;
};

static void test_gemm_shapes() {
    std::vector<HalfVersion> versions = {
        {"tiled fp16-in fp32-acc", gpu::matmul_tiled_fp16_acc32, 2e-5, false},
        {"tiled fp16-in fp16-acc", gpu::matmul_tiled_fp16_acc16, 2e-2, false},
    };
    if (gpu::wmma_supported()) {
        versions.push_back({"v5 wmma fp16-in fp32-acc", gpu::matmul_wmma, 1e-4, true});
    } else {
        std::printf("  (no Tensor Cores on this GPU: WMMA tests skipped)\n");
    }

    struct Shape {
        int M, N, K;
    };
    const std::vector<Shape> shapes = {{16, 16, 16},   {32, 48, 64},  {64, 64, 64},   {128, 256, 96}, {48, 16, 1024},
                                       {256, 128, 512}, {33, 17, 45}, {100, 70, 300}, {1, 768, 256}};
    for (const Shape& s : shapes) {
        std::vector<float> A(static_cast<size_t>(s.M) * s.K), B(static_cast<size_t>(s.K) * s.N);
        fill_random(A, 100 + s.K, -1.0f, 1.0f);
        fill_random(B, 200 + s.N, -1.0f, 1.0f);
        // Exact product of the values the kernels actually see (FP16-rounded).
        const std::vector<float> A16 = round_to_half_on_gpu(A), B16 = round_to_half_on_gpu(B);
        std::vector<double> ref(static_cast<size_t>(s.M) * s.N);
        cpu::matmul_f64(A16.data(), B16.data(), ref.data(), s.M, s.N, s.K);

        for (const HalfVersion& v : versions) {
            const bool multiple_of_16 = s.M % 16 == 0 && s.N % 16 == 0 && s.K % 16 == 0;
            if (v.needs_16 && !multiple_of_16) continue;
            const ErrorStats e = measure_error(ref, run_half_gemm(v.fn, A, B, s.M, s.N, s.K));
            char msg[200];
            std::snprintf(msg, sizeof(msg), "%s M=%d N=%d K=%d: scaled error %.2e (limit %.0e), non-finite %d", v.name,
                          s.M, s.N, s.K, e.scaled_err, v.tol, e.non_finite);
            expect(e.non_finite == 0 && e.scaled_err <= v.tol, msg);
        }
    }
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing FP16 precision on %s (compute capability %d.%d)\n", info.name.c_str(), info.cc_major,
                info.cc_minor);
    std::printf("FP16 conversions:\n");
    test_conversion();
    if (gpu::wmma_supported()) test_wmma_hand_example();
    test_gemm_shapes();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
