// =============================================================================
// test_quantization.cu — INT8 weight quantization and the decode GEMV kernels.
//
// 1. Quantization (CPU): the hand example from docs/08 §9,
//    w = [0.12, -0.5, 0.31, 0.02] -> scale = 0.5/127, q = [30, -127, 79, 5];
//    an all-zero row (scale 0, q 0); per-tensor mode stores one shared scale.
// 2. GEMV, each weight type against an exact (double) product of the weights
//    the kernel actually sees (FP32 W, FP16-rounded W, or q * scale), so only
//    the arithmetic inside the kernel is measured. Tolerance: 2e-5 of the
//    output scale (max |reference|), as for the FP32-accumulating GEMMs in
//    test_precision.cu. An indexing bug produces errors of order 1.
//    Shapes cover N not a multiple of 8 (partial last block) and K whose rows
//    are not a multiple of 16 bytes (the one-weight-per-load fallback path).
//    The INT8 multi-row kernel is checked with 1, 2, 4 and 8 rows per warp, so
//    N not a multiple of the rows per warp (a partly filled last warp) is covered.
// 3. Quantization error bound: each weight is off by at most scale/2, so
//    |y_int8[n] - y_fp32_exact[n]| <= scale[n]/2 * sum_k |x[k]| must hold for
//    every output (plus FP32 rounding of the kernel, covered by the 2e-5 slack).
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "quantization.cuh"
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

static void test_quantize_hand_example() {
    const std::vector<float> w = {0.12f, -0.5f, 0.31f, 0.02f};
    std::vector<std::int8_t> q(4);
    float scale = 0.0f;
    cpu::quantize_int8(w.data(), q.data(), &scale, 1, 4, false);
    const std::int8_t expected[4] = {30, -127, 79, 5};
    bool ok = scale == 0.5f / 127.0f;
    for (int i = 0; i < 4; ++i) ok = ok && q[i] == expected[i];
    char msg[160];
    std::snprintf(msg, sizeof(msg), "quantize [0.12, -0.5, 0.31, 0.02] -> q = [%d, %d, %d, %d], scale = %.7g", q[0],
                  q[1], q[2], q[3], scale);
    std::printf("  %s\n", msg);
    expect(ok, msg);
}

static void test_quantize_special_cases() {
    // Row 0 is all zeros, row 1 has max |w| = 2, row 2 has max |w| = 0.25.
    const std::vector<float> w = {0, 0, 0, 0, 1.0f, -2.0f, 0.5f, 0, 0.25f, 0.1f, -0.2f, 0};
    std::vector<std::int8_t> q(w.size());
    std::vector<float> scale(3);

    cpu::quantize_int8(w.data(), q.data(), scale.data(), 3, 4, false);
    bool zero_ok = scale[0] == 0.0f;
    for (int c = 0; c < 4; ++c) zero_ok = zero_ok && q[c] == 0;
    expect(zero_ok, "all-zero row: scale 0 and q 0");
    expect(q[5] == -127 && q[8] == 127, "the largest |w| of each row maps to +-127 (per-row)");

    cpu::quantize_int8(w.data(), q.data(), scale.data(), 3, 4, true);
    expect(scale[0] == 2.0f / 127.0f && scale[1] == scale[0] && scale[2] == scale[0],
           "per-tensor: every row stores the scale of the whole matrix");
    expect(q[8] == 16, "per-tensor: 0.25 -> round(0.25 / (2/127)) = round(15.875) = 16");
}

static void test_gemv_shapes() {
    struct Shape {
        int N, K;
    };
    const std::vector<Shape> shapes = {{1, 16},     {8, 64},     {13, 1000}, {37, 99},  {33, 45},   {3, 3072},
                                       {768, 768},  {2304, 768}, {100, 4096}, {17, 24}, {64, 4104}};
    const double tol = 2e-5;
    for (const Shape& s : shapes) {
        const size_t n = static_cast<size_t>(s.N) * s.K;
        std::vector<float> W(n), x(s.K);
        fill_random(W, 300 + s.K, -1.0f, 1.0f);
        fill_random(x, 400 + s.N, -1.0f, 1.0f);

        DeviceBuffer<float> d_W(n), d_x(x.size()), d_y(s.N);
        d_W.copy_from_host(W);
        d_x.copy_from_host(x);
        std::vector<float> y(s.N);
        auto check = [&](const char* name, const std::vector<double>& ref) {
            d_y.copy_to_host(y);
            const ErrorStats e = measure_error(ref, y);
            char msg[200];
            std::snprintf(msg, sizeof(msg), "gemv %s N=%d K=%d: scaled error %.2e (limit %.0e), non-finite %d", name,
                          s.N, s.K, e.scaled_err, tol, e.non_finite);
            expect(e.non_finite == 0 && e.scaled_err <= tol, msg);
        };

        // FP32 weights
        std::vector<double> exact(s.N);
        cpu::gemv_f64(W.data(), x.data(), exact.data(), s.N, s.K);
        d_y.copy_from_host(std::vector<float>(s.N, -999.0f));
        gpu::gemv_fp32(d_W.data(), d_x.data(), d_y.data(), s.N, s.K);
        check("fp32", exact);

        // FP16 weights: reference from the FP16-rounded weights
        DeviceBuffer<__half> d_Wh(n);
        gpu::float_to_half(d_W.data(), d_Wh.data(), static_cast<int>(n));
        const std::vector<float> W16 = round_to_half_on_gpu(W);
        std::vector<double> exact16(s.N);
        cpu::gemv_f64(W16.data(), x.data(), exact16.data(), s.N, s.K);
        d_y.copy_from_host(std::vector<float>(s.N, -999.0f));
        gpu::gemv_fp16(d_Wh.data(), d_x.data(), d_y.data(), s.N, s.K);
        check("fp16", exact16);

        // INT8 weights, per-row and per-tensor scales: reference from q and scale
        for (bool per_tensor : {false, true}) {
            std::vector<std::int8_t> q(n);
            std::vector<float> scale(s.N);
            cpu::quantize_int8(W.data(), q.data(), scale.data(), s.N, s.K, per_tensor);
            DeviceBuffer<std::int8_t> d_q(n);
            DeviceBuffer<float> d_scale(s.N);
            d_q.copy_from_host(q);
            d_scale.copy_from_host(scale);
            std::vector<double> exact8(s.N);
            cpu::gemv_int8_f64(q.data(), scale.data(), x.data(), exact8.data(), s.N, s.K);
            d_y.copy_from_host(std::vector<float>(s.N, -999.0f));
            gpu::gemv_int8(d_q.data(), d_scale.data(), d_x.data(), d_y.data(), s.N, s.K);
            check(per_tensor ? "int8 per-tensor" : "int8 per-row", exact8);

            // Multi-row kernel: same weights, same reference, every rows-per-warp setting.
            if (!per_tensor) {
                for (int rows : {1, 2, 4, 8}) {
                    d_y.copy_from_host(std::vector<float>(s.N, -999.0f));
                    gpu::gemv_int8_multirow(d_q.data(), d_scale.data(), d_x.data(), d_y.data(), s.N, s.K, rows);
                    const std::string name = "int8 multirow R=" + std::to_string(rows);
                    check(name.c_str(), exact8);
                }
                d_y.copy_from_host(std::vector<float>(s.N, -999.0f));
                gpu::gemv_int8(d_q.data(), d_scale.data(), d_x.data(), d_y.data(), s.N, s.K);  // y for the bound below
            }

            // Error bound against the ORIGINAL FP32 weights.
            d_y.copy_to_host(y);
            double sum_abs_x = 0.0;
            for (float v : x) sum_abs_x += std::fabs(v);
            double max_abs_ref = 0.0;
            for (double v : exact) max_abs_ref = std::max(max_abs_ref, std::fabs(v));
            int violations = 0;
            for (int r = 0; r < s.N; ++r) {
                const double bound = 0.5 * scale[r] * sum_abs_x + tol * max_abs_ref;
                if (std::fabs(y[r] - exact[r]) > bound) ++violations;
            }
            char msg[200];
            std::snprintf(msg, sizeof(msg), "int8 %s N=%d K=%d: |error| <= scale/2 * sum|x| for every output (%d violations)",
                          per_tensor ? "per-tensor" : "per-row", s.N, s.K, violations);
            expect(violations == 0, msg);
        }
    }
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing INT8 quantization and decode GEMV on %s\n", info.name.c_str());
    test_quantize_hand_example();
    test_quantize_special_cases();
    test_gemv_shapes();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
