// =============================================================================
// test_layernorm.cu — proves every LayerNorm version and the fused
// residual-add + LayerNorm are correct.
//
// 1. Hand example x = [1,2,3,4], gamma = [1,2,1,0.5], beta = [0,0,1,-1]
//    -> y = [-1.3416355, -0.8944236, 1.4472119, -0.3291823] (docs/06 section 3).
// 2. Constant rows (variance 0): y must equal beta — eps prevents 0/0.
// 3. Random shapes (cols around 32, block capacities 1024/4096/16384,
//    real hidden sizes 768/4096) and three value ranges:
//      [-2, 2]      typical activations
//      [50, 52]     large mean, small spread: hard for float sums
//      [-300, 300]  large magnitude
// 4. Fused add + LayerNorm: h must equal x + residual EXACTLY, y must match
//    cpu::add_layernorm.
//
// Tolerances come from a CPU emulation of each kernel's float summation order
// (docs/06 section 9): worst error ~5e-6 for centered data, ~4e-4 for the
// sequential v1 on the [50, 52] range.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "layernorm.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

static int g_failures = 0;
static int g_checks = 0;

static void check(const std::string& name, const std::vector<float>& ref, const std::vector<float>& out,
                  double atol, double rtol) {
    ++g_checks;
    CompareResult r = compare_arrays(ref, out, atol, rtol);
    if (!r.ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", name.c_str());
        print_compare_failure(r, ref, out);
    }
}

struct LayerNormData {
    int rows, cols;
    std::vector<float> x, gamma, beta;
};

static std::vector<float> run_gpu(const gpu::LayerNormVersion& v, const LayerNormData& d) {
    DeviceBuffer<float> d_x(d.x.size()), d_g(d.gamma.size()), d_b(d.beta.size()), d_y(d.x.size());
    d_x.copy_from_host(d.x);
    d_g.copy_from_host(d.gamma);
    d_b.copy_from_host(d.beta);
    std::vector<float> y(d.x.size(), -999.0f);
    d_y.copy_from_host(y);
    v.launch(d_x.data(), d_g.data(), d_b.data(), d_y.data(), d.rows, d.cols);
    d_y.copy_to_host(y);
    return y;
}

// Demonstration only: the one-pass variance formula in float.
static void print_one_pass_demo() {
    std::vector<float> v(1024);
    fill_random(v, 5, 999.5f, 1000.5f);
    float sum = 0.0f, sum_sq = 0.0f;
    for (float a : v) {
        sum += a;
        sum_sq += a * a;
    }
    const float mean = sum / v.size();
    const float one_pass = sum_sq / v.size() - mean * mean;
    float two_pass = 0.0f;
    for (float a : v) two_pass += (a - mean) * (a - mean);
    two_pass /= v.size();
    std::printf("Demonstration: variance of 1024 floats in [999.5, 1000.5] (true value ~0.0862)\n");
    std::printf("               two-pass  mean((x-mean)^2)  = %.6f\n", two_pass);
    std::printf("               one-pass  mean(x^2)-mean^2  = %.6f   <- catastrophic cancellation\n", one_pass);
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing LayerNorm on %s\n", info.name.c_str());
    print_one_pass_demo();
    const auto& versions = gpu::layernorm_versions();
    const float eps = gpu::LAYERNORM_EPS;

    // ---- 1. Hand example ---------------------------------------------------------
    {
        LayerNormData d{1, 4, {1, 2, 3, 4}, {1, 2, 1, 0.5f}, {0, 0, 1, -1}};
        const std::vector<float> expected = {-1.3416355f, -0.8944236f, 1.4472119f, -0.3291823f};
        std::vector<float> cpu_y(4);
        cpu::layernorm(d.x.data(), d.gamma.data(), d.beta.data(), cpu_y.data(), 1, 4, eps);
        check("cpu hand example", expected, cpu_y, 1e-6, 1e-6);
        for (const auto& v : versions) check(std::string(v.name) + " hand example", expected, run_gpu(v, d), 1e-6, 1e-6);
    }

    // ---- 2. Constant rows: variance 0, y = beta ---------------------------------
    for (int cols : {1, 100, 2000}) {
        LayerNormData d{3, cols, std::vector<float>(3 * cols, 7.0f), std::vector<float>(cols), std::vector<float>(cols)};
        fill_random(d.gamma, 1, 0.5f, 1.5f);
        fill_random(d.beta, 2, -0.5f, 0.5f);
        std::vector<float> expected;
        for (int r = 0; r < 3; ++r) expected.insert(expected.end(), d.beta.begin(), d.beta.end());
        for (const auto& v : versions) {
            check(std::string(v.name) + " constant row cols=" + std::to_string(cols), expected, run_gpu(v, d), 1e-6,
                  0.0);
        }
    }

    // ---- 3. Random shapes and ranges --------------------------------------------
    struct Shape {
        int rows, cols;
    };
    const std::vector<Shape> shapes = {{1, 4},    {3, 31},   {5, 32},   {2, 33},   {4, 768},  {7, 1000}, {3, 1024},
                                       {2, 1025}, {5, 4096}, {2, 4097}, {3, 8192}, {2, 16384}, {300, 64}};
    struct Range {
        const char* label;
        float lo, hi;
        double atol;
    };
    const std::vector<Range> ranges = {{"[-2,2]", -2.0f, 2.0f, 5e-5}, {"[50,52]", 50.0f, 52.0f, 2e-3},
                                       {"[-300,300]", -300.0f, 300.0f, 5e-5}};

    for (const Shape& s : shapes) {
        for (const Range& rg : ranges) {
            LayerNormData d{s.rows, s.cols, std::vector<float>(static_cast<size_t>(s.rows) * s.cols),
                            std::vector<float>(s.cols), std::vector<float>(s.cols)};
            fill_random(d.x, 7 * s.cols + s.rows, rg.lo, rg.hi);
            fill_random(d.gamma, 11, 0.5f, 1.5f);
            fill_random(d.beta, 13, -0.5f, 0.5f);
            std::vector<float> ref(d.x.size());
            cpu::layernorm(d.x.data(), d.gamma.data(), d.beta.data(), ref.data(), s.rows, s.cols, eps);
            for (const auto& v : versions) {
                const std::string name = std::string(v.name) + " " + std::to_string(s.rows) + "x" +
                                         std::to_string(s.cols) + " range " + rg.label;
                check(name, ref, run_gpu(v, d), rg.atol, 1e-5);
            }
        }
    }

    // ---- 4. Fused residual add + LayerNorm ---------------------------------------
    for (const Shape& s : shapes) {
        const size_t n = static_cast<size_t>(s.rows) * s.cols;
        std::vector<float> x(n), res(n), gamma(s.cols), beta(s.cols), h_ref(n), y_ref(n);
        fill_random(x, 21 + s.cols, -2.0f, 2.0f);
        fill_random(res, 22 + s.cols, -2.0f, 2.0f);
        fill_random(gamma, 23, 0.5f, 1.5f);
        fill_random(beta, 24, -0.5f, 0.5f);
        cpu::add_layernorm(x.data(), res.data(), gamma.data(), beta.data(), h_ref.data(), y_ref.data(), s.rows, s.cols,
                           eps);

        DeviceBuffer<float> d_x(n), d_r(n), d_g(s.cols), d_b(s.cols), d_h(n), d_y(n);
        d_x.copy_from_host(x);
        d_r.copy_from_host(res);
        d_g.copy_from_host(gamma);
        d_b.copy_from_host(beta);
        std::vector<float> h(n, -999.0f), y(n, -999.0f);
        d_h.copy_from_host(h);
        d_y.copy_from_host(y);
        gpu::add_layernorm(d_x.data(), d_r.data(), d_g.data(), d_b.data(), d_h.data(), d_y.data(), s.rows, s.cols);
        d_h.copy_to_host(h);
        d_y.copy_to_host(y);

        const std::string tag = std::to_string(s.rows) + "x" + std::to_string(s.cols);
        check("fused add_layernorm h " + tag, h_ref, h, 0.0, 0.0);  // a single float add: bit-exact
        check("fused add_layernorm y " + tag, y_ref, y, 5e-5, 1e-5);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
