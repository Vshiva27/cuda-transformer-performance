// =============================================================================
// test_softmax.cu — proves every softmax version is correct.
//
// 1. Hand example: softmax([1, 2, 3]) = [0.0900306, 0.2447285, 0.6652410].
// 2. Stability: softmax([1000, 1001, 1002]) must give the SAME answer (shifting
//    a row by a constant does not change softmax). Also shows, on the CPU, that
//    the naive formula without max-subtraction produces NaN for these inputs.
// 3. All-equal row: every output must be 1/cols.
// 4. Random rows of many shapes (cols below / at / above 32 and the block
//    size, very long rows) and value ranges (small, and very spread out),
//    compared with cpu::softmax (computed in double precision).
// 5. Every output row must sum to 1.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "softmax.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

static int g_failures = 0;
static int g_checks = 0;

static void fail(const std::string& msg) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
}

// Tolerance: each output carries the rounding error of a sum over `cols`
// terms (plus ~1-2 ulp from expf), so the allowed relative error grows with
// cols. An indexing or reduction bug produces errors of 10% or more.
static double softmax_rtol(int cols) {
    return 1e-5 + 1e-7 * cols;
}
static constexpr double SOFTMAX_ATOL = 1e-7;  // for outputs that are essentially 0

using SoftmaxFn = std::function<void(const float*, float*, int, int)>;
struct TestedSoftmax {
    std::string name;
    SoftmaxFn fn;
};

static std::vector<TestedSoftmax> all_configs() {
    std::vector<TestedSoftmax> configs;
    for (const gpu::SoftmaxVersion& v : gpu::softmax_versions()) configs.push_back({v.name, v.launch});
    for (int bs : {32, 64, 128, 512, 1024}) {
        configs.push_back({"v2 block/row bs=" + std::to_string(bs),
                           [bs](const float* x, float* y, int r, int c) { gpu::softmax_block(x, y, r, c, bs); }});
    }
    return configs;
}

static std::vector<float> run_gpu(const TestedSoftmax& s, const std::vector<float>& h_x, int rows, int cols) {
    DeviceBuffer<float> d_x(h_x.size()), d_y(h_x.size());
    d_x.copy_from_host(h_x);
    std::vector<float> h_y(h_x.size(), -999.0f);  // poison
    d_y.copy_from_host(h_y);
    s.fn(d_x.data(), d_y.data(), rows, cols);
    d_y.copy_to_host(h_y);
    return h_y;
}

static void check_case(const TestedSoftmax& s, const std::string& label, const std::vector<float>& x,
                       const std::vector<float>& ref, int rows, int cols) {
    ++g_checks;
    std::vector<float> out = run_gpu(s, x, rows, cols);
    const std::string name = s.name + " " + label + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ")";

    CompareResult r = compare_arrays(ref, out, SOFTMAX_ATOL, softmax_rtol(cols));
    if (!r.ok) {
        fail(name);
        print_compare_failure(r, ref, out);
        return;
    }
    // Each row must sum to 1 (checked in double precision).
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += out[static_cast<size_t>(row) * cols + c];
        if (std::fabs(sum - 1.0) > 1e-4) {
            fail(name + ": row " + std::to_string(row) + " sums to " + std::to_string(sum));
            return;
        }
    }
}

// The formula WITHOUT max subtraction, only to demonstrate why we subtract.
static std::vector<float> unstable_softmax_row(const std::vector<float>& x) {
    float sum = 0.0f;
    for (float v : x) sum += std::exp(v);  // exp(1000) overflows float -> inf
    std::vector<float> y;
    for (float v : x) y.push_back(std::exp(v) / sum);  // inf / inf = NaN
    return y;
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing softmax on %s\n", info.name.c_str());
    const std::vector<TestedSoftmax> configs = all_configs();

    // ---- 1 & 2: hand example and shifted copy -----------------------------------
    const std::vector<float> expected = {0.09003057f, 0.24472847f, 0.66524096f};
    std::vector<float> cpu_out(3);
    const std::vector<float> small = {1.0f, 2.0f, 3.0f};
    const std::vector<float> large = {1000.0f, 1001.0f, 1002.0f};
    cpu::softmax(small.data(), cpu_out.data(), 1, 3);
    if (!compare_arrays(expected, cpu_out, 1e-7, 1e-6).ok) fail("cpu softmax hand example");

    std::vector<float> unstable = unstable_softmax_row(large);
    std::printf("Demonstration: softmax([1000,1001,1002]) WITHOUT max subtraction = [%g, %g, %g]\n",
                unstable[0], unstable[1], unstable[2]);
    std::printf("               WITH max subtraction should be [%.7f, %.7f, %.7f]\n", expected[0], expected[1],
                expected[2]);

    for (const TestedSoftmax& s : configs) {
        check_case(s, "hand [1,2,3]", small, expected, 1, 3);
        check_case(s, "shifted [1000,1001,1002]", large, expected, 1, 3);
    }

    // ---- 3: all-equal rows -> uniform distribution ------------------------------
    {
        const int rows = 3, cols = 100;
        std::vector<float> x(rows * cols, 7.0f), ref(rows * cols, 1.0f / cols);
        for (const TestedSoftmax& s : configs) check_case(s, "all-equal", x, ref, rows, cols);
    }

    // ---- 4 & 5: random shapes and value ranges ----------------------------------
    struct Shape {
        int rows, cols;
    };
    const std::vector<Shape> shapes = {{1, 1},   {1, 5},     {3, 31},    {7, 32},     {5, 33},     {4, 255},
                                       {4, 256}, {4, 257},   {9, 1000},  {64, 128},   {17, 2049},  {33, 4096},
                                       {2, 50257}, {300, 64}};
    struct Range {
        const char* label;
        float lo, hi;
    };
    const std::vector<Range> ranges = {{"values in [-3,3]", -3.0f, 3.0f}, {"values in [-60,90]", -60.0f, 90.0f}};

    for (const Shape& sh : shapes) {
        for (const Range& rg : ranges) {
            std::vector<float> x(static_cast<size_t>(sh.rows) * sh.cols), ref(x.size());
            fill_random(x, 31 * sh.rows + sh.cols, rg.lo, rg.hi);
            cpu::softmax(x.data(), ref.data(), sh.rows, sh.cols);
            for (const TestedSoftmax& s : configs) check_case(s, rg.label, x, ref, sh.rows, sh.cols);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
