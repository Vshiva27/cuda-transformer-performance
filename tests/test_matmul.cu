// =============================================================================
// test_matmul.cu — proves every GEMM version is correct before we time it.
//
// 1. A hand-computed 4x4 example (the same one as docs/04). Small whole numbers
//    are exact in floating point, so the result must match EXACTLY.
// 2. Random matrices of many shapes, compared against cpu::matmul with a
//    tolerance that grows with K (see gemm_tolerance in host_utils.h):
//      - square and non-square (M, N, K all different, so a swapped index fails)
//      - sizes that are not multiples of the block/tile size (bounds checks,
//        zero-padding of partial tiles)
//      - K smaller than one tile, K not a multiple of the tile depth
//      - M = 1: one row times a matrix, the shape of LLM token generation
//      - K = 0: C must be all zeros;  M = 0: nothing to do
// 3. Every version, plus extra configurations (block shapes, tile sizes),
//    because the grid size and the tile loops depend on them.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "matmul.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

static int g_failures = 0;
static int g_checks = 0;

static void check(const std::string& name, const std::vector<float>& ref, const std::vector<float>& out,
                  double tol) {
    ++g_checks;
    CompareResult r = compare_arrays(ref, out, tol, tol);
    if (!r.ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", name.c_str());
        print_compare_failure(r, ref, out);
    }
}

using GemmFn = std::function<void(const float*, const float*, float*, int, int, int)>;

struct TestedGemm {
    std::string name;
    GemmFn fn;
};

// Every version in its default configuration, plus extra configurations.
static std::vector<TestedGemm> all_configs() {
    std::vector<TestedGemm> configs;
    for (const gpu::GemmVersion& v : gpu::gemm_versions()) {
        configs.push_back({v.name, v.launch});
    }
    for (const std::pair<int, int>& block : std::vector<std::pair<int, int>>{{2, 2}, {16, 16}, {8, 4}, {32, 32}, {32, 1}}) {
        // Plain ints (not structured bindings): C++17 lambdas cannot capture structured bindings.
        const int bx = block.first;
        const int by = block.second;
        std::string shape = std::to_string(bx) + "x" + std::to_string(by);
        configs.push_back({"v1 naive block " + shape,
                           [bx, by](const float* A, const float* B, float* C, int M, int N, int K) {
                               gpu::matmul_naive(A, B, C, M, N, K, bx, by);
                           }});
        configs.push_back({"v2 coalesced block " + shape,
                           [bx, by](const float* A, const float* B, float* C, int M, int N, int K) {
                               gpu::matmul_coalesced(A, B, C, M, N, K, bx, by);
                           }});
    }
    for (int tile : {8, 16}) {
        configs.push_back({"v3 tiled-" + std::to_string(tile),
                           [tile](const float* A, const float* B, float* C, int M, int N, int K) {
                               gpu::matmul_tiled(A, B, C, M, N, K, tile);
                           }});
    }
    return configs;
}

// Run one configuration on the GPU and return C on the host.
static std::vector<float> run_gpu(const TestedGemm& g, const std::vector<float>& h_A,
                                  const std::vector<float>& h_B, int M, int N, int K) {
    DeviceBuffer<float> d_A(h_A.size()), d_B(h_B.size()), d_C(static_cast<std::size_t>(M) * N);
    d_A.copy_from_host(h_A);
    d_B.copy_from_host(h_B);
    std::vector<float> h_C(static_cast<std::size_t>(M) * N, -999.0f);  // poison
    d_C.copy_from_host(h_C);
    g.fn(d_A.data(), d_B.data(), d_C.data(), M, N, K);
    d_C.copy_to_host(h_C);
    return h_C;
}

static void test_hand_example(const std::vector<TestedGemm>& configs) {
    // A (4x4)          B (4x4)          C = A x B (computed by hand in docs/04)
    // 1  2  3  4       1 0 2 0           4  6  2  7
    // 5  6  7  8       0 1 0 0          12 14 10 15
    // 9 10 11 12       1 0 0 1          20 22 18 23
    // 13 14 15 16      0 1 0 1          28 30 26 31
    const std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const std::vector<float> B = {1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1};
    const std::vector<float> expected = {4, 6, 2, 7, 12, 14, 10, 15, 20, 22, 18, 23, 28, 30, 26, 31};

    std::vector<float> cpu_C(16);
    cpu::matmul(A.data(), B.data(), cpu_C.data(), 4, 4, 4);
    check("cpu 4x4 hand example", expected, cpu_C, 0.0);

    for (const TestedGemm& g : configs) {
        check(g.name + " 4x4 hand example", expected, run_gpu(g, A, B, 4, 4, 4), 0.0);
    }
}

struct Shape {
    int M, N, K;
};

static void test_random_shapes(const std::vector<TestedGemm>& configs) {
    const std::vector<Shape> shapes = {
        {1, 1, 1},     {4, 4, 4},      {3, 5, 7},      {7, 3, 5},       {32, 32, 32},  {64, 64, 64},
        {33, 31, 17},  {17, 33, 65},   {64, 32, 128},  {65, 63, 9},     {128, 128, 128},
        {255, 257, 129}, {100, 200, 300}, {1, 768, 256}, {1, 300, 1},   {300, 1, 1},   {5, 7, 0},
        {0, 5, 3},
    };

    for (const Shape& s : shapes) {
        std::vector<float> A(static_cast<std::size_t>(s.M) * s.K), B(static_cast<std::size_t>(s.K) * s.N),
            ref(static_cast<std::size_t>(s.M) * s.N);
        fill_random(A, 1000 + s.M * 7 + s.K);
        fill_random(B, 2000 + s.N * 11 + s.K);
        cpu::matmul(A.data(), B.data(), ref.data(), s.M, s.N, s.K);

        for (const TestedGemm& g : configs) {
            std::string name = g.name + " M=" + std::to_string(s.M) + " N=" + std::to_string(s.N) +
                               " K=" + std::to_string(s.K);
            check(name, ref, run_gpu(g, A, B, s.M, s.N, s.K), gemm_tolerance(s.K));
        }
    }
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing GEMM on %s\n", info.name.c_str());

    const std::vector<TestedGemm> configs = all_configs();
    test_hand_example(configs);
    test_random_shapes(configs);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
