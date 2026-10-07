// =============================================================================
// profile_targets.cu — launch each kernel of interest ONCE (by default), at a
// fixed, representative shape, for the profilers.
//
// Why not profile the benchmarks directly? They launch every kernel hundreds of
// times over many shapes. Nsight Compute replays each profiled launch ~10-40
// times to collect its counters, so profiling a benchmark would take very long
// and produce thousands of results. This program produces a short, predictable
// list of launches, and prints it, so profiler result IDs can be matched to
// kernel versions.
//
// Usage:
//   ./profile_targets [case] [--reps N]
//   case: vector_add | gemm | precision | softmax | layernorm | attention | all (default)
//   --reps N: launch each target N times (use > 1 for Nsight Systems timelines)
//
// Every launch is wrapped in an NVTX range named after it (if NVTX is available),
// so it shows up by name on the Nsight Systems timeline.
// Explained in docs/10_nsight_profiling.md.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attention.cuh"
#include "layernorm.cuh"
#include "matmul.cuh"
#include "precision.cuh"
#include "softmax.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "vector_add.cuh"

#ifdef CTP_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#define RANGE_PUSH(name) nvtxRangePushA(name)
#define RANGE_POP() nvtxRangePop()
#else
#define RANGE_PUSH(name) ((void)0)
#define RANGE_POP() ((void)0)
#endif

static int g_reps = 1;
static int g_kernel_index = 0;  // counts kernel launches, like Nsight Compute's result IDs

// Run `fn` (which launches `kernels` kernels) g_reps times, each inside an NVTX
// range, synchronizing after each so ranges match GPU work on the timeline.
template <typename Fn>
static void target(const std::string& label, int kernels, Fn&& fn) {
    for (int r = 0; r < g_reps; ++r) {
        RANGE_PUSH(label.c_str());
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        RANGE_POP();
        const int first = g_kernel_index + 1;
        g_kernel_index += kernels;
        if (kernels == 1) {
            std::printf("  kernel %3d       : %s\n", first, label.c_str());
        } else {
            std::printf("  kernels %3d-%-3d  : %s (%d kernels)\n", first, g_kernel_index, label.c_str(), kernels);
        }
    }
}

// Device buffer filled with random values in [lo, hi).
static DeviceBuffer<float> random_buffer(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> h(n);
    fill_random(h, seed, lo, hi);
    DeviceBuffer<float> d(n);
    d.copy_from_host(h);
    return d;
}

static void case_vector_add() {
    const int n = 1 << 24;
    std::printf("\n[vector_add] n = %d\n", n);
    DeviceBuffer<float> a = random_buffer(n, 1), b = random_buffer(n, 2), c(n);
    target("vector_add naive (block 256)", 1, [&] { gpu::vector_add_naive(a.data(), b.data(), c.data(), n, 256); });
    const int grid = gpu::vector_add_grid_stride_default_grid(256);
    target("vector_add grid-stride (" + std::to_string(grid) + " blocks)", 1,
           [&] { gpu::vector_add_grid_stride(a.data(), b.data(), c.data(), n, 256, grid); });
}

static void case_gemm() {
    const int n = 1024;
    std::printf("\n[gemm] FP32, n = %d\n", n);
    DeviceBuffer<float> A = random_buffer(static_cast<size_t>(n) * n, 3), B = random_buffer(static_cast<size_t>(n) * n, 4),
                        C(static_cast<size_t>(n) * n);
    target("gemm v1 naive (block 32x8)", 1, [&] { gpu::matmul_naive(A.data(), B.data(), C.data(), n, n, n); });
    target("gemm v2 coalesced (block 32x8)", 1, [&] { gpu::matmul_coalesced(A.data(), B.data(), C.data(), n, n, n, 32, 8); });
    target("gemm v2 coalesced (block 32x1)", 1, [&] { gpu::matmul_coalesced(A.data(), B.data(), C.data(), n, n, n, 32, 1); });
    target("gemm v3 tiled-32", 1, [&] { gpu::matmul_tiled(A.data(), B.data(), C.data(), n, n, n, 32); });
    target("gemm v4 register 4x4", 1, [&] { gpu::matmul_register(A.data(), B.data(), C.data(), n, n, n); });
}

static void case_precision() {
    const int n = 1024;
    const size_t nn = static_cast<size_t>(n) * n;
    std::printf("\n[precision] FP16 inputs, n = %d\n", n);
    DeviceBuffer<float> Af = random_buffer(nn, 5), Bf = random_buffer(nn, 6), C(nn);
    DeviceBuffer<__half> A(nn), B(nn);
    gpu::float_to_half(Af.data(), A.data(), static_cast<int>(nn));
    gpu::float_to_half(Bf.data(), B.data(), static_cast<int>(nn));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("  (2 conversion kernels ran before this list starts counting)\n");
    g_kernel_index += 2;
    target("gemm fp16 tiled, fp32 acc", 1, [&] { gpu::matmul_tiled_fp16_acc32(A.data(), B.data(), C.data(), n, n, n); });
    if (gpu::wmma_supported()) {
        target("gemm v5 WMMA tensor cores", 1, [&] { gpu::matmul_wmma(A.data(), B.data(), C.data(), n, n, n); });
    } else {
        std::printf("  (no Tensor Cores: WMMA skipped)\n");
    }
}

static void case_softmax() {
    struct Shape {
        int rows, cols;
    };
    for (Shape s : {Shape{16384, 1024}, Shape{65536, 256}}) {
        const size_t n = static_cast<size_t>(s.rows) * s.cols;
        std::printf("\n[softmax] %d x %d\n", s.rows, s.cols);
        DeviceBuffer<float> x = random_buffer(n, 7, -3.0f, 3.0f), y(n);
        const std::string shape = " " + std::to_string(s.rows) + "x" + std::to_string(s.cols);
        target("softmax v2 block/row" + shape, 1, [&] { gpu::softmax_block(x.data(), y.data(), s.rows, s.cols); });
        target("softmax v3 warp/row" + shape, 1, [&] { gpu::softmax_warp(x.data(), y.data(), s.rows, s.cols); });
        target("softmax v4 online" + shape, 1, [&] { gpu::softmax_online(x.data(), y.data(), s.rows, s.cols); });
    }
}

static void case_layernorm() {
    const int rows = 8192, cols = 4096;
    const size_t n = static_cast<size_t>(rows) * cols;
    std::printf("\n[layernorm] %d x %d\n", rows, cols);
    DeviceBuffer<float> x = random_buffer(n, 8, -2.0f, 2.0f), r = random_buffer(n, 9, -2.0f, 2.0f);
    DeviceBuffer<float> g = random_buffer(cols, 10, 0.5f, 1.5f), b = random_buffer(cols, 11, -0.5f, 0.5f);
    DeviceBuffer<float> h(n), y(n);
    target("layernorm v2 warp/row", 1, [&] { gpu::layernorm_warp(x.data(), g.data(), b.data(), y.data(), rows, cols); });
    target("layernorm v3 block/row regs", 1, [&] { gpu::layernorm_block(x.data(), g.data(), b.data(), y.data(), rows, cols); });
    target("add+layernorm UNFUSED (vector_add, then v3)", 2, [&] {
        gpu::vector_add_naive(x.data(), r.data(), h.data(), static_cast<int>(n), 256);
        gpu::layernorm_block(h.data(), g.data(), b.data(), y.data(), rows, cols);
    });
    target("add+layernorm FUSED", 1,
           [&] { gpu::add_layernorm(x.data(), r.data(), g.data(), b.data(), h.data(), y.data(), rows, cols); });
}

static void case_attention() {
    const int heads = 12, seq = 1024, d = 64;
    std::printf("\n[attention] heads = %d, seq = %d, d = %d\n", heads, seq, d);
    const size_t n = static_cast<size_t>(heads) * seq * d;
    DeviceBuffer<float> Q = random_buffer(n, 12), K = random_buffer(n, 13), V = random_buffer(n, 14), O(n);
    DeviceBuffer<float> scores(static_cast<size_t>(heads) * seq * seq);
    const gpu::AttentionShape full{heads, seq, seq, d, false};
    const gpu::AttentionShape causal{heads, seq, seq, d, true};
    target("attention UNFUSED non-causal (QK^T, softmax, PV)", 3,
           [&] { gpu::attention_unfused(Q.data(), K.data(), V.data(), O.data(), scores.data(), full); });
    target("attention FUSED non-causal", 1, [&] { gpu::attention_fused(Q.data(), K.data(), V.data(), O.data(), full); });
    target("attention FUSED causal", 1, [&] { gpu::attention_fused(Q.data(), K.data(), V.data(), O.data(), causal); });

    const int ctx = 2048;
    std::printf("\n[attention decode] 1 query, %d cached keys\n", ctx);
    const size_t nk = static_cast<size_t>(heads) * ctx * d;
    DeviceBuffer<float> q1 = random_buffer(static_cast<size_t>(heads) * d, 15), Kc = random_buffer(nk, 16),
                        Vc = random_buffer(nk, 17), o1(static_cast<size_t>(heads) * d);
    target("attention FUSED decode (q_len=1, kv_len=2048)", 1,
           [&] { gpu::attention_fused(q1.data(), Kc.data(), Vc.data(), o1.data(), gpu::AttentionShape{heads, 1, ctx, d, true}); });
}

int main(int argc, char** argv) {
    std::string which = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            g_reps = std::atoi(argv[++i]);
        } else {
            which = argv[i];
        }
    }
    GpuInfo info = query_gpu_info(0);
    std::printf("profile_targets on %s, case '%s', %d repetition(s)\n", info.name.c_str(), which.c_str(), g_reps);
    std::printf("Kernel numbers below = Nsight Compute result IDs (when profiling exactly this case).\n");

    bool any = false;
    auto run = [&](const char* name, void (*fn)()) {
        if (which == "all" || which == name) {
            fn();
            any = true;
        }
    };
    run("vector_add", case_vector_add);
    run("gemm", case_gemm);
    run("precision", case_precision);
    run("softmax", case_softmax);
    run("layernorm", case_layernorm);
    run("attention", case_attention);
    if (!any) {
        std::fprintf(stderr, "unknown case '%s'\n", which.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
