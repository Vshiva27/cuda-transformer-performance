// =============================================================================
// bench_attention.cu — attention experiments.
//
// Experiment A (prefill): heads = 12, d = 64 (GPT-2 small), seq = 128 .. 2048,
//   non-causal (same shapes as python/benchmark.py) and causal.
//   Unfused: total time AND the time of each of its 3 kernels, plus the
//   workspace it needs (the heads x seq x seq score matrix).
//   Fused: one kernel, no workspace.
// Experiment B (KV cache): generating ONE new token with a context of L tokens.
//   with cache   : 1 query attends to L cached keys/values      (q_len = 1)
//   without cache: recompute attention for all L tokens          (q_len = L)
//   (Without a cache, the K/V projection GEMMs for all L tokens would also
//   have to be recomputed; that cost is not included here, so the real gap is
//   even larger.) Also prints the KV-cache memory per token for GPT-2 small.
//
// GFLOP/s counts the two matmuls: 4 * heads * q_len * kv_len * d (non-causal).
//
// Usage:  ./bench_attention
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attention.cuh"
#include <string>

#include "benchmark/bench_utils.cuh"
#include "benchmark/csv_log.h"
#include "cpu/cpu_ops.h"
#include "softmax.cuh"
#include "utils/cuda_check.cuh"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

struct AttentionProblem {
    gpu::AttentionShape s;
    std::vector<float> h_Q, h_K, h_V;
    DeviceBuffer<float> d_Q, d_K, d_V, d_O, d_scores;

    explicit AttentionProblem(const gpu::AttentionShape& shape)
        : s(shape),
          h_Q(static_cast<size_t>(shape.heads) * shape.q_len * shape.d),
          h_K(static_cast<size_t>(shape.heads) * shape.kv_len * shape.d),
          h_V(h_K.size()),
          d_Q(h_Q.size()), d_K(h_K.size()), d_V(h_V.size()), d_O(h_Q.size()),
          d_scores(static_cast<size_t>(shape.heads) * shape.q_len * shape.kv_len) {
        fill_random(h_Q, 1, -1.0f, 1.0f);
        fill_random(h_K, 2, -1.0f, 1.0f);
        fill_random(h_V, 3, -1.0f, 1.0f);
        d_Q.copy_from_host(h_Q);
        d_K.copy_from_host(h_K);
        d_V.copy_from_host(h_V);
    }
    void unfused() { gpu::attention_unfused(d_Q.data(), d_K.data(), d_V.data(), d_O.data(), d_scores.data(), s); }
    void fused() { gpu::attention_fused(d_Q.data(), d_K.data(), d_V.data(), d_O.data(), s); }
    std::vector<float> download() {
        std::vector<float> O(h_Q.size());
        d_O.copy_to_host(O);
        return O;
    }
};

static void verify(const char* what, AttentionProblem& p, const std::vector<float>& ref) {
    const std::vector<float> out = p.download();
    CompareResult r = compare_arrays(ref, out, 1e-4, 1e-4);
    if (!r.ok) {
        std::fprintf(stderr, "%s attention gave a WRONG result\n", what);
        print_compare_failure(r, ref, out);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    GpuInfo info = query_gpu_info(0);
    print_gpu_info(info);
    CsvLog log(argc, argv, "attention", info.name);
    const int heads = 12, d = 64;

    // ------------------------------------------------------------------------
    // Experiment A: prefill
    // ------------------------------------------------------------------------
    std::printf("Experiment A: prefill attention, heads = %d, d = %d, FP32\n", heads, d);
    for (bool causal : {false, true}) {
        std::printf("\n%s\n", causal ? "CAUSAL (decoder models)" : "NON-CAUSAL (same shapes as python/benchmark.py)");
        std::printf("  %6s | %10s | %9s %9s %9s | %10s | %9s | %12s | %8s\n", "seq", "unfused ms", "QK^T",
                    "softmax", "PV", "fused ms", "speedup", "workspace MB", "CPU ms");
        for (int seq : {128, 512, 1024, 2048}) {
            const gpu::AttentionShape s{heads, seq, seq, d, causal};
            const size_t workspace = static_cast<size_t>(heads) * seq * seq * sizeof(float);
            if (workspace * 2 > info.free_mem_bytes / 2) {
                std::printf("  %6d | skipped: not enough GPU memory\n", seq);
                continue;
            }
            AttentionProblem p(s);

            // Correctness against the CPU for the sizes where the CPU is fast enough.
            double cpu_ms = -1.0;
            if (seq <= 1024) {
                std::vector<float> ref(p.h_Q.size());
                cpu_ms = time_cpu_ms(
                    [&] { cpu::attention(p.h_Q.data(), p.h_K.data(), p.h_V.data(), ref.data(), heads, seq, seq, d, causal); },
                    0, 1);
                p.unfused();
                verify("unfused", p, ref);
                p.fused();
                verify("fused", p, ref);
            }

            const int iters = seq >= 1024 ? 10 : 50;
            const float unfused_ms = time_gpu_ms([&] { p.unfused(); }, 2, iters);
            const float fused_ms = time_gpu_ms([&] { p.fused(); }, 2, iters);

            // Breakdown of the unfused pipeline: time each kernel on its own.
            const float scale = 1.0f / 8.0f;  // 1/sqrt(64)
            const float qk_ms = time_gpu_ms(
                [&] { gpu::batched_gemm(p.d_Q.data(), p.d_K.data(), p.d_scores.data(), heads, seq, seq, d, true, scale, causal, 0); },
                2, iters);
            const float sm_ms =
                time_gpu_ms([&] { gpu::softmax_online(p.d_scores.data(), p.d_scores.data(), heads * seq, seq); }, 2, iters);
            const float pv_ms = time_gpu_ms(
                [&] { gpu::batched_gemm(p.d_scores.data(), p.d_V.data(), p.d_O.data(), heads, seq, d, seq, false, 1.0f, false, 0); },
                2, iters);

            char cpu_col[32] = "-";
            if (cpu_ms >= 0) std::snprintf(cpu_col, sizeof(cpu_col), "%.1f", cpu_ms);
            std::printf("  %6d | %10.4f | %9.4f %9.4f %9.4f | %10.4f | %8.2fx | %12.1f | %8s\n", seq, unfused_ms, qk_ms,
                        sm_ms, pv_ms, fused_ms, unfused_ms / fused_ms, workspace / 1048576.0, cpu_col);

            // Same label format as python/benchmark.py, so the summary can line them up.
            const std::string shape = "h=" + std::to_string(heads) + " seq=" + std::to_string(seq) + " d=" + std::to_string(d);
            const std::string experiment = causal ? "A prefill causal" : "A prefill";
            const double flops = 4.0 * heads * seq * seq * d;  // non-causal count, for comparability
            char ws_note[64];
            std::snprintf(ws_note, sizeof(ws_note), "workspace_MB=%.1f", workspace / 1048576.0);
            log.add(experiment, "unfused (3 kernels)", shape, "fp32", unfused_ms, flops / (unfused_ms * 1e6), -1, ws_note);
            log.add(experiment, "  unfused: QK^T", shape, "fp32", qk_ms);
            log.add(experiment, "  unfused: softmax", shape, "fp32", sm_ms);
            log.add(experiment, "  unfused: PV", shape, "fp32", pv_ms);
            log.add(experiment, "fused (1 kernel)", shape, "fp32", fused_ms, flops / (fused_ms * 1e6), -1, "workspace_MB=0");
            if (cpu_ms >= 0) log.add(experiment, "cpu (double, 1 thread)", shape, "fp32", cpu_ms);
        }
    }

    // ------------------------------------------------------------------------
    // Experiment B: KV cache
    // ------------------------------------------------------------------------
    std::printf("\n\nExperiment B: one decoding step (generate 1 token) with a context of L tokens, fused kernel\n");
    std::printf("  %6s | %18s | %20s | %10s\n", "L", "with KV cache (ms)", "recompute all (ms)", "ratio");
    for (int L : {128, 512, 1024, 2048}) {
        AttentionProblem with_cache(gpu::AttentionShape{heads, 1, L, d, true});
        AttentionProblem without_cache(gpu::AttentionShape{heads, L, L, d, true});
        std::vector<float> ref(with_cache.h_Q.size());
        cpu::attention(with_cache.h_Q.data(), with_cache.h_K.data(), with_cache.h_V.data(), ref.data(), heads, 1, L, d,
                       true);
        with_cache.fused();
        verify("decode", with_cache, ref);

        const float cached_ms = time_gpu_ms([&] { with_cache.fused(); }, 5, 200);
        const float recompute_ms = time_gpu_ms([&] { without_cache.fused(); }, 2, 20);
        std::printf("  %6d | %18.4f | %20.4f | %9.1fx\n", L, cached_ms, recompute_ms, recompute_ms / cached_ms);
        const std::string ctx = "context=" + std::to_string(L);
        log.add("B kv cache", "decode with KV cache (q_len=1)", ctx, "fp32", cached_ms);
        log.add("B kv cache", "recompute all (q_len=L)", ctx, "fp32", recompute_ms);
    }

    // KV cache size: every layer stores one K row and one V row (heads * d values each) per token.
    const int layers = 12, hidden = heads * d;
    const double bytes_per_token_fp16 = 2.0 /*K and V*/ * layers * hidden * 2 /*bytes per FP16*/;
    std::printf("\nKV cache size, GPT-2 small (12 layers, hidden 768), FP16:\n");
    std::printf("  per token: 2 x %d layers x %d x 2 bytes = %.0f KB\n", layers, hidden, bytes_per_token_fp16 / 1024);
    for (int L : {1024, 2048}) {
        std::printf("  context %4d tokens: %.1f MB per sequence\n", L, bytes_per_token_fp16 * L / 1048576.0);
    }
    return EXIT_SUCCESS;
}
