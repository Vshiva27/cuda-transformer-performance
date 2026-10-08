// =============================================================================
// test_attention.cu — proves both attention implementations are correct.
//
// 1. Hand example (docs/07 section 3): Q = K = identity (2x2), V = [[1,2],[3,4]], d = 2
//      non-causal: O = [[1.6604769, 2.6604769], [2.3395231, 3.3395231]]
//      causal:     O = [[1, 2],                 [2.3395231, 3.3395231]]
//    (unfused only: the fused kernel needs d in {32, 64, 128})
// 2. Random cases against cpu::attention (double precision), both versions:
//    one token, q_len != kv_len, causal with an offset (kv_len > q_len),
//    a decode step (q_len = 1 against 513 cached keys), q_len not a multiple
//    of 8 (partially filled blocks), kv_len not a multiple of 32 (partial key
//    tiles), all supported head dims.
// 3. A KV cache with spare capacity (kv_capacity > kv_len, unused rows NaN):
//    must match the packed layout bit for bit.
// A CPU emulation of both kernels' exact logic passed these cases with a worst
// error of 1.2e-7 (docs/07 section 9); the tolerance 1e-4 leaves a wide margin.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "attention.cuh"
#include "cpu/cpu_ops.h"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"

static int g_failures = 0;
static int g_checks = 0;

static void check(const std::string& name, const std::vector<float>& ref, const std::vector<float>& out, double tol) {
    ++g_checks;
    CompareResult r = compare_arrays(ref, out, tol, tol);
    if (!r.ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", name.c_str());
        print_compare_failure(r, ref, out);
    }
}

// Runs one implementation on the GPU; returns O.
static std::vector<float> run_gpu(bool fused, const std::vector<float>& Q, const std::vector<float>& K,
                                  const std::vector<float>& V, const gpu::AttentionShape& s) {
    DeviceBuffer<float> d_Q(Q.size()), d_K(K.size()), d_V(V.size()), d_O(Q.size());
    DeviceBuffer<float> d_scores(fused ? 0 : static_cast<size_t>(s.heads) * s.q_len * s.kv_len);
    d_Q.copy_from_host(Q);
    d_K.copy_from_host(K);
    d_V.copy_from_host(V);
    std::vector<float> O(Q.size(), -999.0f);
    d_O.copy_from_host(O);
    if (fused) {
        gpu::attention_fused(d_Q.data(), d_K.data(), d_V.data(), d_O.data(), s);
    } else {
        gpu::attention_unfused(d_Q.data(), d_K.data(), d_V.data(), d_O.data(), d_scores.data(), s);
    }
    d_O.copy_to_host(O);
    return O;
}

static std::string describe(const gpu::AttentionShape& s) {
    return "heads=" + std::to_string(s.heads) + " q_len=" + std::to_string(s.q_len) + " kv_len=" +
           std::to_string(s.kv_len) + " d=" + std::to_string(s.d) + (s.causal ? " causal" : "");
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing attention on %s\n", info.name.c_str());

    // ---- 1. Hand example -----------------------------------------------------------
    {
        const std::vector<float> Q = {1, 0, 0, 1}, K = {1, 0, 0, 1}, V = {1, 2, 3, 4};
        const std::vector<float> expected = {1.6604769f, 2.6604769f, 2.3395231f, 3.3395231f};
        const std::vector<float> expected_causal = {1.0f, 2.0f, 2.3395231f, 3.3395231f};
        std::vector<float> cpu_O(4);
        cpu::attention(Q.data(), K.data(), V.data(), cpu_O.data(), 1, 2, 2, 2, false);
        check("cpu hand example", expected, cpu_O, 1e-6);
        check("unfused hand example", expected, run_gpu(false, Q, K, V, {1, 2, 2, 2, false}), 1e-5);
        check("unfused hand example causal", expected_causal, run_gpu(false, Q, K, V, {1, 2, 2, 2, true}), 1e-5);
    }

    // ---- 2. Random cases -----------------------------------------------------------
    const std::vector<gpu::AttentionShape> cases = {
        {1, 1, 1, 32, false},   {2, 5, 7, 64, false},     {3, 33, 33, 64, true},  {4, 100, 300, 32, true},
        {2, 1, 513, 128, true}, {2, 64, 64, 128, true},   {1, 17, 1000, 64, false}, {2, 40, 40, 64, false},
        {1, 9, 9, 32, true},    {12, 128, 128, 64, true}, {12, 128, 128, 64, false},
    };
    for (const gpu::AttentionShape& s : cases) {
        const size_t nq = static_cast<size_t>(s.heads) * s.q_len * s.d;
        const size_t nk = static_cast<size_t>(s.heads) * s.kv_len * s.d;
        std::vector<float> Q(nq), K(nk), V(nk), ref(nq);
        fill_random(Q, 1 + s.q_len, -1.0f, 1.0f);
        fill_random(K, 2 + s.kv_len, -1.0f, 1.0f);
        fill_random(V, 3 + s.d, -1.0f, 1.0f);
        cpu::attention(Q.data(), K.data(), V.data(), ref.data(), s.heads, s.q_len, s.kv_len, s.d, s.causal);
        check("unfused " + describe(s), ref, run_gpu(false, Q, K, V, s), 1e-4);
        check("fused   " + describe(s), ref, run_gpu(true, Q, K, V, s), 1e-4);
    }

    // ---- 3. KV cache with spare capacity (fused only) ---------------------------------
    // K and V stored [heads][capacity][d] with only the first kv_len rows per head filled,
    // the rest NaN: any read past kv_len would poison the output. The result must be
    // bit-identical to the packed layout (same arithmetic, only the addresses differ).
    for (const gpu::AttentionShape& packed :
         {gpu::AttentionShape{12, 1, 300, 64, true}, gpu::AttentionShape{3, 5, 37, 32, true}}) {
        const int capacity = packed.kv_len + 45;
        const size_t nq = static_cast<size_t>(packed.heads) * packed.q_len * packed.d;
        const size_t nk = static_cast<size_t>(packed.heads) * packed.kv_len * packed.d;
        std::vector<float> Q(nq), K(nk), V(nk);
        fill_random(Q, 7, -1.0f, 1.0f);
        fill_random(K, 8, -1.0f, 1.0f);
        fill_random(V, 9, -1.0f, 1.0f);
        const size_t nk_cap = static_cast<size_t>(packed.heads) * capacity * packed.d;
        std::vector<float> Kc(nk_cap, NAN), Vc(nk_cap, NAN);
        for (int h = 0; h < packed.heads; ++h) {
            for (size_t i = 0; i < static_cast<size_t>(packed.kv_len) * packed.d; ++i) {
                Kc[static_cast<size_t>(h) * capacity * packed.d + i] = K[static_cast<size_t>(h) * packed.kv_len * packed.d + i];
                Vc[static_cast<size_t>(h) * capacity * packed.d + i] = V[static_cast<size_t>(h) * packed.kv_len * packed.d + i];
            }
        }
        gpu::AttentionShape with_capacity = packed;
        with_capacity.kv_capacity = capacity;
        check("fused kv_capacity=" + std::to_string(capacity) + " " + describe(packed), run_gpu(true, Q, K, V, packed),
              run_gpu(true, Q, Kc, Vc, with_capacity), 0.0);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
