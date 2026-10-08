// =============================================================================
// gpt2_ops.cpp — PyTorch bindings that run GPT-2 decoding on this project's kernels.
//
// Built at run time by llm/run_gpt2.py (torch.utils.cpp_extension.load) together
// with kernels/gemv.cu, layernorm.cu, attention.cu and softmax.cu.
//
// What runs on our kernels, per decoder layer and generated token:
//   LayerNorm            gpu::layernorm_block
//   4 linear layers      gpu::gemv_fp32 / gemv_fp16 / gemv_int8 / gemv_int8_multirow
//   attention            gpu::attention_fused (q_len = 1, causal, KV cache with spare capacity)
//   residual + LayerNorm gpu::add_layernorm (fused)
// What runs on PyTorch (ATen) kernels, as glue: bias adds, GELU, the second residual
// add, writing k and v into the cache, the embedding lookup (in Python).
//
// Streams: our launchers use the default stream, which is also PyTorch's current
// stream unless the caller switches streams, so the two are ordered correctly.
// Explained in docs/14_llm_integration.md.
// =============================================================================

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "attention.cuh"
#include "layernorm.cuh"
#include "quantization.cuh"

namespace {

void check_vector(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32 && t.dim() == 1, name,
                " must be a contiguous 1-D float32 CUDA tensor");
}

}  // namespace

// y = W x (+ bias). W is [out, in] (= N x K, like nn.Linear.weight) in float32, float16
// or int8; for int8, `scale` holds one float32 scale per output row. rows_per_warp
// selects the INT8 kernel: 0 = the original one-row kernel (gemv_int8), 1/2/4/8 =
// gemv_int8_multirow. Pass an empty tensor for `scale` or `bias` when not used.
torch::Tensor linear(const torch::Tensor& w, const torch::Tensor& scale, const torch::Tensor& bias,
                     const torch::Tensor& x, int64_t rows_per_warp) {
    check_vector(x, "x");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && w.size(1) == x.size(0),
                "w must be a contiguous [out, in] CUDA tensor with in = x.size(0)");
    const int N = static_cast<int>(w.size(0));
    const int K = static_cast<int>(w.size(1));
    auto y = torch::empty({N}, x.options());
    switch (w.scalar_type()) {
        case torch::kFloat32:
            gpu::gemv_fp32(w.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), N, K);
            break;
        case torch::kFloat16:
            gpu::gemv_fp16(reinterpret_cast<const __half*>(w.data_ptr<at::Half>()), x.data_ptr<float>(),
                           y.data_ptr<float>(), N, K);
            break;
        case torch::kInt8:
            check_vector(scale, "scale");
            TORCH_CHECK(scale.size(0) == N, "scale must have one entry per output row");
            if (rows_per_warp == 0) {
                gpu::gemv_int8(w.data_ptr<int8_t>(), scale.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(),
                               N, K);
            } else {
                gpu::gemv_int8_multirow(w.data_ptr<int8_t>(), scale.data_ptr<float>(), x.data_ptr<float>(),
                                        y.data_ptr<float>(), N, K, static_cast<int>(rows_per_warp));
            }
            break;
        default:
            TORCH_CHECK(false, "w must be float32, float16 or int8");
    }
    if (bias.numel() > 0) {
        check_vector(bias, "bias");
        y.add_(bias);
    }
    return y;
}

// LayerNorm of one vector (GPT-2: eps = 1e-5).
torch::Tensor layernorm(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta) {
    check_vector(x, "x");
    check_vector(gamma, "gamma");
    check_vector(beta, "beta");
    auto y = torch::empty_like(x);
    gpu::layernorm_block(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), 1,
                         static_cast<int>(x.size(0)));
    return y;
}

// One GPT-2 decoder layer for ONE token at position `pos`. Returns the new residual stream.
// p (16 tensors): ln1_w, ln1_b, attn_w, attn_s, attn_b, aproj_w, aproj_s, aproj_b,
//                 ln2_w, ln2_b, fc_w, fc_s, fc_b, mproj_w, mproj_s, mproj_b
// k_cache, v_cache: [heads][capacity][head_dim] float32; row `pos` of every head is written here.
torch::Tensor layer_decode(const torch::Tensor& x, const std::vector<torch::Tensor>& p, torch::Tensor k_cache,
                           torch::Tensor v_cache, int64_t pos, int64_t n_head, int64_t rows_per_warp) {
    TORCH_CHECK(p.size() == 16, "layer_decode expects 16 parameter tensors");
    check_vector(x, "x");
    const int64_t E = x.size(0);
    const int64_t d = E / n_head;
    TORCH_CHECK(k_cache.dim() == 3 && k_cache.size(0) == n_head && k_cache.size(2) == d && k_cache.is_contiguous() &&
                    v_cache.sizes() == k_cache.sizes() && v_cache.is_contiguous(),
                "k_cache and v_cache must be contiguous [heads, capacity, head_dim]");
    TORCH_CHECK(pos >= 0 && pos < k_cache.size(1), "pos is outside the KV cache capacity");

    // ---- attention block
    const torch::Tensor h1 = layernorm(x, p[0], p[1]);
    const torch::Tensor qkv = linear(p[2], p[3], p[4], h1, rows_per_warp);  // [3E]: q | k | v, each [heads][d]
    k_cache.select(1, pos).copy_(qkv.narrow(0, E, E).view({n_head, d}));
    v_cache.select(1, pos).copy_(qkv.narrow(0, 2 * E, E).view({n_head, d}));
    auto o = torch::empty({E}, x.options());
    gpu::AttentionShape s{static_cast<int>(n_head), 1, static_cast<int>(pos + 1), static_cast<int>(d), true};
    s.kv_capacity = static_cast<int>(k_cache.size(1));
    gpu::attention_fused(qkv.data_ptr<float>(), k_cache.data_ptr<float>(), v_cache.data_ptr<float>(),
                         o.data_ptr<float>(), s);  // q = first E elements of qkv
    const torch::Tensor a = linear(p[5], p[6], p[7], o, rows_per_warp);

    // ---- h = a + x (new residual), y = LayerNorm(h), in one kernel
    auto h = torch::empty_like(x);
    auto y = torch::empty_like(x);
    check_vector(p[8], "ln2_w");
    check_vector(p[9], "ln2_b");
    gpu::add_layernorm(a.data_ptr<float>(), x.data_ptr<float>(), p[8].data_ptr<float>(), p[9].data_ptr<float>(),
                       h.data_ptr<float>(), y.data_ptr<float>(), 1, static_cast<int>(E));

    // ---- MLP block (GPT-2's "gelu_new" is the tanh approximation)
    const torch::Tensor m = at::gelu(linear(p[10], p[11], p[12], y, rows_per_warp), "tanh");
    torch::Tensor out = linear(p[13], p[14], p[15], m, rows_per_warp);
    out.add_(h);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear, "y = W x (+ bias) with FP32, FP16 or INT8 weights (decode GEMV)");
    m.def("layernorm", &layernorm, "LayerNorm of one vector");
    m.def("layer_decode", &layer_decode, "one GPT-2 decoder layer for one token, with KV cache");
}
