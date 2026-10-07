#pragma once
// =============================================================================
// attention.cuh — scaled dot-product attention, unfused and fused.
//
//   O = softmax( Q K^T / sqrt(d) ) V          (per head)
//
// Memory layout (FP32, row-major, heads stored one after another):
//   Q: [heads][q_len][d]    K, V: [heads][kv_len][d]    O: [heads][q_len][d]
//
// q_len and kv_len may differ:
//   prefill (whole prompt at once):      q_len = kv_len = number of tokens
//   decode step with a KV cache:         q_len = 1, kv_len = tokens so far
// Causal masking: query i is the token at position (kv_len - q_len + i) and
// may only attend to keys j <= that position.
// Explained in docs/07_attention.md.
// =============================================================================

namespace gpu {

struct AttentionShape {
    int heads;
    int q_len;
    int kv_len;
    int d;        // head dimension
    bool causal;  // apply the causal mask (requires kv_len >= q_len)
};

// Batched GEMM used by the unfused attention: for each b in [0, batch):
//   C_b = scale * A_b x op(B_b),  op(B) = B (K x N) or B^T (B stored N x K)
// Optionally writes -inf where col > row + causal_offset (the causal mask).
void batched_gemm(const float* d_A, const float* d_B, float* d_C, int batch, int M, int N, int K,
                  bool b_transposed, float scale, bool causal, int causal_offset);

// Unfused: 3 kernels. d_scores is a workspace of heads * q_len * kv_len floats
// (the "seq^2" memory) that the caller must allocate.
void attention_unfused(const float* d_Q, const float* d_K, const float* d_V, float* d_O, float* d_scores,
                       const AttentionShape& s);

// Fused: 1 kernel, online softmax, no score matrix in global memory.
// Supported head dimensions: 32, 64, 128.
void attention_fused(const float* d_Q, const float* d_K, const float* d_V, float* d_O, const AttentionShape& s);

}  // namespace gpu
