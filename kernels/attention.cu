// =============================================================================
// attention.cu — attention built from our own kernels (unfused), and a fused
// FlashAttention-style kernel.
//
// Unfused (what eager PyTorch's "naive" attention does):
//   1. S = Q K^T * scale           batched GEMM with B transposed  -> heads x q_len x kv_len in global memory
//   2. P = softmax(S) row-wise     softmax v4 (online), in place
//   3. O = P V                     batched GEMM
//
// Fused (one kernel): each warp owns one query row. The block walks over the
// keys in tiles of 32, staged in shared memory and shared by the block's 8
// warps. For every key the warp computes the score, then updates a running
// max m, running sum l and running output o with the online-softmax rescaling
// from docs/05 section 10. S and P never exist in global memory.
//
// Explained line by line in docs/07_attention.md.
// =============================================================================

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "attention.cuh"
#include "softmax.cuh"
#include "utils/cuda_check.cuh"
#include "warp_reduce.cuh"

// -----------------------------------------------------------------------------
// Batched tiled GEMM (GEMM v3 + a batch index + optional transposed B + scale + mask)
// -----------------------------------------------------------------------------
template <int TILE, bool B_TRANSPOSED>
__global__ void batched_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float scale,
                                    bool causal, int causal_offset) {
    __shared__ float As[TILE][TILE];
    // +1 column of padding: the transposed store below writes Bs[tx][ty], i.e.
    // a COLUMN of the tile per warp. Without padding all 32 addresses would be
    // in the same shared-memory bank (32-way conflict). docs/07 section 6.
    __shared__ float Bs[TILE][TILE + 1];

    // blockIdx.z selects the batch entry (here: the attention head).
    const size_t batch = blockIdx.z;
    A += batch * M * K;
    B += batch * K * N;  // K*N elements whether stored as K x N or N x K
    C += batch * M * N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        if constexpr (B_TRANSPOSED) {
            // B is stored N x K (for attention: the K matrix, kv_len x d).
            // We need Bs[k][n]. To keep the GLOBAL read coalesced, consecutive
            // threads (tx) read consecutive k of one stored row n, and the
            // tile is transposed while being written to shared memory.
            const int n = blockIdx.x * TILE + ty;
            const int k = t * TILE + tx;
            Bs[tx][ty] = (n < N && k < K) ? B[n * K + k] : 0.0f;
        } else {
            const int b_row = t * TILE + ty;
            Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = sum * scale;
        if (causal && col > row + causal_offset) v = -INFINITY;  // key is in the query's future
        C[row * N + col] = v;
    }
}

// -----------------------------------------------------------------------------
// Fused attention
// -----------------------------------------------------------------------------
namespace {
constexpr int FUSED_WARPS = 8;  // query rows per block
constexpr int KEY_TILE = 32;    // keys per shared-memory tile
}  // namespace

template <int D>
__global__ void attention_fused_kernel(const float* Q, const float* K, const float* V, float* O, int q_len,
                                       int kv_len, float scale, bool causal, int causal_offset) {
    constexpr int PER_LANE = D / 32;  // elements of a d-vector held by each lane
    __shared__ float Ks[KEY_TILE][D];
    __shared__ float Vs[KEY_TILE][D];

    const int head = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int q_row = blockIdx.x * FUSED_WARPS + warp_id;
    // Inactive warps (past the last query) must NOT return early: they still
    // help load tiles and must reach every __syncthreads().
    const bool active = q_row < q_len;

    Q += static_cast<size_t>(head) * q_len * D;
    K += static_cast<size_t>(head) * kv_len * D;
    V += static_cast<size_t>(head) * kv_len * D;
    O += static_cast<size_t>(head) * q_len * D;

    // This lane's slice of the query (pre-multiplied by the scale) and of the output.
    // Lane l holds dimensions l, l + 32, l + 64, ...
    float q[PER_LANE];
    float o[PER_LANE];
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
        q[i] = active ? Q[q_row * D + lane + 32 * i] * scale : 0.0f;
        o[i] = 0.0f;
    }
    float m = -INFINITY;  // running max of the scores seen so far
    float l = 0.0f;       // running sum of exp(score - m)

    // Keys this query may see, and keys ANY query of this block may see.
    const int my_keys = causal ? min(kv_len, q_row + causal_offset + 1) : kv_len;
    const int last_row = min(q_len - 1, (int)(blockIdx.x * FUSED_WARPS + FUSED_WARPS - 1));
    const int block_keys = causal ? min(kv_len, last_row + causal_offset + 1) : kv_len;

    for (int k0 = 0; k0 < block_keys; k0 += KEY_TILE) {
        // ---- All 256 threads load a KEY_TILE x D tile of K and of V.
        for (int i = threadIdx.x; i < KEY_TILE * D; i += blockDim.x) {
            const int r = i / D;
            const int c = i % D;
            const int key = k0 + r;
            Ks[r][c] = key < kv_len ? K[static_cast<size_t>(key) * D + c] : 0.0f;
            Vs[r][c] = key < kv_len ? V[static_cast<size_t>(key) * D + c] : 0.0f;
        }
        __syncthreads();

        if (active) {
            const int tile_keys = min(KEY_TILE, my_keys - k0);  // <= 0 means: all masked for me
            for (int j = 0; j < tile_keys; ++j) {
                // Score = q . k_j : each lane multiplies its slice, the warp sums.
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < PER_LANE; ++i) partial += q[i] * Ks[j][lane + 32 * i];
                const float s = warp_reduce_sum(partial);  // every lane gets the full score

                // Online softmax update (docs/05 section 10), applied to the
                // output too: o = o * exp(m_old - m_new) + exp(s - m_new) * v_j
                const float m_new = fmaxf(m, s);
                const float correction = expf(m - m_new);  // 0 on the first key (m = -inf)
                const float p = expf(s - m_new);
                l = l * correction + p;
#pragma unroll
                for (int i = 0; i < PER_LANE; ++i) o[i] = o[i] * correction + p * Vs[j][lane + 32 * i];
                m = m_new;
            }
        }
        __syncthreads();  // tiles fully used before the next iteration overwrites them
    }

    if (active) {
        const float inv_l = 1.0f / l;
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) O[q_row * D + lane + 32 * i] = o[i] * inv_l;
    }
}

// -----------------------------------------------------------------------------
// Host-side launch functions
// -----------------------------------------------------------------------------
namespace gpu {

void batched_gemm(const float* d_A, const float* d_B, float* d_C, int batch, int M, int N, int K, bool b_transposed,
                  float scale, bool causal, int causal_offset) {
    if (batch <= 0 || M <= 0 || N <= 0) return;
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);  // z = batch
    if (b_transposed) {
        batched_gemm_kernel<TILE, true><<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale, causal, causal_offset);
    } else {
        batched_gemm_kernel<TILE, false><<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale, causal, causal_offset);
    }
    CUDA_CHECK_KERNEL();
}

static void check_shape(const AttentionShape& s) {
    if (s.causal && s.kv_len < s.q_len) {
        std::fprintf(stderr, "attention: causal masking needs kv_len >= q_len (got %d < %d)\n", s.kv_len, s.q_len);
        std::exit(EXIT_FAILURE);
    }
}

void attention_unfused(const float* d_Q, const float* d_K, const float* d_V, float* d_O, float* d_scores,
                       const AttentionShape& s) {
    check_shape(s);
    if (s.heads <= 0 || s.q_len <= 0 || s.kv_len <= 0) return;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.d));
    const int offset = s.kv_len - s.q_len;
    // 1. S = Q K^T * scale (+ mask)
    batched_gemm(d_Q, d_K, d_scores, s.heads, s.q_len, s.kv_len, s.d, /*b_transposed=*/true, scale, s.causal, offset);
    // 2. P = softmax(S), in place: each lane reads x[c] before writing y[c] (same element).
    softmax_online(d_scores, d_scores, s.heads * s.q_len, s.kv_len);
    // 3. O = P V
    batched_gemm(d_scores, d_V, d_O, s.heads, s.q_len, s.d, s.kv_len, /*b_transposed=*/false, 1.0f, false, 0);
}

template <int D>
static void launch_fused(const float* d_Q, const float* d_K, const float* d_V, float* d_O, const AttentionShape& s) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    dim3 grid((s.q_len + FUSED_WARPS - 1) / FUSED_WARPS, s.heads);  // x: groups of 8 queries, y: head
    attention_fused_kernel<D><<<grid, FUSED_WARPS * 32>>>(d_Q, d_K, d_V, d_O, s.q_len, s.kv_len, scale, s.causal,
                                                          s.kv_len - s.q_len);
    CUDA_CHECK_KERNEL();
}

void attention_fused(const float* d_Q, const float* d_K, const float* d_V, float* d_O, const AttentionShape& s) {
    check_shape(s);
    if (s.heads <= 0 || s.q_len <= 0 || s.kv_len <= 0) return;
    switch (s.d) {
        case 32: launch_fused<32>(d_Q, d_K, d_V, d_O, s); break;
        case 64: launch_fused<64>(d_Q, d_K, d_V, d_O, s); break;
        case 128: launch_fused<128>(d_Q, d_K, d_V, d_O, s); break;
        default:
            std::fprintf(stderr, "attention_fused: head dimension %d not supported (32, 64 or 128)\n", s.d);
            std::exit(EXIT_FAILURE);
    }
}

}  // namespace gpu
