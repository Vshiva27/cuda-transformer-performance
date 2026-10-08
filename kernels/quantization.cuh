#pragma once
// =============================================================================
// quantization.cuh — decode GEMV with FP32, FP16 and INT8 (weight-only) weights.
//
// One decoding step multiplies ONE token's activations x (length K) by every
// weight matrix of the layer: y = W x. With M = 1 there is no data reuse, so
// the time is set by how many bytes of W must be read (docs/03 §5, docs/08 §9).
// The three versions below run the SAME kernel and differ only in the type of
// the weights, so the benchmark isolates the effect of bytes per weight:
//   FP32  4 bytes / weight
//   FP16  2 bytes / weight
//   INT8  1 byte  / weight + one FP32 scale per output row ("W8A32")
//
// Layout: W is N x K, row-major, one row per OUTPUT (like PyTorch's
// nn.Linear.weight [out_features, in_features]), so each output reads one
// contiguous row. x (K) and y (N) are FP32; accumulation is FP32.
// Explained in docs/08_precision.md §9.
// =============================================================================

#include <cstdint>
#include <cuda_fp16.h>

namespace gpu {

// y[n] = sum_k W[n][k] * x[k]
void gemv_fp32(const float* d_W, const float* d_x, float* d_y, int N, int K);
void gemv_fp16(const __half* d_W, const float* d_x, float* d_y, int N, int K);

// Symmetric INT8 weights with one scale per output row:
//   W[n][k] ~= q[n][k] * scale[n]   =>   y[n] = scale[n] * sum_k q[n][k] * x[k]
// The scale is constant along the row, so it is applied ONCE per output,
// after the sum, instead of once per weight (same result, K-1 fewer multiplies).
// A per-tensor scale is the special case where all scale[n] are equal.
void gemv_int8(const int8_t* d_q, const float* d_scale, const float* d_x, float* d_y, int N, int K);

// Same result, but each warp computes `rows_per_warp` consecutive outputs (1, 2, 4 or 8).
// Why: with one row per warp, every weight costs 4 bytes of x read through L1 whatever the
// weight type, and for INT8 that x traffic, not DRAM, became the limit (Nsight Compute:
// L1/TEX 93%, DRAM 48%; docs/12 §9.1). Reading each x chunk once into registers and using
// it for R rows divides the x traffic by R.
void gemv_int8_multirow(const int8_t* d_q, const float* d_scale, const float* d_x, float* d_y, int N, int K,
                        int rows_per_warp);

}  // namespace gpu
