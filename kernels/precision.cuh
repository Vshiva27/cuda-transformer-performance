#pragma once
// =============================================================================
// precision.cuh — FP16 conversion and reduced-precision GEMM launchers.
//
// __half is CUDA's 16-bit floating-point type (from <cuda_fp16.h>):
// 1 sign bit, 5 exponent bits, 10 mantissa bits. Half the bytes of float,
// ~3 decimal digits of precision, maximum value 65504.
// Explained in docs/08_precision.md.
//
// All GEMMs here: A (M x K) and B (K x N) are FP16, row-major; C (M x N) is
// FP32, so that the accumulation error can be measured precisely.
// =============================================================================

#include <cuda_fp16.h>

namespace gpu {

// Elementwise conversions on the GPU (round to nearest even).
void float_to_half(const float* d_in, __half* d_out, int n);
void half_to_float(const __half* d_in, float* d_out, int n);

// Shared-memory tiled GEMM (same algorithm as v3) with FP16 inputs.
// The ONLY difference between the two is the type of the running sum:
void matmul_tiled_fp16_acc32(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);  // float sum
void matmul_tiled_fp16_acc16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);  // __half sum

// GEMM v5 — warp-level Tensor Core GEMM using the WMMA API.
// FP16 inputs, FP32 accumulation. Each warp computes a 16 x 16 tile of C.
// Requirements: compute capability >= 7.0, and M, N, K multiples of 16.
void matmul_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);

// True if this GPU has Tensor Cores usable by WMMA (compute capability >= 7.0).
bool wmma_supported();

}  // namespace gpu
