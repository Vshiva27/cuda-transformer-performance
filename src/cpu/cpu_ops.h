#pragma once
// =============================================================================
// cpu_ops.h — CPU reference implementations.
//
// Every GPU kernel in this project has a simple CPU version here. The CPU
// version is (1) the "ground truth" that tests compare GPU results against,
// and (2) the baseline that benchmarks compare GPU speed against.
// They are written for clarity, not for maximum CPU speed.
// =============================================================================

#include <cstddef>

namespace cpu {

// c[i] = a[i] + b[i] for i = 0 .. n-1
void vector_add(const float* a, const float* b, float* c, std::size_t n);

// C = A x B. All matrices are row-major (see docs/03_memory_hierarchy.md):
//   A is M x K   (element A[row][k] is at A[row * K + k])
//   B is K x N   (element B[k][col] is at B[k * N + col])
//   C is M x N   (element C[row][col] is at C[row * N + col])
void matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Same product, accumulated and returned in double precision: the "exact"
// answer used to MEASURE the error of FP32 / FP16 GEMMs (docs/08_precision.md).
void matmul_f64(const float* A, const float* B, double* C, int M, int N, int K);

// Row-wise softmax of a row-major (rows x cols) matrix, numerically stable:
//   y[r][c] = exp(x[r][c] - max_r) / sum_j exp(x[r][j] - max_r)
// Computed in double precision internally, so it serves as "ground truth".
void softmax(const float* x, float* y, int rows, int cols);

// Row-wise LayerNorm of a row-major (rows x cols) matrix, in double precision:
//   y[r][c] = (x[r][c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
// var is the population variance (divide by cols), like PyTorch.
void layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps);

// Scaled dot-product attention in double precision, per head:
//   O = softmax(Q K^T / sqrt(d)) V
// Q: [heads][q_len][d], K, V: [heads][kv_len][d], O: [heads][q_len][d].
// causal: query i (position kv_len - q_len + i) only sees keys j <= its position.
void attention(const float* Q, const float* K, const float* V, float* O, int heads, int q_len, int kv_len, int d,
               bool causal);

// h = x + residual (float addition, exactly as on the GPU), then y = LayerNorm(h).
void add_layernorm(const float* x, const float* residual, const float* gamma, const float* beta, float* h, float* y,
                   int rows, int cols, float eps);

}  // namespace cpu
