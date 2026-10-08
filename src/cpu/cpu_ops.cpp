#include "cpu/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cpu {

void vector_add(const float* a, const float* b, float* c, std::size_t n) {
    // One CPU thread walks through all n elements, one after another.
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Mathematically: C[row][col] = sum over k of A[row][k] * B[k][col].
    //
    // The loops are ordered row -> k -> col (not row -> col -> k) on purpose.
    // The innermost loop then walks along a ROW of B and a row of C, i.e.
    // through consecutive memory addresses, which the CPU cache handles well.
    // The textbook row -> col -> k order would walk DOWN a column of B, jumping
    // N floats each step, and is several times slower for large N.
    // Each C[row][col] still receives its products in the order k = 0, 1, 2, ...
    // exactly like the GPU kernels, so the results are directly comparable.
    for (int row = 0; row < M; ++row) {
        float* c_row = C + static_cast<std::size_t>(row) * N;
        for (int col = 0; col < N; ++col) {
            c_row[col] = 0.0f;
        }
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<std::size_t>(row) * K + k];
            const float* b_row = B + static_cast<std::size_t>(k) * N;
            for (int col = 0; col < N; ++col) {
                c_row[col] += a * b_row[col];
            }
        }
    }
}

void matmul_f64(const float* A, const float* B, double* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        double* c_row = C + static_cast<std::size_t>(row) * N;
        for (int col = 0; col < N; ++col) c_row[col] = 0.0;
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<std::size_t>(row) * K + k];
            const float* b_row = B + static_cast<std::size_t>(k) * N;
            for (int col = 0; col < N; ++col) c_row[col] += a * b_row[col];
        }
    }
}

void softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + static_cast<std::size_t>(r) * cols;
        float* y_row = y + static_cast<std::size_t>(r) * cols;

        // 1. Row maximum. Subtracting it makes the largest exponent exp(0) = 1,
        //    so exp() can never overflow (docs/05_softmax.md, section 2).
        double row_max = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < cols; ++c) row_max = std::fmax(row_max, static_cast<double>(x_row[c]));

        // 2. Sum of the shifted exponentials.
        double row_sum = 0.0;
        for (int c = 0; c < cols; ++c) row_sum += std::exp(x_row[c] - row_max);

        // 3. Normalize.
        for (int c = 0; c < cols; ++c) {
            y_row[c] = static_cast<float>(std::exp(x_row[c] - row_max) / row_sum);
        }
    }
}

void layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + static_cast<std::size_t>(r) * cols;
        float* y_row = y + static_cast<std::size_t>(r) * cols;

        // Two-pass: mean first, then the mean of squared differences.
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += x_row[c];
        const double mean = sum / cols;

        double sq_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double d = x_row[c] - mean;
            sq_sum += d * d;
        }
        const double inv_std = 1.0 / std::sqrt(sq_sum / cols + eps);

        for (int c = 0; c < cols; ++c) {
            y_row[c] = static_cast<float>((x_row[c] - mean) * inv_std * gamma[c] + beta[c]);
        }
    }
}

void attention(const float* Q, const float* K, const float* V, float* O, int heads, int q_len, int kv_len, int d,
               bool causal) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const int offset = kv_len - q_len;
    std::vector<double> scores(kv_len);
    for (int h = 0; h < heads; ++h) {
        const float* Qh = Q + static_cast<std::size_t>(h) * q_len * d;
        const float* Kh = K + static_cast<std::size_t>(h) * kv_len * d;
        const float* Vh = V + static_cast<std::size_t>(h) * kv_len * d;
        float* Oh = O + static_cast<std::size_t>(h) * q_len * d;
        for (int i = 0; i < q_len; ++i) {
            const int visible = causal ? std::min(kv_len, i + offset + 1) : kv_len;  // keys 0 .. visible-1
            // 1. scores and their max
            double row_max = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < visible; ++j) {
                double dot = 0.0;
                for (int c = 0; c < d; ++c) dot += static_cast<double>(Qh[i * d + c]) * Kh[j * d + c];
                scores[j] = dot * scale;
                row_max = std::max(row_max, scores[j]);
            }
            // 2. softmax weights
            double sum = 0.0;
            for (int j = 0; j < visible; ++j) {
                scores[j] = std::exp(scores[j] - row_max);
                sum += scores[j];
            }
            // 3. weighted sum of value rows
            for (int c = 0; c < d; ++c) {
                double acc = 0.0;
                for (int j = 0; j < visible; ++j) acc += scores[j] * Vh[j * d + c];
                Oh[i * d + c] = static_cast<float>(acc / sum);
            }
        }
    }
}

void add_layernorm(const float* x, const float* residual, const float* gamma, const float* beta, float* h, float* y,
                   int rows, int cols, float eps) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    for (std::size_t i = 0; i < n; ++i) h[i] = x[i] + residual[i];
    layernorm(h, gamma, beta, y, rows, cols, eps);
}

void quantize_int8(const float* W, std::int8_t* q, float* scale, int rows, int cols, bool per_tensor) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    float tensor_max = 0.0f;
    if (per_tensor) {
        for (std::size_t i = 0; i < n; ++i) tensor_max = std::max(tensor_max, std::fabs(W[i]));
    }
    for (int r = 0; r < rows; ++r) {
        const float* w_row = W + static_cast<std::size_t>(r) * cols;
        float max_abs = tensor_max;
        if (!per_tensor) {
            max_abs = 0.0f;
            for (int c = 0; c < cols; ++c) max_abs = std::max(max_abs, std::fabs(w_row[c]));
        }
        // An all-zero row gets scale 0 and q = 0, which dequantizes to exactly 0.
        const float s = max_abs / 127.0f;
        scale[r] = s;
        for (int c = 0; c < cols; ++c) {
            // nearbyint rounds to the nearest integer (ties to even); |w| <= max_abs keeps
            // the result in [-127, 127], the clamp only guards against rounding at the edge.
            const float v = s > 0.0f ? std::nearbyint(w_row[c] / s) : 0.0f;
            q[static_cast<std::size_t>(r) * cols + c] = static_cast<std::int8_t>(std::min(127.0f, std::max(-127.0f, v)));
        }
    }
}

void gemv_f64(const float* W, const float* x, double* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        const float* w_row = W + static_cast<std::size_t>(n) * K;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += static_cast<double>(w_row[k]) * x[k];
        y[n] = acc;
    }
}

void gemv_int8_f64(const std::int8_t* q, const float* scale, const float* x, double* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        const std::int8_t* q_row = q + static_cast<std::size_t>(n) * K;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += static_cast<double>(q_row[k]) * x[k];
        y[n] = acc * scale[n];
    }
}

}  // namespace cpu
