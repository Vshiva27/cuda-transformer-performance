#pragma once
// =============================================================================
// host_utils.h — CPU-side helpers: random input data and result comparison.
// Pure C++ (no CUDA), so it can be used from .cpp and .cu files alike.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

// Fill `v` with random numbers in [lo, hi). A fixed `seed` gives the same
// numbers every run, so a failing test can be reproduced.
inline void fill_random(std::vector<float>& v, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) {
        x = dist(gen);
    }
}

struct CompareResult {
    bool ok = true;
    double max_abs_err = 0.0;          // largest |out - ref|
    double max_rel_err = 0.0;          // largest |out - ref| / |ref|
    std::size_t num_bad = 0;           // how many elements failed the tolerance
    std::size_t first_bad_index = 0;   // where the first failure is (if any)
};

// Element i passes if  |out[i] - ref[i]| <= atol + rtol * |ref[i]|
// (the same rule as numpy.allclose / torch.allclose).
//   atol = absolute tolerance: matters when values are near zero.
//   rtol = relative tolerance: matters when values are large.
// A NaN in the output always fails.
inline CompareResult compare_arrays(const std::vector<float>& ref, const std::vector<float>& out,
                                    double atol, double rtol) {
    CompareResult r;
    if (ref.size() != out.size()) {
        r.ok = false;
        r.num_bad = std::max(ref.size(), out.size());
        return r;
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        double ref_i = ref[i];
        double out_i = out[i];
        double diff = std::fabs(out_i - ref_i);
        double tol = atol + rtol * std::fabs(ref_i);
        bool bad = std::isnan(out_i) || diff > tol;
        if (bad) {
            if (r.num_bad == 0) r.first_bad_index = i;
            ++r.num_bad;
            r.ok = false;
        }
        r.max_abs_err = std::max(r.max_abs_err, diff);
        r.max_rel_err = std::max(r.max_rel_err, diff / std::max(std::fabs(ref_i), 1e-12));
    }
    return r;
}

// Tolerance for comparing an FP32 GEMM result of inner dimension K.
// Every addition rounds to the nearest float (relative error up to ~6e-8), and
// a dot product of length K performs K additions, so the possible difference
// between two correct implementations grows with K. Two legitimate sources of
// difference between our CPU and GPU code:
//   - the GPU uses FMA (a*b+c rounded ONCE), the CPU usually does a*b and +c
//     separately (rounded TWICE);
//   - later, optimized kernels add the products in a different order.
// 1e-6 * K + 1e-5 gives ~1e-3 at K = 1024: loose enough for rounding, tight
// enough that any real indexing bug (errors of order 0.1 - 10) is caught.
inline double gemm_tolerance(int K) {
    return 1e-6 * K + 1e-5;
}

inline void print_compare_failure(const CompareResult& r, const std::vector<float>& ref,
                                  const std::vector<float>& out) {
    std::size_t i = r.first_bad_index;
    std::fprintf(stderr, "  %zu mismatches. First at index %zu: expected %.8g, got %.8g\n",
                 r.num_bad, i, i < ref.size() ? ref[i] : 0.0f, i < out.size() ? out[i] : 0.0f);
    std::fprintf(stderr, "  max abs err = %.3g, max rel err = %.3g\n", r.max_abs_err, r.max_rel_err);
}
