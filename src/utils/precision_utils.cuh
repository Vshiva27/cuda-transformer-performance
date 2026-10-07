#pragma once
// =============================================================================
// precision_utils.cuh — host helpers for the FP16 experiments (tests and
// benchmarks share them).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include "precision.cuh"
#include "utils/device_buffer.cuh"

// Round every value of `v` to the nearest FP16 number, using the GPU's own
// conversion (float -> half -> float). Afterwards `v` holds exactly the values
// an FP16 kernel will see, so an "exact" reference can be computed from them.
inline std::vector<float> round_to_half_on_gpu(const std::vector<float>& v) {
    const int n = static_cast<int>(v.size());
    DeviceBuffer<float> d_in(v.size()), d_out(v.size());
    DeviceBuffer<__half> d_half(v.size());
    d_in.copy_from_host(v);
    gpu::float_to_half(d_in.data(), d_half.data(), n);
    gpu::half_to_float(d_half.data(), d_out.data(), n);
    std::vector<float> out(v.size());
    d_out.copy_to_host(out);
    return out;
}

// Error of `out` against an exact (double) reference, relative to the
// largest |reference| value. One number that is comparable across sizes:
// "the worst error, as a fraction of the output's scale".
struct ErrorStats {
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;
    double scaled_err = 0.0;  // max_abs_err / max_abs_ref
    int non_finite = 0;       // inf or NaN outputs (overflow)
};

inline ErrorStats measure_error(const std::vector<double>& ref, const std::vector<float>& out) {
    ErrorStats s;
    for (size_t i = 0; i < ref.size(); ++i) {
        s.max_abs_ref = std::max(s.max_abs_ref, std::fabs(ref[i]));
        if (!std::isfinite(out[i])) {
            ++s.non_finite;
            continue;
        }
        s.max_abs_err = std::max(s.max_abs_err, std::fabs(out[i] - ref[i]));
    }
    s.scaled_err = s.max_abs_ref > 0 ? s.max_abs_err / s.max_abs_ref : s.max_abs_err;
    return s;
}
