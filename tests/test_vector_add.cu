// =============================================================================
// test_vector_add.cu — proves both vector-add kernels are correct.
//
// Sizes are chosen to hit the edge cases of the thread-index formula:
//   1, 7          : fewer elements than one warp
//   31, 32, 33    : just below / exactly / just above one warp
//   255, 256, 257 : just below / exactly / just above one 256-thread block
//   large + odd   : many blocks, last block partly empty
//   0             : empty input must not launch or crash
// The grid-stride kernel is also run with tiny grids (1 and 3 blocks) to prove
// the loop really covers every element when there are fewer threads than elements.
//
// Expected tolerance is EXACTLY zero: a + b is one IEEE-754 operation, so the
// CPU and the GPU must produce bit-identical results.
// =============================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu/cpu_ops.h"
#include "utils/device_buffer.cuh"
#include "utils/device_info.h"
#include "utils/host_utils.h"
#include "vector_add.cuh"

static int g_failures = 0;
static int g_checks = 0;

static void check(const std::string& name, const std::vector<float>& ref, const std::vector<float>& out) {
    ++g_checks;
    CompareResult r = compare_arrays(ref, out, 0.0, 0.0);
    if (!r.ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", name.c_str());
        print_compare_failure(r, ref, out);
    }
}

int main() {
    GpuInfo info = query_gpu_info(0);
    std::printf("Testing vector add on %s\n", info.name.c_str());

    const std::vector<int> sizes = {0, 1, 7, 31, 32, 33, 255, 256, 257, 1000, (1 << 20) + 3};
    const std::vector<int> block_sizes = {32, 128, 256, 1024};

    for (int n : sizes) {
        std::vector<float> h_a(n), h_b(n), h_ref(n);
        fill_random(h_a, 100 + n);
        fill_random(h_b, 200 + n);
        cpu::vector_add(h_a.data(), h_b.data(), h_ref.data(), n);

        DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
        d_a.copy_from_host(h_a);
        d_b.copy_from_host(h_b);

        for (int bs : block_sizes) {
            std::vector<float> h_out(n, -999.0f);  // poison value: catches unwritten elements

            // Fill the output with garbage first, so a kernel that skips an
            // element cannot pass by accident with a leftover correct value.
            d_c.copy_from_host(h_out);
            gpu::vector_add_naive(d_a.data(), d_b.data(), d_c.data(), n, bs);
            d_c.copy_to_host(h_out);
            check("naive n=" + std::to_string(n) + " block=" + std::to_string(bs), h_ref, h_out);

            for (int grid : {1, 3, gpu::vector_add_grid_stride_default_grid(bs)}) {
                std::fill(h_out.begin(), h_out.end(), -999.0f);
                d_c.copy_from_host(h_out);
                gpu::vector_add_grid_stride(d_a.data(), d_b.data(), d_c.data(), n, bs, grid);
                d_c.copy_to_host(h_out);
                check("grid-stride n=" + std::to_string(n) + " block=" + std::to_string(bs) +
                          " grid=" + std::to_string(grid),
                      h_ref, h_out);
            }
        }
    }

    // Catch any error raised while kernels were running.
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("%d / %d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
