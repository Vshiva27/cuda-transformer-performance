#include "utils/device_info.h"

#include <cstdio>
#include <cstdlib>

#include "utils/cuda_check.cuh"

// CUDA has no API for "FP32 cores per SM", so we use NVIDIA's published numbers
// per architecture (compute capability). Returns 0 for unknown architectures.
static int fp32_cores_per_sm(int major, int minor) {
    switch (major) {
        case 6: return minor == 0 ? 64 : 128;   // Pascal: P100 = 64, GTX 10xx = 128
        case 7: return 64;                      // Volta (V100), Turing (T4, RTX 20xx)
        case 8: return minor == 0 ? 64 : 128;   // A100 = 64; A10/RTX 30xx/L4/RTX 40xx = 128
        case 9: return 128;                     // Hopper (H100)
        case 10:
        case 11:
        case 12: return 128;                    // Blackwell
        default: return 0;
    }
}

GpuInfo query_gpu_info(int device_id) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::fprintf(stderr,
                     "No usable NVIDIA GPU found (%s).\n"
                     "On Colab: Runtime -> Change runtime type -> GPU.\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    GpuInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.warp_size = prop.warpSize;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.regs_per_block = prop.regsPerBlock;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    info.l2_cache_bytes = prop.l2CacheSize;

    // Memory clock and bus width are read as "attributes" because newer CUDA
    // versions (13.x) removed these fields from cudaDeviceProp.
    CUDA_CHECK(cudaDeviceGetAttribute(&info.mem_clock_khz, cudaDevAttrMemoryClockRate, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&info.mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth,
                                      device_id));

    // Theoretical peak bandwidth (bytes per second):
    //   2                        -> data moves on both clock edges ("double data rate")
    //   * clock (Hz)             -> transfers per second per pin
    //   * bus width (bits) / 8   -> bytes moved per transfer
    // Example T4: 2 * 5.001e9 * (256/8) = 320 GB/s.
    info.peak_bandwidth_gbs =
        2.0 * (info.mem_clock_khz * 1e3) * (info.mem_bus_width_bits / 8.0) / 1e9;

    // Theoretical peak FP32 compute (FLOP per second):
    //   cores per SM * number of SMs * clock (Hz) * 2
    // The 2 is because one FMA instruction (fused multiply-add: a*b+c) counts
    // as 2 floating-point operations. Example T4: 64 * 40 * 1.59e9 * 2 = 8.1 TFLOP/s.
    CUDA_CHECK(cudaDeviceGetAttribute(&info.sm_clock_khz, cudaDevAttrClockRate, device_id));
    info.fp32_cores_per_sm = fp32_cores_per_sm(info.cc_major, info.cc_minor);
    info.peak_fp32_gflops =
        2.0 * info.fp32_cores_per_sm * info.sm_count * (info.sm_clock_khz * 1e3) / 1e9;

    CUDA_CHECK(cudaMemGetInfo(&info.free_mem_bytes, &info.total_mem_bytes));
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

void print_gpu_info(const GpuInfo& info) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf("=================== GPU ===================\n");
    std::printf("Device %d            : %s\n", info.device_id, info.name.c_str());
    std::printf("Compute capability  : %d.%d\n", info.cc_major, info.cc_minor);
    std::printf("CUDA runtime/driver : %d.%d / %d.%d\n",
                info.runtime_version / 1000, (info.runtime_version % 1000) / 10,
                info.driver_version / 1000, (info.driver_version % 1000) / 10);
    std::printf("SMs                 : %d\n", info.sm_count);
    std::printf("Warp size           : %d\n", info.warp_size);
    std::printf("Max threads / block : %d\n", info.max_threads_per_block);
    std::printf("Max threads / SM    : %d\n", info.max_threads_per_sm);
    std::printf("Registers / block   : %d\n", info.regs_per_block);
    std::printf("Shared mem / block  : %zu KB\n", info.shared_mem_per_block / 1024);
    std::printf("Shared mem / SM     : %zu KB\n", info.shared_mem_per_sm / 1024);
    std::printf("L2 cache            : %d KB\n", info.l2_cache_bytes / 1024);
    std::printf("Global memory       : %.2f GiB total, %.2f GiB free\n",
                info.total_mem_bytes / gib, info.free_mem_bytes / gib);
    std::printf("Peak mem bandwidth  : %.1f GB/s (theoretical)\n", info.peak_bandwidth_gbs);
    if (info.peak_fp32_gflops > 0) {
        std::printf("Peak FP32 compute   : %.0f GFLOP/s (approx: %d cores/SM x %d SMs x %.2f GHz x 2)\n",
                    info.peak_fp32_gflops, info.fp32_cores_per_sm, info.sm_count,
                    info.sm_clock_khz / 1e6);
        // Ridge point: the arithmetic intensity at which a kernel stops being
        // limited by memory and starts being limited by compute (see docs/03).
        std::printf("Ridge point         : %.1f FLOP/byte\n",
                    info.peak_fp32_gflops / info.peak_bandwidth_gbs);
    } else {
        std::printf("Peak FP32 compute   : unknown for compute capability %d.%d\n", info.cc_major,
                    info.cc_minor);
    }
    std::printf("===========================================\n\n");
}
