#pragma once
// =============================================================================
// device_info.h — ask the GPU what it is, at runtime.
//
// We never assume a specific GPU. Colab may give a T4, L4 or A100; each has a
// different number of SMs, memory size and memory bandwidth. Benchmarks use
// this information to (1) skip sizes that do not fit in memory and
// (2) compare measured bandwidth against the GPU's theoretical peak.
// =============================================================================

#include <cstddef>
#include <string>

struct GpuInfo {
    int device_id = 0;
    std::string name;

    int cc_major = 0;  // compute capability, e.g. 7.5 for T4, 8.0 for A100
    int cc_minor = 0;

    int sm_count = 0;               // number of Streaming Multiprocessors
    int warp_size = 0;              // always 32 on NVIDIA GPUs so far
    int max_threads_per_block = 0;  // 1024 on all current GPUs
    int max_threads_per_sm = 0;     // e.g. 1024 (T4), 2048 (A100)
    int regs_per_block = 0;         // 32-bit registers available to one block
    std::size_t shared_mem_per_block = 0;  // bytes (default limit, 48 KB)
    std::size_t shared_mem_per_sm = 0;     // bytes
    int l2_cache_bytes = 0;

    std::size_t total_mem_bytes = 0;  // global memory (VRAM)
    std::size_t free_mem_bytes = 0;   // currently unused VRAM

    int runtime_version = 0;  // CUDA runtime we compiled against, e.g. 12040 = 12.4
    int driver_version = 0;   // newest CUDA version the installed driver supports

    int mem_clock_khz = 0;
    int mem_bus_width_bits = 0;
    double peak_bandwidth_gbs = 0.0;  // theoretical maximum, GB/s

    int sm_clock_khz = 0;             // maximum (boost) SM clock
    int fp32_cores_per_sm = 0;        // from a table by compute capability; 0 = unknown GPU
    double peak_fp32_gflops = 0.0;    // approximate theoretical maximum, 0 if unknown
};

GpuInfo query_gpu_info(int device_id = 0);
void print_gpu_info(const GpuInfo& info);
