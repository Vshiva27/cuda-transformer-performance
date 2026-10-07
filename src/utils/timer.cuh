#pragma once
// =============================================================================
// timer.cuh — two kinds of stopwatch.
//
// CpuTimer: measures real ("wall-clock") time seen by the CPU.
// GpuTimer: measures time between two points in the GPU's own work queue,
//           using CUDA events.
//
// They can disagree because a kernel launch is ASYNCHRONOUS: the CPU only puts
// the kernel in a queue and immediately continues. See
// docs/02_cuda_fundamentals.md, section "Asynchronous execution and timing".
// =============================================================================

#include <chrono>
#include <cuda_runtime.h>

#include "utils/cuda_check.cuh"

class CpuTimer {
public:
    using Clock = std::chrono::steady_clock;  // never jumps (unlike system time)

    void start() { start_ = Clock::now(); }

    // Milliseconds since start().
    double stop_ms() const {
        Clock::time_point end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    Clock::time_point start_{};
};

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Puts a "start" marker into the GPU's queue. Returns immediately.
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Puts a "stop" marker into the queue, waits until the GPU reaches it,
    // and returns the GPU time between the two markers in milliseconds.
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
