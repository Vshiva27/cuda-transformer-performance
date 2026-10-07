#pragma once
// =============================================================================
// device_buffer.cuh — owns one block of GPU memory (RAII).
//
// RAII = "Resource Acquisition Is Initialization": the constructor acquires a
// resource (here: GPU memory via cudaMalloc) and the destructor releases it
// (cudaFree). Because C++ always runs the destructor when an object goes out
// of scope, we can never forget cudaFree, even on an early return.
// Explained in docs/02_cuda_fundamentals.md, section "DeviceBuffer".
// =============================================================================

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "utils/cuda_check.cuh"

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // Allocate room for `count` elements of type T on the GPU.
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        // No CUDA_CHECK here: a destructor must not exit the program, and
        // cudaFree(nullptr) is allowed and does nothing.
        cudaFree(ptr_);
    }

    // Copying is forbidden: two objects would own the same pointer and both
    // would call cudaFree on it (a "double free").
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Moving is allowed: ownership is handed over and the source is emptied.
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    // CPU memory -> GPU memory ("host to device").
    void copy_from_host(const std::vector<T>& host) {
        check_size(host.size());
        if (count_ == 0) return;
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
    }

    // GPU memory -> CPU memory ("device to host").
    void copy_to_host(std::vector<T>& host) const {
        check_size(host.size());
        if (count_ == 0) return;
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

private:
    void check_size(std::size_t host_count) const {
        if (host_count != count_) {
            std::fprintf(stderr, "DeviceBuffer size mismatch: host has %zu, device has %zu\n",
                         host_count, count_);
            std::exit(EXIT_FAILURE);
        }
    }

    T* ptr_ = nullptr;       // address in GPU memory (NOT usable on the CPU)
    std::size_t count_ = 0;  // number of elements
};
