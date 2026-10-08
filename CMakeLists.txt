# =============================================================================
# CUDA-Accelerated Transformer Inference Performance Benchmark
# Build file. See docs/02_cuda_fundamentals.md, section "How the project is built".
# =============================================================================

cmake_minimum_required(VERSION 3.18)

# Which GPU architecture(s) nvcc should generate machine code for.
# This must be decided BEFORE project(), because project() turns on the CUDA
# language and picks its own default if we have not chosen one.
#   "native" = "whatever GPU is in this machine right now" (needs CMake >= 3.24).
#   Otherwise we fall back to a list of common GPUs:
#   75 = T4 (Colab free tier), 80 = A100, 86 = RTX 30xx / A10, 89 = L4 / RTX 40xx, 90 = H100.
# You can always override it:  cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
  endif()
endif()

project(cuda_transformer_perf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Benchmarks are meaningless without compiler optimizations, so default to Release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -lineinfo keeps a mapping from GPU machine code back to our .cu source lines.
# It does NOT slow the code down. Nsight Compute uses it to show which source
# line is responsible for slow memory accesses.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Turn this ON to make every kernel launch wait for the kernel to finish and
# report errors immediately (slow, for debugging only). See cuda_check.cuh.
option(CTP_DEBUG_SYNC "Synchronize after every kernel launch to locate errors" OFF)
if(CTP_DEBUG_SYNC)
  add_compile_definitions(CTP_DEBUG_SYNC)
endif()

# -----------------------------------------------------------------------------
# One library that holds all shared code: CPU reference ops, CUDA kernels,
# and utilities. Every benchmark and test links against it.
# -----------------------------------------------------------------------------
add_library(ctp_core STATIC
  src/utils/device_info.cu
  src/cpu/cpu_ops.cpp
  kernels/vector_add.cu
  kernels/matmul_basic.cu
  kernels/matmul_tiled.cu
  kernels/matmul_register.cu
  kernels/softmax.cu
  kernels/layernorm.cu
  kernels/matmul_fp16.cu
  kernels/matmul_wmma.cu
  kernels/attention.cu
  kernels/gemv.cu
)
target_include_directories(ctp_core PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/kernels
)

# -----------------------------------------------------------------------------
# Benchmarks (measure speed)
# -----------------------------------------------------------------------------
add_executable(bench_vector_add src/benchmark/bench_vector_add.cu)
target_link_libraries(bench_vector_add PRIVATE ctp_core)

add_executable(bench_matmul src/benchmark/bench_matmul.cu)
target_link_libraries(bench_matmul PRIVATE ctp_core)

add_executable(bench_softmax src/benchmark/bench_softmax.cu)
target_link_libraries(bench_softmax PRIVATE ctp_core)

add_executable(bench_layernorm src/benchmark/bench_layernorm.cu)
target_link_libraries(bench_layernorm PRIVATE ctp_core)

add_executable(bench_precision src/benchmark/bench_precision.cu)
target_link_libraries(bench_precision PRIVATE ctp_core)

add_executable(bench_attention src/benchmark/bench_attention.cu)
target_link_libraries(bench_attention PRIVATE ctp_core)

add_executable(bench_quantization src/benchmark/bench_quantization.cu)
target_link_libraries(bench_quantization PRIVATE ctp_core)

# -----------------------------------------------------------------------------
# Profiling driver (docs/10_nsight_profiling.md). NVTX ranges label each launch
# on the Nsight Systems timeline; NVTX is header-only and ships with the CUDA
# toolkit. If CMake cannot find it, the program still builds, without labels.
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
add_executable(profile_targets src/benchmark/profile_targets.cu)
target_link_libraries(profile_targets PRIVATE ctp_core)
if(TARGET CUDA::nvtx3)
  target_link_libraries(profile_targets PRIVATE CUDA::nvtx3)
  target_compile_definitions(profile_targets PRIVATE CTP_HAVE_NVTX)
else()
  message(STATUS "NVTX3 not found: profile_targets will not label its launches")
endif()

# -----------------------------------------------------------------------------
# Tests (prove correctness). Run all with:  ctest --test-dir build --output-on-failure
# -----------------------------------------------------------------------------
enable_testing()

add_executable(test_vector_add tests/test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE ctp_core)
add_test(NAME test_vector_add COMMAND test_vector_add)

add_executable(test_matmul tests/test_matmul.cu)
target_link_libraries(test_matmul PRIVATE ctp_core)
add_test(NAME test_matmul COMMAND test_matmul)

add_executable(test_softmax tests/test_softmax.cu)
target_link_libraries(test_softmax PRIVATE ctp_core)
add_test(NAME test_softmax COMMAND test_softmax)

add_executable(test_layernorm tests/test_layernorm.cu)
target_link_libraries(test_layernorm PRIVATE ctp_core)
add_test(NAME test_layernorm COMMAND test_layernorm)

add_executable(test_precision tests/test_precision.cu)
target_link_libraries(test_precision PRIVATE ctp_core)
add_test(NAME test_precision COMMAND test_precision)

add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE ctp_core)
add_test(NAME test_attention COMMAND test_attention)

add_executable(test_quantization tests/test_quantization.cu)
target_link_libraries(test_quantization PRIVATE ctp_core)
add_test(NAME test_quantization COMMAND test_quantization)
