# 01 — Project Overview

## 1. What this project is

A small, self-written performance lab for **the GPU operations that dominate Transformer inference**.
Each operation is implemented several times, from simple to optimized:

| Operation | CPU reference | PyTorch | CUDA basic | CUDA optimized |
|---|---|---|---|---|
| Vector add (residual connection) | ✔ | ✔ | one thread per element | grid-stride loop |
| GEMM (matrix multiply) | ✔ | ✔ (cuBLAS) | naive | coalesced → shared-memory tiles → register tiles → warp-level Tensor Cores (FP16) |
| Softmax | ✔ | ✔ | one thread per row | one block/warp per row + warp shuffles |
| LayerNorm | ✔ | ✔ | one thread per row | parallel reduction + fused residual add |
| Attention (small) | ✔ | ✔ | QKᵀ → softmax → ×V using our kernels | fused/fp16 variants |
| Precision | FP32 | FP32 / FP16 | FP32 | FP16 inputs + FP32 accumulation |

Every version is **tested for correctness first**, then **benchmarked**, then **profiled**.
Then we explain *why* the faster version is faster.

What this project is **not**: a full LLM, a serving system, or a vLLM clone. We keep the
scope to the kernels, because that is where the GPU performance ideas live.

---

## 2. The pipeline (architecture)

```
             Input data (random FP32 / FP16 tensors, fixed seed)
                                  │
            ┌─────────────────────┼──────────────────────┐
            ▼                     ▼                      ▼
     CPU reference          PyTorch baseline       Our CUDA implementation
     (src/cpu)              (python/)              (kernels/*.cu)
     "ground truth"         "what a framework           │
                             gives you for free"        ▼
                                                  Kernel launch  kernel<<<grid, block>>>(...)
                                                        │
                                                        ▼
                                          GPU threads (grid → blocks → warps → threads)
                                                        │
                                                        ▼
                                          Memory hierarchy
                                          registers → shared memory → L1/L2 → global memory
                                                        │
                                                        ▼
                                          CUDA computation
                                                        │
            ┌───────────────────────────────────────────┘
            ▼
         Result ──► Correctness check (tests/: compare against CPU, with tolerance)
            │
            ▼
         Benchmark (src/benchmark/: warm-up, many iterations, CUDA events)
            │
            ▼
         Profiler (Nsight Systems: timeline; Nsight Compute: per-kernel metrics)
            │
            ▼
         Optimization decision ("memory-bound → improve access pattern / reuse")
            │
            ▼
         Final result (docs/12_results.md — only real, measured numbers)
```

### How kernels connect to AI inference

```
CUDA kernel            e.g. matmul_tiled_kernel
   ↓ runs as
GPU execution          thousands of threads on many SMs, reading/writing GPU memory
   ↓ implements
AI operation           GEMM, softmax, LayerNorm, elementwise add
   ↓ which form
Transformer operation  Q/K/V projections, attention scores, MLP, residual + norm
   ↓ which determine
Inference performance  latency per token, tokens per second, cost per request
```

---

## 3. Directory structure

```
cuda-transformer-performance/
├── README.md               How to build & run (Colab instructions)
├── CMakeLists.txt          Build description for all C++/CUDA code
├── requirements.txt        Python packages for the PyTorch baseline
│
├── src/
│   ├── cpu/                CPU reference implementations (ground truth + baseline)
│   ├── benchmark/          One benchmark program per operation + shared timing helpers
│   └── utils/              Error checking, GPU memory (RAII), timers, device query, comparison
│
├── kernels/                CUDA kernels. Each X.cu holds the kernels AND the host
│                           functions that launch them; X.cuh declares the launchers.
│
├── python/                 PyTorch baseline, Python benchmark driver, results summarizer
├── scripts/                run_all.sh: build → tests (stop on failure) → benchmarks → summary
├── notebooks/              colab_run.ipynb: the whole run in a few clicks
├── tests/                  Correctness tests (one program per operation, run by ctest)
├── benchmarks/             Saved benchmark output (CSV / text) from real runs, one folder per GPU
├── profiling/              Nsight commands/scripts and saved reports
└── docs/                   Learning documentation (this folder)
```

### Changes from the originally suggested structure, and why

1. **No `src/cuda/` folder.** The suggestion had both `src/cuda/` and `kernels/`. That would
   split one idea (a kernel and the code that launches it) across two folders. Instead, each
   `kernels/X.cu` contains the `__global__` kernel(s) **and** the small host function that
   computes the grid size and launches them. You read one file to understand one operation.
2. **Headers next to their source.** `kernels/vector_add.cuh` sits next to `vector_add.cu`.
   `.cuh` is a convention for "header that may contain CUDA code".
3. **`src/utils/` is split into small single-purpose headers** (`cuda_check.cuh`,
   `device_buffer.cuh`, `timer.cuh`, `host_utils.h`, `device_info.*`) so each one can be
   explained on its own.
4. **One static library (`ctp_core`)** holds all shared code. Each test and benchmark is a
   tiny program linked against it. This avoids compiling the same kernels many times.
5. **Docs per topic, not per file.** Each topic document (e.g. `02_cuda_fundamentals.md`)
   contains a "file guide" covering every source file of that topic.

---

## 4. Learning sequence (phases)

Each phase ends with a full explanation before the next one starts.

| Phase | Topic | New files | New concepts |
|---|---|---|---|
| **1** | Foundations | build, utils, `vector_add.cu`, first test & benchmark | kernel, grid/block/thread, indexing, bounds check, grid-stride loop, error checking, async execution, CUDA events, memory-bound, bandwidth, occupancy (first look) |
| 2 | Memory hierarchy + GEMM v1/v2 | `matmul_basic.cu`, CPU GEMM, tests | 2D indexing, row-major layout, coalescing, arithmetic intensity |
| 3 | GEMM v3/v4 | `matmul_tiled.cu`, `matmul_register.cu` | shared memory, `__syncthreads()`, tiling, bank conflicts, register blocking, compute-bound |
| 4 | PyTorch baseline | `python/*.py` | framework overhead, cuBLAS as a reference ceiling |
| 5 | Softmax | `softmax.cu` | numerical stability, reductions, `__shfl_sync()` |
| 6 | LayerNorm | `layernorm.cu` | mean/variance reduction, fused residual-add + LayerNorm |
| 7 | Precision + GEMM v5 | FP16 GEMM, warp-level Tensor Core GEMM (WMMA) | FP16 vs FP32, FP32 accumulation, warp-level MMA, error measurement, quantization (concept) |
| 8 | Attention | `attention.cu` | Q/K/V, QKᵀ, scaling, KV cache (concept) |
| 9 | Benchmark framework | CSV output, experiments | throughput, speedup tables, size sweeps |
| 10 | Nsight profiling | `profiling/` | timeline, occupancy, throughput, bottleneck analysis |
| 11 | Results & interview prep | `12_results.md`, `13_interview_questions.md`, `resume_bullets.md` | turning measurements into conclusions |

---

## 5. Ground rules of the project

1. **Correctness before speed.** Every benchmark first checks its result against the CPU.
2. **No invented numbers.** Tables in `12_results.md` are filled only from real runs.
3. **No assumed GPU.** Everything adapts at runtime (`src/utils/device_info.cu`).
4. **No hidden logic.** We use PyTorch/cuBLAS only as *baselines to compare against*,
   never inside our implementations.
