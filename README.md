# CUDA-Accelerated Transformer Inference Performance Benchmark

A hands-on study of the GPU kernels behind Transformer inference: vector add (residual),
GEMM, softmax, LayerNorm and attention. Each one is implemented on the CPU, compared against
PyTorch, and written in CUDA from a basic version to an optimized one. Every version is
**tested for correctness first**, then benchmarked with CUDA events, then profiled with Nsight.

Start reading at [docs/01_project_overview.md](docs/01_project_overview.md).

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Foundations: build, error checking, timers, device query, vector add (naive + grid-stride) | ✅ implemented |
| 2 | Memory hierarchy, GEMM naive + coalesced | ✅ implemented |
| 3 | GEMM tiled (shared memory), register-blocked | ✅ implemented |
| 4 | PyTorch baseline (all ops) + benchmarking methodology | ✅ implemented |
| 5 | Softmax: thread/row, block/row (shared-memory tree), warp/row (shuffles), online | ✅ implemented |
| 6 | LayerNorm (thread/warp/block-in-registers) + fused residual add + LayerNorm | ✅ implemented |
| 7 | FP16 / FP32 accumulation, GEMM v5 (warp-level Tensor Cores), quantization concept | ✅ implemented |
| 8 | Attention: unfused (batched GEMM + softmax) and fused (online softmax), causal mask, KV cache | ✅ implemented |
| 9 | Benchmark framework: one-command run, CSV output, automatic summary, Colab notebook | ✅ implemented |
| – | **First real run: Tesla T4, CUDA 13.0 — all 6 test programs pass; results in [docs/12_results.md](docs/12_results.md)** | ✅ measured |
| 10 | Nsight profiling: resource usage, Nsight Systems timeline, Nsight Compute on every kernel; 13 hypotheses checked against counters ([docs/10 §9](docs/10_nsight_profiling.md)) | ✅ measured (Tesla T4) |
| 11 | [Optimization story](docs/11_optimization.md), [interview Q&A](docs/13_interview_questions.md), [resume bullets](docs/resume_bullets.md) | ✅ |

**Reading order:** [01 overview](docs/01_project_overview.md) → 02–08 (one topic per phase) →
[09 benchmarking](docs/09_benchmarking.md) → [10 profiling](docs/10_nsight_profiling.md) →
[11 optimization story](docs/11_optimization.md) → [12 results](docs/12_results.md) →
[13 interview questions](docs/13_interview_questions.md).

**Headline results (Tesla T4, FP32 unless noted, all measured):**
- **GEMM 1024³:** 61 → 2,236 GFLOP/s across four versions (**36.5×**); the final version is **60% of cuBLAS**.
- **Tensor Cores:** a WMMA kernel is 1.49× faster than the best FP32 kernel.
- **Softmax:** up to **12% faster than PyTorch** on attention-shaped rows.
- **LayerNorm:** reaches 72–74% of peak bandwidth.
- **Fused residual add + LayerNorm:** 1.21× faster than the two separate kernels, and up to **1.5× faster than PyTorch's**.
- **Fused causal attention:** needs no seq² score buffer.
- **KV-cache decoding step:** **30× cheaper** than recomputing attention at a 2,048-token context.

## Requirements

- An NVIDIA GPU (any; the code detects it at runtime)
- CUDA Toolkit (nvcc) 11.x or newer, CMake ≥ 3.18, a C++17 compiler
- Linux is the tested target (Google Colab works out of the box)

## Run everything (one command)

```bash
bash scripts/run_all.sh
```

It records the environment, builds in Release mode, runs **all correctness tests (and stops if
any fails)**, runs every benchmark and the PyTorch baseline, and writes
`benchmarks/<GPU>/summary.md`. See [docs/09_benchmarking.md](docs/09_benchmarking.md), Part 2.

**On Google Colab:** open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb), choose
`Runtime → Change runtime type → GPU`, and run the cells (clone or upload the project, run,
view the summary, download the results).

Individual steps, if you want them:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure        # correctness
./build/bench_matmul --csv matmul.csv             # any bench_*; --csv is optional
python3 python/benchmark.py --ops matmul softmax  # PyTorch baseline (subset)
```

PyTorch is preinstalled on Colab. Elsewhere: `pip install -r requirements.txt`.

If CMake complains about `native` architectures, give the architecture explicitly, e.g.
`-DCMAKE_CUDA_ARCHITECTURES=75` for a T4 (`80` A100, `86` A10/RTX 30xx, `89` L4/RTX 40xx).

To debug a crashing kernel: configure with `-DCTP_DEBUG_SYNC=ON`, or run
`compute-sanitizer ./build/test_vector_add`.

## Layout

```
src/cpu/        CPU reference implementations
src/utils/      error checking, GPU memory (RAII), timers, device query, comparisons
src/benchmark/  benchmark programs
kernels/        CUDA kernels + their launch functions
tests/          correctness tests (ctest)
python/         PyTorch baseline, results summarizer
scripts/        run_all.sh (build → test → benchmark → summary)
notebooks/      Colab notebook
benchmarks/     saved results from real runs (one folder per GPU)
profiling/      Nsight commands and reports
docs/           learning documentation
```
