# 09 — Benchmarking Methodology and the PyTorch Baseline

Part 1 (Phase 4): how every number in this project is measured, and the PyTorch baseline.
Part 2 (Phase 9, later): the experiment framework, CSV collection, and the full comparison.

Files covered: `python/pytorch_baseline.py`, `python/benchmark.py`,
`src/benchmark/bench_utils.cuh` (timing rules shared with C++).

---

## 1. What problem are we solving?

A speed claim is only worth something if:
1. the result is **correct** (fast and wrong is worthless),
2. the **right thing** was timed (the kernel? the copy? the Python call?),
3. the measurement is **repeatable** (warm-up, many iterations),
4. the comparison is **fair** (same shapes, same data type, same arithmetic).

This document defines those rules once. Every benchmark, in C++ and in Python, follows them.

---

## 2. What we measure

| Metric | Definition | Use it for |
|---|---|---|
| **Latency** | Time for one operation (ms) | "How long does one layer / one token take?" |
| **Throughput** | Work per second: elements/s, tokens/s, GFLOP/s, GB/s | "How much can the GPU process?" |
| **Kernel time** | GPU time of the kernel(s) only (CUDA events) | Comparing kernel implementations |
| **End-to-end time** | Everything the caller waits for: copies + launches + kernels | Real application cost |
| **GFLOP/s** | FLOPs ÷ time ÷ 10⁹ | Compute-bound ops (large GEMM) |
| **GB/s** | Bytes that *must* move ÷ time ÷ 10⁹ | Memory-bound ops (add, softmax, LayerNorm, decode GEMM) |
| **% of peak** | Achieved ÷ theoretical (from device query) | How much headroom is left |
| **Speedup** | baseline time ÷ new time | Always say *which* baseline and *which* time |
| **Extra memory** | Peak GPU bytes allocated during one call | Memory cost (e.g. attention's seq² matrix) |

Use the roofline (03 §4) to choose between GFLOP/s and GB/s. A memory-bound kernel judged by
GFLOP/s always looks terrible, and that tells you nothing.

---

## 3. Three different clocks

```
CPU (Python)  : [call][call][call] ...................... [sync]
                 │     │     │
GPU queue     :  ▼     ▼     ▼
GPU execution :  [k1]  [k2]  [k3]
                 ↑                ↑
          start event        stop event
```

1. **CPU wall clock** (`time.perf_counter`, `std::chrono::steady_clock`): real elapsed time seen
   by the program. It includes Python overhead, PyTorch's dispatcher (the C++ code that picks
   which kernel to run), the launch itself, and waiting for the GPU (only if you synchronize).
   **Without synchronization it only measures launches** (02 §10).
2. **CUDA events**: timestamps recorded *by the GPU* when it reaches a marker in its queue.
   `elapsed_time(start, stop)` is the GPU time between the two markers.
3. **Profiler kernel duration** (Nsight, Phase 10): the exact start and end of each kernel on
   the GPU.

### The subtle point: when events also measure overhead

Events measure the **span** between two markers, including any time the GPU sat **idle**
waiting for the CPU to send the next kernel.

- **Large kernels** (e.g. a 1024³ GEMM, ~1 ms): the CPU queues launches much faster than the GPU
  finishes them, so the queue never empties. Event time ≈ pure kernel time.
- **Tiny kernels** (e.g. vector add of 1024 elements, ~2 µs of real work): each Python call
  costs ~5–20 µs of CPU time, so the GPU finishes each kernel and then **waits**. Event time ≈
  wall time ≈ CPU overhead. The GPU is mostly idle. This is called being **launch-bound** (or
  CPU-bound).

That is why `benchmark.py` prints both `GPU ms` and `wall ms`:
- `wall ≈ GPU` and the time is flat across small sizes → launch-bound. The number measures
  overhead, not the kernel.
- The time grows with the size → the GPU work dominates, and the number is meaningful.

**Inference consequence:** a decode step of a small model runs hundreds of tiny kernels. If each
costs 10 µs of launch overhead, overhead alone limits tokens per second. Real systems fight this
by **fusing kernels** (fewer launches; Phases 6 and 8) and with **CUDA Graphs** (record a whole
sequence of launches once, then replay it with a single launch).

Our C++ benchmarks have the same effect, but smaller: a C++ launch costs a few µs rather than
the 10+ µs of a Python call. For true per-kernel durations of tiny kernels, use Nsight Systems
(Phase 10).

---

## 4. Warm-up: what the first calls pay for

| One-time cost | Typical size | Who pays |
|---|---|---|
| Creating the CUDA context (driver setup, memory mapping) | 100 ms – 1 s | the very first CUDA call |
| Loading kernel code onto the GPU (lazy module loading) | ms | the first launch of each kernel |
| cuBLAS handle creation and algorithm selection | ms | the first `torch.matmul` of each shape/dtype |
| PyTorch caching allocator obtaining memory via `cudaMalloc` | µs – ms | the first allocation of each size |
| GPU raising its clock from idle | ms | after idle periods |

Warm-up runs absorb all of these. `time_cuda` does 5 warm-up calls; `choose_iters` also calls
the function twice before it measures anything.

## 5. Iterations, averaging, noise

- One call is too short and too noisy to time. We run N calls between one pair of events and
  divide by N. `choose_iters` picks N so that one measurement lasts ~0.2 s (5 ≤ N ≤ 200).
- The averaged number is **steady-state** time: what you get when this op runs many times in a
  row, which is how inference runs.
- Sources of noise: GPU boost clocks vary with temperature and power; on Colab you share the
  physical machine with other users. Run a benchmark twice; if numbers differ by more than
  ~5%, report a range rather than one value.

## 6. Fairness rules for comparing against PyTorch

1. **Same shapes.** `benchmark.py` uses the same shape lists as `src/benchmark/*.cu`.
2. **Same arithmetic.** On Ampere+ GPUs, PyTorch may use **TF32** for FP32 matmul: inputs are
   rounded to 10 mantissa bits inside Tensor Cores. That is faster but different math. We
   disable it (`configure_for_fair_comparison`), and measure it separately as an experiment.
3. **Same accumulation rule for FP16.** We forbid FP16 "reduced precision reduction", so cuBLAS
   accumulates in FP32, matching our Phase 7 kernel.
4. **No autograd.** `torch.inference_mode()` turns off gradient bookkeeping, as in real
   inference.
5. **Same correctness bar.** Every PyTorch result is checked against a float64 CPU reference
   (`max err` column), just like our kernels are checked against `cpu::*`.
6. **CPU comparisons are labelled.** PyTorch's CPU ops are multi-threaded and use vectorized
   math libraries; our C++ CPU reference is single-threaded and plain. They answer different
   questions, so never mix them in one speedup claim.

---

## 7. What happens inside a PyTorch call

```
c = a @ b          (Python)
   ↓  Python → C++ binding
   ↓  dispatcher: device = CUDA, dtype = float32, inference mode
   ↓  output allocated from the caching allocator (usually no cudaMalloc)
   ↓  cuBLAS: choose a GEMM kernel for this shape/dtype/GPU (cached after the first call)
   ↓  launch on the current CUDA stream (asynchronous)
returns immediately with a tensor whose data will exist "soon"
```

| Our operation | PyTorch call | Kernel behind it |
|---|---|---|
| vector add | `a + b` | PyTorch elementwise kernel (vectorized, grid-stride) |
| GEMM | `a @ b` | cuBLAS (tiling at block, warp and thread level, vectorized loads, double buffering, Tensor Cores for FP16/TF32) |
| softmax | `torch.softmax(x, -1)` | PyTorch's own softmax kernel (warp/block reductions) |
| LayerNorm | `F.layer_norm` | PyTorch's own LayerNorm kernel |
| attention (naive) | `QKᵀ`, softmax, `@V` | 2 cuBLAS batched GEMMs + 1 softmax kernel + 1 scaling kernel |
| attention (SDPA) | `F.scaled_dot_product_attention` | a backend chosen by PyTorch: FlashAttention / memory-efficient kernel (never stores seq×seq) **or** a "math" fallback that does. Measured on T4 with FP32: no memory saving (12_results §6) |

**Why cuBLAS is the reference ceiling for GEMM:** NVIDIA tunes it per GPU architecture using all
the techniques in 04 §23. Our v4 being, say, 50% of cuBLAS would be a good result for a
readable kernel. Whatever the ratio, the interesting part is explaining *where the gap comes
from*.

---

## 8. Measuring memory

PyTorch never gives freed GPU memory back to CUDA; it keeps it in a **caching allocator** for
reuse (because `cudaMalloc`/`cudaFree` are slow and synchronize the GPU). So `nvidia-smi` shows
reserved memory, not what an op actually needs.

`peak_extra_memory_mb(fn)` asks the allocator instead:
```
before = memory_allocated()       # bytes in live tensors now
reset_peak_memory_stats()
out = fn()
peak = max_memory_allocated()     # highest point during the call
extra = peak - before             # output + every temporary
```

Expected pattern (verify with your run):
- vector add, softmax, LayerNorm: extra ≈ the output size.
- **naive attention**: extra includes `scores` and `probs`, each heads × seq × seq floats.
  For 12 heads × 2048 × 2048 × 4 B ≈ 200 MB each. This grows with **seq²**.
- **SDPA**: *if* PyTorch selects a fused backend, extra ≈ output + small buffers, because the
  kernel works on tiles of the score matrix and never writes the whole thing to global memory.
  **Measured on a T4 with FP32 inputs, this did not happen:** SDPA allocated 438 MB at seq 2048,
  more than the naive path (390 MB). The memory-saving backends evidently weren't used for FP32
  on this GPU. `benchmark.py` therefore also measures SDPA with FP16 inputs. This is a good
  example of why you measure instead of assuming what a library does.

This memory difference is the main reason FlashAttention exists. We return to it in Phase 8.

---

## 9. File guide

### `python/pytorch_baseline.py`
1. **What:** PyTorch implementations of all ops + timing/memory/error helpers.
2. **Why:** a professional reference to measure our kernels against.
3. **Inputs:** CUDA (or CPU) tensors.
4. **Outputs:** result tensors; `Timing(gpu_ms, wall_ms)`; floats for memory and error.
5. **Data flow:** tensors are created on the CPU, copied to the GPU once, and the ops run on the
   GPU.
6. **Functions:** `gpu_info`, `configure_for_fair_comparison`, `time_cuda`, `time_cpu`,
   `peak_extra_memory_mb`, `max_abs_error`, `vector_add`, `matmul`, `softmax`, `layernorm`,
   `attention_naive`, `attention_sdpa`.
7. **Important variables:** `start`/`stop` (`torch.cuda.Event(enable_timing=True)`), `iters`,
   `before`/`peak`.
8. **CUDA concepts:** events, synchronization, asynchronous launch, caching allocator.
9. **Memory:** naive attention stores seq² intermediates; SDPA avoids them only when a fused
   backend is selected (not the case for FP32 on the measured T4).
10. **Thread mapping:** hidden inside PyTorch/cuBLAS. That is exactly why it is only a baseline.
11. **Synchronization:** `torch.cuda.synchronize()` after warm-up; `stop.synchronize()` before
    reading the time.
12. **Performance:** small ops are launch-bound (§3).
13. **Mistakes:** §10.

Python features used:
- `@dataclass`: a decorator that auto-generates the constructor for `Timing`.
- `lambda: …`: a small unnamed function, used to pass "the work to time".
- Type hints such as `-> float`: documentation only; Python does not enforce them.

### `python/benchmark.py`
Runs `bench_vector_add`, `bench_matmul` (FP32, FP16, transformer shapes, TF32 experiment on
cc ≥ 8.0), `bench_softmax`, `bench_layernorm`, `bench_attention` (naive vs SDPA). It prints one
table per op and writes `benchmarks/pytorch_<gpu>.csv`.
- `choose_iters`: adaptive iteration count.
- `fits_in_gpu`: the half-of-free-memory rule, the same as in C++.
- `make_row`: derives GFLOP/s and GB/s from GPU time.
- `--ops` selects a subset.

Shapes are module-level constants shared conceptually with the C++ benchmarks. Softmax,
LayerNorm and attention C++ benchmarks (Phases 5, 6, 8) will use these exact shapes.

---

## 10. Common mistakes

1. Timing without synchronizing → measures launches only.
2. No warm-up → measures CUDA context creation or cuBLAS setup.
3. Leaving TF32 on and calling it "FP32".
4. Comparing one tiny op's time and concluding "PyTorch is slow". It is launch-bound; the
   kernel itself may be fast.
5. Reading `nvidia-smi` for an op's memory use (caching allocator).
6. Comparing a multi-threaded CPU library with a single-threaded loop without saying so.
7. Quoting a speedup without naming the baseline and the kind of time (kernel or end-to-end).
8. Benchmarking with autograd enabled.

## 11. Interview explanation

> "I benchmark with CUDA events after warm-up, averaging many back-to-back iterations. Events
> measure GPU time between two points in the stream, which excludes CPU noise. But I also record
> wall-clock time, because when kernels are tiny the GPU sits idle waiting for launches, and
> then both numbers just measure overhead. That launch-bound regime is exactly what kernel
> fusion and CUDA Graphs address in inference engines. For the PyTorch baseline I disable TF32
> so FP32 means true FP32, use inference mode, use the same shapes as my CUDA benchmarks, and
> check every result against a float64 reference. Memory comes from the caching allocator's
> peak statistics. That's how I measured that naive attention allocates seq-squared
> intermediates, 390 MB at sequence length 2048, and also that PyTorch's SDPA with FP32 inputs
> on a T4 did not avoid them. It allocated even more, so it clearly didn't pick a memory-saving
> fused backend for that dtype. My own fused kernel needs no score buffer at all."

## 12. What to remember

- Correct first, then timed. Name the baseline and the kind of time.
- Warm-up, then many iterations between one event pair.
- Events measure a GPU-side span. For tiny ops it is mostly launch overhead → you are
  launch-bound.
- Fair comparison: same shapes, same precision (TF32 off), inference mode.
- PyTorch memory: use allocator stats, not `nvidia-smi`.
- GFLOP/s for compute-bound work, GB/s for memory-bound work.

---
---

# Part 2 (Phase 9): the benchmark framework

## 13. What problem are we solving?

Each phase added its own benchmark program. Running them one by one and copying numbers by hand
into a results table is slow, and it's how mistakes and invented-looking numbers get into
reports. The framework makes a full run **one command**. It is **correctness-gated**, so it
refuses to benchmark if a test fails, it is **recorded** (environment plus raw output plus CSV),
and it is **summarized automatically**.

```
scripts/run_all.sh
   │
   ├─ environment.txt        nvidia-smi, nvcc, cmake, torch versions, clocks
   ├─ cmake build (Release)
   ├─ ctest  ──── any failure → STOP (set -euo pipefail)
   ├─ bench_vector_add --csv … ┐
   ├─ bench_matmul     --csv … │  each prints its tables (→ .txt)
   ├─ bench_softmax    --csv … │  and writes CSV rows   (→ .csv)
   ├─ bench_layernorm  --csv … │
   ├─ bench_precision  --csv … │
   ├─ bench_attention  --csv … ┘
   ├─ python/benchmark.py --csv pytorch.csv
   └─ python/summarize.py  →  summary.md (headline numbers + every table)
```

## 14. How to run it

On Colab: open `notebooks/colab_run.ipynb`, choose a GPU runtime, run the cells. Anywhere else
with an NVIDIA GPU:

```bash
bash scripts/run_all.sh
# results: benchmarks/<GPU_name>/summary.md
```

## 15. `scripts/run_all.sh`, explained

```bash
set -euo pipefail
```
- `-e`: stop at the first command that fails.
- `-u`: using an undefined variable is an error (catches typos).
- `-o pipefail`: in `ctest | tee tests.txt`, the pipeline normally reports only the *last*
  command's status (`tee`, which always succeeds). With `pipefail`, a failing `ctest` fails the
  whole line, so the script stops **before any benchmark runs**. This one option enforces
  "correctness before speed".

```bash
cd "$(dirname "$0")/.."
```
`$0` is the script's own path. Its directory's parent is the project root. So the script works
no matter which directory it's started from.

```bash
GPU_TAG=$(echo "$GPU_NAME" | sed 's/[^A-Za-z0-9]/_/g')
```
"Tesla T4" → `Tesla_T4`: a safe folder name. Results from different GPUs never overwrite each
other.

```bash
"./build/bench_$bench" --csv "$OUT/$bench.csv" 2>&1 | tee "$OUT/$bench.txt"
```
`2>&1` also captures error messages. `tee` shows the output live *and* saves it.

**Line endings:** a script saved on Windows has CRLF line endings, and bash on Linux then fails
with errors like `$'\r': command not found`. The repository uses LF (`.gitattributes`), and the
notebook strips `\r` before running, in case files were uploaded as a zip from Windows.

## 16. CSV output (`src/benchmark/csv_log.h`)

Every C++ benchmark now has `int main(int argc, char** argv)` and creates
`CsvLog log(argc, argv, "<benchmark>", info.name);`. Without `--csv`, `log.add(...)` does
nothing. With `--csv path`, each measurement becomes one row:

| Column | Meaning |
|---|---|
| gpu | device name from the runtime query |
| benchmark | `matmul`, `softmax`, … |
| experiment | e.g. `A square`, `B fusion`, `B kv cache` |
| impl | version name, e.g. `v4 register 4x4`, `fused add_layernorm` |
| shape | e.g. `square 1024x1024x1024`, `4096x1024`, `h=12 seq=2048 d=64` |
| dtype | `fp32` / `fp16` |
| ms, gflops, gbs | metrics; an empty cell means "not applicable" |
| note | extra facts: workspace MB, error values, overflow counts, launch counts |

C++ concepts used:
- `argc`/`argv`: the number of command-line words and the words themselves.
- `std::strcmp`: compares C strings.
- `std::fflush`: makes sure each row reaches the file immediately, so the rows survive even if a
  later benchmark crashes.
- **RAII** again: the destructor closes the file.

The shape labels are **deliberately identical** to `python/benchmark.py`'s labels (e.g.
`square 1024x1024x1024`, `4096x1024`, `h=12 seq=512 d=64`), so that `summarize.py` can put our
kernel and PyTorch's on the same line.

## 17. `python/summarize.py`

1. Reads all CSVs of one run folder.
2. **Headline numbers**: looks up specific measured rows (for example the GEMM versions and cuBLAS
   at 1024³, softmax/LayerNorm at 4096×1024, fusion, attention at seq 512/2048, KV cache) and
   computes the ratios. If a row is missing (e.g. no Tensor Cores), it prints **n/a**. It never
   fills in a guess.
3. **Every measurement** as Markdown tables, grouped by benchmark and experiment.

`find(rows, **criteria)` returns the first row matching all given column values. `**criteria`
collects keyword arguments into a dictionary: `find(rows, impl="v1 naive", shape="…")`.

## 18. Which experiment answers which question

| Question (from the project requirements) | Where |
|---|---|
| Basic CUDA vs CPU | `vector_add` A; `matmul` A (`vs CPU`); softmax/LayerNorm A |
| Tiled CUDA vs basic CUDA | `matmul` A (`vs prev` for v3) |
| Register optimization vs tiled | `matmul` A (`vs prev` for v4) |
| Basic vs optimized softmax | `softmax` A and B |
| Different block sizes | `vector_add` B, `matmul` C, `softmax` C; tile sizes: `matmul` D |
| Different matrix sizes (32 … 2048) | `matmul` A |
| FP32 vs FP16 vs FP16-in/FP32-acc | `precision` A–C |
| Kernel fusion | `layernorm` B; attention unfused vs fused |
| Memory usage | attention workspace column; PyTorch `extra MB` |
| KV cache | `attention` B |
| Comparison with a production library | `pytorch.csv` (cuBLAS, PyTorch kernels, SDPA) |

## 19. Repeatability

- **Run twice.** If headline numbers differ by more than ~5% between runs, report a range.
  Colab GPUs are shared and their clocks vary with temperature and power.
- `environment.txt` records the clocks at the start of the run. Under heavy load,
  `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv` shows whether the GPU is
  running below its maximum clock.
- Keep **one folder per GPU**. Never mix numbers from different GPUs in one comparison.

## 20. Common mistakes (framework)

1. Benchmarking after a failing test (prevented by `pipefail`).
2. Copying numbers by hand into the report (typos, selective reporting). Quote `summary.md`.
3. Mixing results from different runs or GPUs in one table.
4. CRLF line endings in shell scripts.
5. Changing a shape label in one benchmark but not the other, which silently breaks the
   C++ ↔ PyTorch matching in the summary (it shows n/a, not a wrong number).
