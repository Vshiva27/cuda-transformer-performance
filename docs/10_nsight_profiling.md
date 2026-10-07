# 10 — Profiling with Nsight Systems and Nsight Compute

Files covered: `src/benchmark/profile_targets.cu`, `profiling/resource_usage.sh`,
`profiling/nsys_timeline.sh`, `profiling/ncu_kernels.sh`, `notebooks/colab_profile.ipynb`.

> **No profiler numbers appear in this document until they are measured.** §9 is filled in from
> your reports. Everything before it explains *how to read* a report, and §8 lists hypotheses:
> statements the profiler will confirm or refute.

---

## 1. Why profile, when we already have benchmarks?

A benchmark tells you **how long** a kernel takes. It does not tell you **why**. For example:
- v4 GEMM reached 60% of cuBLAS. Is the remaining 40% lost to shared memory, instruction
  issue, register pressure, or something else?
- 32×1 blocks were the fastest v2 GEMM, against our occupancy argument. Why?

A profiler reads the GPU's **hardware performance counters**: counts of bytes moved at every
cache level, instructions issued per pipeline, cycles each warp spent stalled and for what
reason. With those, "why" becomes measurable.

The workflow used in this project:

```
measure (benchmark) → hypothesis ("v3 is limited by shared memory") → profile (check the metric)
      ↑                                                                         │
      └──── re-measure ◄──── change one thing ◄──── decide (which bottleneck) ◄──┘
```

Three tools, three levels of detail:

| Tool | Level | Answers | Cost |
|---|---|---|---|
| `cuobjdump --dump-resource-usage` | compiled code | registers, shared memory, spills per kernel | instant, no GPU needed |
| **Nsight Systems** (`nsys`) | whole program, timeline | which kernels run, how long, gaps, launch overhead, copies | low overhead (~normal speed) |
| **Nsight Compute** (`ncu`) | one kernel, in depth | why this kernel is slow: memory, compute, occupancy, stalls | high (each kernel replayed many times) |

Rule of thumb: **Nsight Systems first** (find *which* kernel or gap matters), **then Nsight
Compute on that kernel** (find *why*).

---

## 2. Setup

**Build:** profile a Release build (`-O3`). The project compiles with `-lineinfo`, so Nsight
Compute can map metrics back to `.cu` source lines.

**Availability:** both tools ship with the CUDA toolkit (`ncu`, `nsys` in
`/usr/local/cuda/bin`). Colab's CUDA image normally includes them; the profiling notebook
checks this. If they're missing, install "Nsight Compute" / "Nsight Systems" for your CUDA
version from NVIDIA's developer site.

**Permissions:** reading performance counters may require admin rights. If `ncu` prints
`ERR_NVGPUCTRPERM`, the environment forbids counter access, and only `nsys` and
`resource_usage.sh` will work there.

**Three things Nsight Compute does that change what you see:**
1. **Replay.** Each profiled kernel is run many times (once per group of counters), and its
   memory is saved and restored between replays. That's why profiling is slow, and why
   `profile_targets` launches each kernel only once.
2. **Clock control.** By default, `ncu` locks the GPU to its **base clock** (for a T4, well below
   its 1,590 MHz boost clock), so results are repeatable. Durations in `ncu` are therefore
   *longer* than in our benchmarks. Use the **percentages** (% of peak), not the milliseconds,
   when comparing with the benchmarks. `sm__cycles_elapsed.avg.per_second` in the metrics CSV
   shows the actual clock during the measurement.
3. **Cache control.** Caches are flushed before each kernel by default. Every kernel starts
   "cold", so DRAM byte counts show what the kernel truly needs from memory. This is unlike our
   benchmarks, where small inputs stayed in L2 between iterations (12_results, rows marked *).

---

## 3. Static resource usage: `bash profiling/resource_usage.sh`

**Command:**

```bash
cuobjdump --dump-resource-usage build/libctp_core.a | c++filt
```
- `cuobjdump` inspects compiled GPU code inside a binary.
- `c++filt` converts mangled C++ names (`_Z20matmul_register_kernelPKfS0_Pfiii`) into readable
  ones (`matmul_register_kernel(const float*, const float*, float*, int, int, int)`).

**Output format** (one line per kernel and per template instantiation):

```
Function <kernel name>: REG:<registers per thread> STACK:<bytes> SHARED:<bytes per block> LOCAL:<bytes per thread> ...
```

| Field | Meaning | Good / bad | Decision |
|---|---|---|---|
| REG | 32-bit registers per thread | more registers = fewer threads fit per SM | if occupancy is too low *and* the kernel is latency-bound: reduce the per-thread tile or use `__launch_bounds__` |
| SHARED | static shared memory per block | uses the SM's 64 KB (T4) | must allow ≥ 2 blocks/SM if barriers stall |
| LOCAL | per-thread local memory | **should be 0**. Non-zero means **register spilling** or a register array indexed at run time | fix the indexing (`#pragma unroll`, compile-time sizes), or reduce the per-thread work |
| STACK | call stack | normally 0 | non-zero usually means non-inlined calls or spills |

**How registers limit occupancy (T4):** an SM has 65,536 registers and at most 1,024 threads
(32 warps). A block of 256 threads using R registers per thread needs 256·R registers:

```
blocks per SM (register limit) = floor(65536 / (256 · R))
R = 32  → 8 blocks (but the 1,024-thread limit allows only 4) → 100% occupancy
R = 64  → 4 blocks → 1,024 threads → 100%
R = 128 → 2 blocks →   512 threads →  50%
```

(Registers are actually allocated in chunks, so real limits can be slightly lower. Nsight
Compute's Occupancy section does the exact calculation.)

**Project-specific things to check:** v4's `acc[4][4]` and the attention kernel's `q[]`/`o[]`
arrays should show **LOCAL:0**. Otherwise the "register tile" is really in slow local memory.
Also compare REG of `layernorm_block_kernel<1024,16>` with `<256,4>`.

---

## 4. Nsight Systems: `bash profiling/nsys_timeline.sh all`

**Command:**

```bash
nsys profile --trace=cuda,nvtx --force-overwrite=true -o profiling/reports/timeline_all \
     ./build/profile_targets all --reps 5
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum profiling/reports/timeline_all.nsys-rep
```

- `--trace=cuda,nvtx`: record every CUDA runtime call on the CPU side, every kernel and copy on
  the GPU side, and our NVTX ranges. **NVTX** (NVIDIA Tools Extension) lets a program name time
  ranges; `profile_targets` wraps each target in a range with its label, e.g.
  `add+layernorm FUSED`.
- `--reps 5`: each target runs 5 times, so the first (cold) launch can be told apart from the
  steady state.

**What it measures:** timestamps, nothing inside the kernel. Its overhead is small, so the
timeline is close to real behavior, at boost clocks (unlike `ncu`).

### The timeline (open `.nsys-rep` in the Nsight Systems GUI on your PC)

```
CPU thread   ── cudaLaunchKernel ─┐ cudaDeviceSynchronize ──────────┐ cudaLaunchKernel …
NVTX         [ add+layernorm UNFUSED ............................... ]
GPU kernels                      └─[vector_add]─┐  [layernorm_block]─┘
                                                └gap┘
```

### What to look at, and what it means

| Look at | Report | Good | Bad | Decision |
|---|---|---|---|---|
| Kernel duration (avg, min, max) | `cuda_gpu_kern_sum` | max ≈ avg | first launch much slower (module load), or high variance (throttling) | exclude warm-up; rerun noisy measurements (the † row in 12_results §6) |
| Share of total GPU time per kernel | `cuda_gpu_kern_sum` "Time (%)" | – | – | **optimize the kernel with the largest share first** |
| Gaps between kernels | timeline | kernels back-to-back | GPU idle between short kernels | launch-bound: fuse kernels, CUDA Graphs (09 §3) |
| CPU time per `cudaLaunchKernel` | `cuda_api_sum` | a few µs | – | compare with kernel duration: a launch costing more than the kernel = launch-bound |
| `cudaMalloc` / `cudaMemcpy` time | `cuda_api_sum` | outside the hot loop | inside it | allocate once, keep data on the GPU (12_results §1: PCIe was 55× slower than the kernel) |
| NVTX range vs kernel time | `nvtx_sum` | range ≈ sum of kernels | range ≫ kernels | CPU-side overhead (launch + sync) dominates |

**Project-specific:**
- `add+layernorm UNFUSED` should show **two** kernels with a gap; `FUSED`, one.
- `attention UNFUSED` should show three kernels: compare their individual durations with
  `bench_attention`'s breakdown.
- `attention FUSED decode` is a single short kernel using only 12 blocks. Compare its duration
  with the launch API time.

---

## 5. Nsight Compute: `bash profiling/ncu_kernels.sh <case>`

**Command (run 1, sections):**

```bash
ncu --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    --section ComputeWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
    --section SourceCounters -o profiling/reports/ncu_gemm ./build/profile_targets gemm
ncu --import profiling/reports/ncu_gemm.ncu-rep --page details > profiling/reports/ncu_gemm_details.txt
```

`profile_targets` prints a numbered list ("kernel 3: gemm v2 coalesced (block 32x1)").
Nsight Compute numbers profiled kernels in the same order, so result ID 3 is that launch.

A **section** is a group of related metrics plus ncu's own analysis rules, which print hints
such as "uncoalesced global accesses" or "occupancy limited by registers". Read the hints, but
verify them with the numbers.

### 5.1 GPU Speed Of Light (SOL)

Two numbers: **Compute (SM) Throughput %** and **Memory Throughput %**, each as a percentage
of the hardware maximum for the busiest unit.

| SM % | Memory % | Diagnosis | Next step |
|---|---|---|---|
| low | **high (> ~60%)** | **memory-bound** | move fewer bytes: coalescing, reuse (tiling), fusion, lower precision |
| **high** | low | **compute-bound** | cheaper math: Tensor Cores, fewer instructions, FP16 |
| high | high | well balanced, near the roofline | only algorithmic changes help |
| **low** | **low** | **latency-bound**: neither unit is busy; warps are waiting | look at stall reasons (§5.6), occupancy (§5.3), parallelism (waves) |

The GUI also shows a **roofline chart** (03 §4) with the kernel's measured arithmetic intensity
and performance.

### 5.2 Launch Statistics

| Metric | Meaning |
|---|---|
| Grid size / block size | what we launched |
| Registers per thread | the same as `cuobjdump` REG |
| Shared memory per block | static + dynamic |
| **Waves per SM** | grid blocks ÷ (blocks that fit at once on all SMs) |

**Waves:** if 40 SMs can hold 4 blocks each (160 at once) and the grid has 200 blocks, that's
1.25 waves. The last 0.25 wave runs with most SMs idle (the "tail effect"). Few waves (< 1),
like GEMM v4 at n = 128 (4 blocks for 40 SMs), means most of the GPU does nothing.

### 5.3 Occupancy

- **Theoretical occupancy**: the maximum active warps per SM ÷ the hardware maximum (32 on T4),
  given the limits from registers, shared memory, block size and blocks per SM. The section
  names the limiting factor ("Block Limit Registers", "Block Limit Shared Mem",
  "Block Limit SM"…).
- **Achieved occupancy** (`sm__warps_active.avg.pct_of_peak_sustained_active`): the
  measured average.

| Situation | Meaning | Decision |
|---|---|---|
| achieved ≈ theoretical, both high | fine | look elsewhere |
| theoretical low, kernel latency-bound | not enough warps to hide latency | fewer registers/shared memory per block, or smaller blocks |
| theoretical low, kernel already near SOL | **no problem**: occupancy is a means, not a goal | leave it (v4 trades occupancy for reuse, 04 §22) |
| achieved ≪ theoretical | imbalance, tail effect, or too few blocks | more, smaller work units; check waves |

**Open question 1** (GEMM v2: 32×1 fastest) is exactly the case where the theoretical
occupancy argument didn't predict performance. Compare achieved occupancy, L1 hit rate and
stall reasons between the 32×8 and 32×1 launches.

### 5.4 Memory Workload Analysis

The **memory chart** shows bytes flowing kernel → L1 → L2 → DRAM, with hit rates.

| Metric | Meaning | Good | Bad → action |
|---|---|---|---|
| **DRAM throughput %** | bandwidth used vs peak | memory-bound kernels: 70–90% | low with a memory-bound kernel → not enough loads in flight (latency-bound) |
| DRAM bytes read / written | the real traffic | ≈ the minimum the algorithm needs | far above it → no reuse, or wasted sector bytes |
| **L1 hit rate**, **L2 hit rate** | % of requests served by the cache | high when data is reused | – |
| **Sectors per request** (global loads) | sectors ÷ requests | **4** for 32 consecutive floats (§03 §2); **1** for a broadcast | **up to 32** = fully uncoalesced → fix the thread → data mapping (GEMM v1) |
| **Shared bank conflicts** | extra shared-memory wavefronts | **0** | > 0 → pad arrays (`[T][T+1]`) or change the access pattern (04 §18, 07 §5) |

### 5.5 Compute Workload Analysis

Utilization of each **pipeline** (a pipe is a group of execution units for one instruction
type):

| Pipe | Executes | Busy in our kernels |
|---|---|---|
| FMA | FP32 multiply-add | GEMM (FP32 versions) |
| ALU | integer and logic ops | index arithmetic |
| LSU | load/store instructions (global, shared, local) | everything; dominant in v3 GEMM |
| XU (special function unit) | `expf`, `rsqrtf`, … | softmax, attention, LayerNorm |
| Tensor (HMMA) | Tensor Core matrix ops | v5 WMMA only |

A pipe near 100% means the kernel is bound by that instruction type. Example to check: if
softmax v4 shows a higher XU utilization than v3, that supports the explanation in 12_results §4
(extra `expf` work).

### 5.6 Warp State Statistics: why are warps waiting?

Every cycle, each warp is either issuing an instruction or **stalled** for a reason. This
section shows the average number of cycles per instruction each warp spent in each state. It is
the most direct answer to "what is this kernel waiting for?".

| Stall reason | Meaning | Typical cause here | Fix |
|---|---|---|---|
| **Long Scoreboard** | waiting for a global/local memory load (L1TEX) | memory-bound kernels; uncoalesced loads; spills | coalescing, reuse via shared memory/registers, more loads in flight |
| **Short Scoreboard** | waiting for shared memory or special-function results | tiled GEMM, shuffles, `expf` | register tiling (v4), fewer shared loads per FMA |
| **MIO Throttle** | the shared-memory/special-function instruction queue is full | very frequent shared loads or shuffles | fewer, wider (vectorized) shared accesses |
| **Barrier** | waiting at `__syncthreads()` for other warps | tiled kernels, block reductions | more work between barriers; fewer reduction levels |
| **Math Pipe Throttle** | the needed pipe is busy | compute-bound | good sign for GEMM; next step is Tensor Cores |
| **Not Selected** | ready, but another warp was chosen | plenty of ready warps | **good**: the scheduler has spare work |
| **Wait** | fixed-latency dependency on the previous instruction | dependent arithmetic chains | instruction-level parallelism (independent accumulators, unrolling) |
| **No Instruction** / **Branch Resolving** | instruction fetch or divergence | large unrolled code, divergent `if` | – |

### 5.7 Scheduler Statistics

"Eligible warps per scheduler" and "Issued warp per scheduler". The SM has 4 schedulers, and
each can issue at most one instruction per cycle. If "No Eligible" is high (cycles in which no
warp was ready), the kernel is latency-bound. Combine this with §5.6 to see why.

### 5.8 Source Counters

Per **source line** (thanks to `-lineinfo`): which lines have the most stalls, and warnings like
"uncoalesced global access" with the line number. Open the report in the GUI's *Source* page to
see the `.cu` code next to SASS (GPU assembly).

---

## 6. The metrics CSV (`ncu_<case>_metrics.csv`)

Run 2 of the script collects a fixed list of metrics, so that numbers can be quoted precisely.
Names follow `unit__counter.rollup`: `dram__bytes_read.sum` = the sum over all DRAM units of
bytes read.

| Metric | Use |
|---|---|
| `gpu__time_duration.sum` | kernel time at the (locked) profiling clock |
| `sm__cycles_elapsed.avg.per_second` | **the actual SM clock**, to relate ncu times to benchmark times |
| `sm__throughput…pct`, `gpu__compute_memory_throughput…pct` | the two SOL numbers |
| `dram__throughput…pct`, `dram__bytes_read.sum`, `dram__bytes_write.sum` | DRAM usage; **bytes per element = (read + write) / elements** |
| `lts__t_sector_hit_rate.pct`, `l1tex__t_sector_hit_rate.pct` | L2 / L1 hit rates |
| `l1tex__t_requests…global_op_ld.sum`, `l1tex__t_sectors…global_op_ld.sum` | **sectors per request = sectors / requests** (coalescing quality) |
| `l1tex__data_bank_conflicts…shared_op_ld/st.sum` | shared-memory bank conflicts |
| `sm__warps_active…pct` | achieved occupancy |
| `launch__registers_per_thread`, `launch__occupancy_limit_*`, `launch__waves_per_multiprocessor` | resource limits and waves |

If a metric name isn't available on a particular GPU, ncu reports an error in the CSV. The
script warns and continues.

---

## 7. Bottleneck identification: the decision table used in this project

| Symptom (metrics) | Diagnosis | Optimization (where it appears in this project) |
|---|---|---|
| sectors/request ≫ 4, high Long Scoreboard | uncoalesced global access | change the thread → data mapping (GEMM v1 → v2) |
| DRAM/L2 bytes ≫ algorithmic minimum, Long Scoreboard | no data reuse | shared-memory tiling (v2 → v3) |
| LSU/shared pipe busy, Short Scoreboard / MIO Throttle | shared-memory bandwidth bound | register tiling (v3 → v4) |
| Barrier stalls high | too much synchronization per unit of work | warp shuffles instead of shared-memory trees (softmax v2 → v3), hierarchical reduction (LayerNorm `block_reduce_sum`) |
| DRAM throughput 70–90%, memory-bound | at the hardware limit | move fewer bytes: read once (LayerNorm v3), fusion, FP16 |
| FMA pipe near peak | compute-bound on CUDA cores | Tensor Cores (v5 WMMA) |
| Tensor pipe low, Long Scoreboard high (in a Tensor Core kernel) | Tensor Cores starved of data | shared-memory staging, fragment reuse (what cuBLAS does) |
| bank conflicts > 0 | conflicting shared-memory addresses | padding (`Bs[32][33]` in attention) |
| LOCAL > 0 | register spills | compile-time indexing, smaller per-thread tiles |
| few waves, many idle SMs | not enough parallel work | smaller tiles, split rows across blocks (softmax 1 × 50,257) |
| GPU idle between kernels (nsys) | launch-bound | fusion, CUDA Graphs |

---

## 8. Hypotheses to test with your reports

Each hypothesis is a **prediction**. §9 records whether it held.

**GEMM (`ncu_kernels.sh gemm`, `precision`)**
- H1. v1: global-load **sectors/request far above 4** (strided A loads, 04 §5); dominant stall
  Long Scoreboard; L1 hit rate high (the cache partly rescues it).
- H2. v2 (32×8): sectors/request ≈ 1–4 (broadcast A, coalesced B); still Long Scoreboard-bound;
  DRAM bytes far above the 12 MB minimum (no reuse).
- H3. **Open question 1:** 32×1 vs 32×8. 32×1 has lower *theoretical* occupancy (the
  16-blocks/SM limit), but faster runtime. Hypothesis: a better L1 hit rate (fewer warps
  competing for L1), visible in `l1tex__t_sector_hit_rate`. *To be checked; not assumed.*
- H4. v3: **0 bank conflicts** (04 §18); stalls dominated by Short Scoreboard/MIO and Barrier;
  LSU busier than FMA.
- H5. v4 (**open question 3**): LOCAL = 0 (no spills); FMA pipe utilization much higher than v3;
  fewer shared loads per FMA. The remaining gap to cuBLAS should show up as a remaining Short
  Scoreboard/MIO share, or issue-slot limits.
- H6. WMMA (**open question 4**): Tensor pipe utilization **low**; Long Scoreboard dominant;
  L2/DRAM traffic high. In other words, Tensor Cores starved of data.

**Softmax (`softmax`)**
- H7. **Open question 2:** at 16,384 × 1,024, v2 (block/row) achieves higher DRAM throughput than
  v3 (warp/row). Expected evidence: v3 has more Long Scoreboard stalls per instruction (each
  warp has few loads in flight while it loops 32 times per pass). At 65,536 × 256, the order
  reverses.
- H8. v4 shows higher XU (special function) utilization than v3 (extra `expf` on rescaling).

**LayerNorm and fusion (`layernorm`)**
- H9. v3: DRAM bytes ≈ 8 B/element (8192 × 4096 = 33.55 M elements → ≈ 268 MB); v2 more
  (re-reads).
- H10. **Fusion, measured in bytes:** unfused (vector_add + v3) ≈ 20 B/element ≈ 671 MB in total;
  fused ≈ 16 B/element ≈ 537 MB. This checks the 1.25× byte-count argument directly in
  hardware counters.

**Attention (`attention`)**
- H11. QKᵀ batched GEMM: **0 shared-memory store bank conflicts** thanks to `Bs[32][33]` padding
  (07 §5).
- H12. Fused kernel: low FMA utilization, many Short Scoreboard/MIO stalls (5 shuffles + 2 `expf`
  per key), which explains why it is slower than unfused for non-causal attention (12_results §6).
- H13. Decode (q_len = 1): very few waves (12 blocks for 40 SMs); low SM and memory throughput,
  i.e. latency/parallelism-bound, as described in 07 §8.

---

## 9. Measured results (Tesla T4, Nsight Compute 2025.3.1, CUDA 13.0)

Source: [`profiling/reports/`](../profiling/reports/), i.e. `resource_usage.txt`,
`timeline_all_stats.txt`, `ncu_<case>_details.txt` and `ncu_<case>_metrics.csv`. The CSV
tables can be pivoted to one row per kernel with `node profiling/pivot_metrics.js <csv>`.
Nsight Compute ran at the locked base clock (**585 MHz**, `sm__cycles_elapsed.avg.per_second`)
with caches flushed, so its durations are about 2.5× the benchmark durations. Compare
percentages and byte counts, not milliseconds.

### 9.1 Resource usage (cuobjdump): no spills anywhere

**Every kernel has `LOCAL:0`**: no register spills, and every register-tile array (`acc[4][4]`,
`a_reg`, `b_reg`, LayerNorm's `vals[]`, attention's `q[]`/`o[]`) really lives in registers.
Selected values:

| Kernel | REG | SHARED (bytes) |
|---|---|---|
| matmul_naive / coalesced | 52 | 0 |
| matmul_tiled<32> | 42 | 8,192 |
| **matmul_register** | **72** | 4,096 |
| matmul_wmma | 64 | 0 |
| softmax_warp / online | 46 / 20 | 0 |
| layernorm_block<512,8> | 33 | 64 |
| attention_fused<64> | 35 | 16,384 |
| batched_gemm<32> | 46 | 8,320 (= 32·32 + 32·33 floats: the padded tile) |

### 9.2 Nsight Systems (timeline, boost clocks)

- **Clock throttling is real on the Colab T4.** Five identical launches of GEMM v1 took
  between **36.8 ms and 92.5 ms** (median 69.1 ms). Long kernels slow down as the GPU hits its
  power and thermal limits.
- That explains the † row in 12_results §6. In the timeline, attention at seq 1024 took
  **4.12 ms unfused** and **5.08 ms fused** (NVTX medians), not the 9.25 / 12.05 ms of the
  outlier benchmark row. The ratio (fused slower for non-causal) is unchanged.
- First launches are slower: the first `vector_add` launch took 3.17 ms vs a median of 0.80 ms
  (module loading). This is why every benchmark warms up first.
- `cudaLaunchKernel`: median 15.9 µs per call (min 6.2 µs) in this program, where every launch
  is followed by a synchronization. For a kernel shorter than that, the launch costs more than
  the work.

### 9.3 Hypotheses

| # | Hypothesis | Measured | Verdict |
|---|---|---|---|
| H1 | v1: sectors/request ≫ 4, high L1 hit rate, memory stalls | **16.50 sectors/request**: exactly (32 + 1)/2, the 04 §5 prediction for strided A + broadcast B. ncu: "85% of sectors excessive (uncoalesced)". L1 hit **98.0%**. 445 cycles per issued instruction, dominated by **LG Throttle** (435 cycles: the global-memory instruction queue is full). DRAM read 433 MB for 8 MB of inputs | ✔ (the stall is LG Throttle rather than Long Scoreboard, but in the same memory family) |
| H2 | v2: sectors/request 1–4, DRAM ≫ 12 MB, memory-limited | **2.50 sectors/request** = (1 + 4)/2 exactly. DRAM read 199 MB. L1/TEX throughput 81% (the busiest unit) | ✔ v2 is bound by L1 traffic, not DRAM |
| H3 | 32×1 beats 32×8 via a **better L1** hit rate | L1 hit **19.8% vs 87.4%** (worse, not better). But L2 hit **98.1% vs 80.7%**, DRAM reads **129 vs 199 MB** (−35%), and cycles per issued instruction **16.8 vs 33.3** (half the LG-queue stalls: 9.2 vs 24.8 cycles). Durations at base clock: **equal** (7.10 vs 7.07 ms) | ✘ as stated. The real difference is **less DRAM traffic and less queue contention with half the warps**. At base clock that's a tie; at boost clock (compute 2.7× faster, DRAM unchanged), the lower DRAM traffic wins (the +13% in the benchmark). Occupancy halved (49.7% vs 98.7%) without hurting. |
| H4 | v3: 0 bank conflicts, shared-memory stalls | **0** load and **0** store bank conflicts. **4.00 sectors/request** (perfect coalescing). Dominant stall **MIO Throttle** (24.4 cycles: shared-memory queue full). DRAM read 147 MB | ✔ |
| H5 | v4: no spills, much better FMA use; the gap to cuBLAS visible | LOCAL 0. **0 bank conflicts** (the strided 4×4 ownership works). FMA = busiest pipe at **41.1%**. Cycles per instruction **12.95** (v3: 37.9). DRAM read **32.9 MB** (v3: 147 MB, 4.5× less). Remaining limits: (a) **75% theoretical occupancy, limited by 72 registers/thread** (3 blocks/SM); (b) **2.13 waves**: the last partial wave may cost up to 33% (ncu estimate); (c) only 0.86 eligible warps per scheduler (latency-bound) | ✔ The gap to cuBLAS is explained by the tail effect, the register-limited occupancy, and latency |
| H6 | WMMA: Tensor Cores starved, memory stalls | Issue slots busy **5.5%**, **124 cycles per instruction**, of which **110.6 in LG Throttle**. SM throughput 17%. "50% excessive sectors" (16-element half rows = 32-byte pieces) | ✔ Tensor Cores idle waiting for global loads |
| H7 | softmax 16384×1024: v2 higher DRAM throughput than v3; v3 more Long Scoreboard | v3 waits on memory (Long Scoreboard 32.7 cycles/instruction) ✔. But **v3 has the higher DRAM throughput (86.5% vs 49.7%)** because it **reads x 2.77× from DRAM** (186 MB for a 67 MB input) vs v2's **1.22×** (81.7 MB). With one warp per row, ~32 rows × 4 KB per SM are in flight across 40 SMs (≈ 5 MB), more than the 4 MB L2, so rows are evicted between the 3 passes. v2 keeps few rows in flight, so passes 2–3 hit L1/L2 (L1 hit 41%) | ✘ / ✔ (the mechanism is **re-reading from DRAM**, not DRAM throughput). At base clock both take ~1.0 ms; at boost, v2's lower traffic wins (1.8× in the benchmark) |
| H8 | softmax v4: higher special-function (XU) use than v3 | XU never appears as the busiest pipe. v4 issues more instructions (issue slots 29.1% vs 19.4%), and **reads as much DRAM as v3** (199 vs 186 MB): its saved pass didn't save DRAM traffic | ~ (XU share not visible in the text export; the "no DRAM saving" part explains 12_results §4) |
| H9 | LayerNorm v3 ≈ 8 B/element; v2 more | v3: 148 MB read + 150 MB written = **8.88 B/element** (ideal 8). v2: **19.8 B/element** (re-reads), stalled 98.7 cycles/instruction on memory, issue slots 6.9% | ✔ |
| H10 | fusion: unfused ≈ 20, fused ≈ 16 B/element | Unfused (vector_add + v3): **740 MB = 22.05 B/element**. Fused: **608 MB = 18.12 B/element**. Ratio **1.22**: the benchmark speedup was 1.21× | ✔ **The fusion speedup is fully explained by DRAM bytes** (both ~10% above the ideal counts, from write overhead) |
| H11 | batched GEMM with `Bs[32][33]`: 0 store conflicts | **0** load and **0** store bank conflicts for the transposed (QKᵀ) variant | ✔ the padding works |
| H12 | fused attention: low FMA use, shared/shuffle stalls | **FMA is the busiest pipe at 60.3%**, issue slots **59.7%** busy, 12.8 cycles/instruction, DRAM 0.4% | ✘ as stated: the kernel is **instruction-issue-bound** (busy, not starved). It executes many more instructions per useful FLOP than the GEMMs (per-key dot products + shuffles + rescaling, one key at a time). Same conclusion as 07 §6: it needs GEMM-style tiles / Tensor Cores |
| H13 | decode (q_len = 1): few waves, latency-bound | **0.07 waves** (12 blocks for 40 SMs), achieved occupancy **25%**, SM 3.3%, DRAM 4.9%. Dominant stall: **Barrier** (14.0 cycles): the 7 inactive warps of each block wait at `__syncthreads()` while one warp works | ✔ the GPU is almost empty; split the keys across blocks (flash-decoding) |

### 9.4 One unexpected measurement

**Shared-memory store "bank conflicts" in the fused attention kernel:** 675,893 (non-causal)
and 382,510 (causal; ncu: "10.23% of the 3,737,514 shared store wavefronts"). The stores
`Ks[r][c]`/`Vs[r][c]` are written by 32 lanes to 32 consecutive words, which by address analysis
(04 §18) is conflict-free, and the identical pattern in other kernels measured 0. **The cause is
not yet determined.** The next step is the Source page in the Nsight Compute GUI
(`ncu_attention.ncu-rep`), which attributes excess shared-memory wavefronts to individual SASS
instructions. The cost is bounded: about 10% extra store wavefronts in a kernel whose
bottleneck is instruction issue (H12).

### 9.5 What profiling changed in our understanding

1. **Each GEMM step was confirmed by the counters:** sectors/request 16.5 → 2.5 → 4.0 (coalesced
   and staged); DRAM reads 433 → 199 → 147 → 33 MB; stall reason LG Throttle → MIO Throttle →
   (none dominant, FMA busiest). That's one bottleneck removed per version.
2. **Occupancy is not the goal:** v2 at 49.7% occupancy matched v2 at 98.7%; v4 at 66% is the
   fastest FP32 kernel.
3. **Cache capacity explains two "surprises":** v2 32×1 (less DRAM traffic) and softmax v2 vs v3
   (whether the rows survive in L2 between passes).
4. **Fusion's benefit can be measured in bytes:** 22.05 → 18.12 B/element = 1.22×, matching the
   1.21× speedup.
5. **Benchmarks on shared cloud GPUs need repeats:** a 2.5× clock-throttling spread on one kernel.

### 9.6 Next optimizations, chosen from these measurements

| Kernel | Bottleneck (measured) | Next step |
|---|---|---|
| GEMM v4 | 2.13 waves (tail), 75% occupancy (72 registers) | tile sizes that give whole waves on 40 SMs; `__launch_bounds__(256, 4)` to cap registers |
| GEMM v5 WMMA | LG Throttle 110 cycles/instruction | stage A/B tiles in shared memory and reuse fragments across warps |
| softmax (long rows) | DRAM re-reads with warp-per-row | keep the row in registers (like LayerNorm v3) so x is read once |
| fused attention | instruction issue | process tiles of queries × keys as small GEMMs (FlashAttention style, Tensor Cores) |
| attention decode | 0.07 waves | split keys across blocks and merge (m, l, o) partials (flash-decoding) |

---

## 10. Common mistakes

1. Profiling a Debug build.
2. Comparing `ncu` durations with benchmark durations. `ncu` locks clocks and flushes caches.
3. Profiling hundreds of launches. Use a driver like `profile_targets` and filters.
4. Treating occupancy as the goal. It's a tool for hiding latency (H3, H5).
5. Reading `nvidia-smi`'s "GPU-Util" as SM utilization. It only means "a kernel was running
   during the sample period", even if it used one SM.
6. Including first launches (module loading, clock ramp-up) in timeline conclusions.
7. Trusting the analysis hints without checking the numbers behind them.
8. Changing several things at once after profiling. Change one, re-measure, re-profile.

## 11. Interview explanation

> "I use three levels. `cuobjdump` resource usage gives registers, shared memory and spills per
> kernel without running anything. Nsight Systems gives the timeline: which kernel dominates,
> gaps between kernels, launch overhead, copies. That's how I'd spot a launch-bound pipeline.
> Then Nsight Compute on the specific kernel. I start at Speed of Light to classify it as
> memory-bound, compute-bound or latency-bound, then use memory workload analysis for sectors
> per request, hit rates, DRAM bytes and bank conflicts, and warp state statistics to see what
> warps are stalled on: long scoreboard for global memory, short scoreboard or MIO for shared
> memory, barrier for `__syncthreads`. Each stall reason maps to an optimization: uncoalesced
> loads led to the remapping in GEMM v2, missing reuse led to tiling, shared-memory pressure led
> to register tiling, barrier stalls led to warp shuffles. I'm aware that ncu locks clocks and
> flushes caches, so I compare percentages rather than its milliseconds. And I used DRAM byte
> counters to verify my fusion argument directly: unfused should move about 20 bytes per element
> and fused about 16."

## 12. What to remember

- nsys = *where* the time goes (timeline). ncu = *why* a kernel is slow (counters).
  cuobjdump = resource usage.
- Speed of Light first: memory-bound, compute-bound, or latency-bound.
- Sectors/request: 4 is ideal for consecutive floats, 32 is the worst case.
- Stall reasons map to fixes: Long Scoreboard → memory; Short Scoreboard/MIO → shared/SFU;
  Barrier → synchronization.
- `ncu` uses base clocks and cold caches: compare percentages.
- Occupancy is a means, not a goal.

---

## 13. File guide

### `src/benchmark/profile_targets.cu`
1. **What:** a driver that launches each kernel of interest once (or `--reps N` times) at a fixed
shape. 2. **Why:** profilers need a short, predictable list of launches. 3. **Inputs:** a case
name and `--reps`. 4. **Output:** a numbered list of launches (= ncu result IDs) on stdout.
5. **Data flow:** random inputs → each target → synchronize. 6. **Functions:** `target`,
`random_buffer`, `case_*`. 7. **Variables:** `g_reps`, `g_kernel_index`. 8. **Concepts:** NVTX
ranges (`nvtxRangePushA`/`nvtxRangePop`), optional at build time via `CTP_HAVE_NVTX`.
9.–12. Not a performance measurement itself: it only feeds the profilers. 13. **Mistakes:**
profiling it with `--reps > 1` under ncu (multiplies the replay time).

### `profiling/*.sh`, `notebooks/colab_profile.ipynb`
§3–§5. All scripts stop on errors (`set -euo pipefail`) and write into `profiling/reports/`.
The binary `.ncu-rep`/`.nsys-rep` reports are excluded from git (`.gitignore`); their text and
CSV exports are kept.
