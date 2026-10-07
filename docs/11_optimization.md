# 11 — The Optimization Story: Bottleneck → Evidence → Change → Result

This document ties the project together. For every kernel version it states the bottleneck,
the **evidence** (benchmark or profiler counter), the change made, and the **measured** result.
It includes the changes that did *not* help, because explaining those is part of the skill.

All numbers: Tesla T4, CUDA 13.0, FP32 unless noted. Sources:
[12_results.md](12_results.md) (benchmarks) and [10_nsight_profiling.md §9](10_nsight_profiling.md)
(profiler counters, measured at the profiler's locked 585 MHz base clock).

---

## 1. The method

```
1. Prove correctness (tests vs CPU reference; the run stops on any failure)
2. Measure (CUDA events, warm-up, averaged iterations)
3. Classify with the roofline: memory-bound or compute-bound?   (03 §4)
4. Form a hypothesis about the bottleneck
5. Check it with counters (Nsight Compute / Systems)
6. Change ONE thing, re-test, re-measure, re-profile
```

Step 5 overturned the hypothesis three times in this project (§7). That's the reason to profile
instead of guessing.

---

## 2. GEMM (1024³): 61 → 2,236 GFLOP/s

| Step | Bottleneck | Evidence | Change | Result |
|---|---|---|---|---|
| v1 naive | uncoalesced loads: a warp's 32 threads read 32 different rows of A | **16.5 sectors/request** (ideal 4); "85% of sectors excessive"; 445 cycles per instruction waiting on the global-memory queue; DRAM reads **433 MB** for 8 MB of input | – | 61.2 GFLOP/s |
| v2 coalesced | (fix) map `threadIdx.x` to columns | sectors/request **2.5** (= (1 broadcast + 4 coalesced)/2) | swap the row/col mapping, nothing else | **582.1 GFLOP/s (9.5×)** |
| v2 → v3 | no data reuse: every thread re-reads A and B through L1 | DRAM 199 MB; L1 the busiest unit (81%) | 32×32 shared-memory tiles, 2 barriers per tile | **867.4 (1.5×)**; DRAM 147 MB |
| v3 → v4 | shared-memory bandwidth: 2 shared loads per FMA | dominant stall **MIO Throttle** (shared-memory queue full, 24.4 cycles/instruction); 0 bank conflicts | each thread computes a 4×4 outer product from registers (0.5 loads per FMA) | **2,235.6 (2.6×)**; DRAM 33 MB; cycles/instruction 37.9 → 12.95 |
| v4 today | tail effect + register-limited occupancy + latency | 2.13 waves (ncu: up to 33% lost); 72 registers → 75% theoretical occupancy; 0.86 eligible warps/scheduler; FMA busiest at 41% | (next) whole-wave tile sizes, `__launch_bounds__` | **60% of cuBLAS FP32** (3,729.9) |
| v5 WMMA (FP16) | Tensor Cores starved of data | 5.5% issue slots busy; 110.6 of 124 cycles/instruction waiting on global loads | (next) stage tiles in shared memory, reuse fragments | 3,289.5 GFLOP/s: **1.49× v4**, but only ~9% of cuBLAS FP16 |

**Size matters:** at n = 128, v2 was fastest and v4 was 2.8× slower than v2 (4 blocks of 64×64
for 40 SMs). **Shape matters:** for decode (M = 1), v2 reached 205–224 GB/s (86–92% of cuBLAS)
while v3/v4 wasted almost every tile row. **There is no single best kernel**, which is why
libraries choose a kernel per shape.

---

## 3. Softmax and LayerNorm (memory-bound): judged in GB/s

| Kernel | Bottleneck | Evidence | Change | Result |
|---|---|---|---|---|
| softmax v1 (thread/row) | uncoalesced + too few threads | ≈ 32 GB/s (10% of peak) | – | baseline |
| softmax v2 (block/row) | – | reads x **1.22×** from DRAM (passes 2–3 hit cache) | block-stride loop + shared-memory tree | **243.2 GB/s at 12288×1024 (76% of peak), 12% faster than PyTorch** |
| softmax v3 (warp/row) | for long rows: **re-reads x from DRAM** | 186 MB read for a 67 MB input (**2.77×**): ~5 MB of rows in flight > 4 MB L2 | (none needed for short rows) | wins for ≤ 256 columns (227 vs 107 GB/s); loses 1.8× at 1,024 columns |
| softmax v4 (online) | more work per element, same DRAM traffic | 199 MB DRAM (no saving); more instructions issued | – | **not faster than v3**: a negative result (§7) |
| LayerNorm v1 → v3 | v2 re-reads x 3 times (19.8 B/element) | DRAM counters | keep the row in **registers**, read x once; hierarchical block reduction (2 barriers) | **8.90 B/element** (ideal 8); 230.1 GB/s at 8192×4096 (72% of peak), **1.38× PyTorch** |
| add + LayerNorm, fused | intermediate h written then read back | unfused **22.07 B/element** (740 MB) | compute h in registers, write it once | **18.11 B/element** (608 MB) → ratio 1.22; **measured 1.21×**; **1.5× PyTorch eager** |

---

## 4. Attention

| Version | Bottleneck | Evidence | Result |
|---|---|---|---|
| unfused (3 kernels) | seq² score matrix in DRAM | 192 MB workspace at seq 2048 (ours stores S once and runs softmax in place; PyTorch naive: 390 MB) | 18.4 ms at seq 2048; PyTorch's cuBLAS-based naive path is 2.4× faster |
| fused (online softmax) | **instruction issue**, not memory | FMA busiest pipe at 60%, issue slots 60% busy, DRAM 0.4% | no score buffer; **slower non-causal** (5.08 vs 4.12 ms at seq 1024), **1.2× faster causal** (skips future keys) |
| decode, q_len = 1 | **not enough parallel work** | 0.07 waves (12 blocks for 40 SMs), 25% occupancy, warps waiting at barriers | KV cache step **30× cheaper** than recomputing at 2,048 tokens of context; next step: split keys across blocks |

---

## 5. Why inference performance depends on each factor: evidence from this project

| Factor | What we measured | Inference consequence |
|---|---|---|
| **Memory movement** | Fusion: 22.07 → 18.11 bytes/element gave 1.21×. LayerNorm v3 vs v2: 8.90 vs 19.8 bytes/element | Memory-bound ops (norms, softmax, residuals, decode) speed up only by moving fewer bytes: fusion, read-once kernels, lower precision |
| **Compute** | GEMM 61 → 2,236 GFLOP/s through reuse; Tensor Cores 1.49× (ours) and cuBLAS FP16 at 35.9 TFLOP/s vs 3.7 FP32 | Prefill is dominated by GEMMs; Tensor Cores + FP16/BF16 are the main lever |
| **Kernel launch overhead** | vector add: a 2.7 µs floor regardless of size; `cudaLaunchKernel` median 15.9 µs in the profiled program; PyTorch small ops: GPU time ≈ wall time (launch-bound) | Decode runs hundreds of tiny kernels per token. Fusion and CUDA Graphs cut launches |
| **Parallelism** | decode attention: 0.07 waves, 3% SM throughput; softmax of one 50,257-wide row: 7.2 GB/s; GEMM v4 at n = 128: 2.8× slower than v2 | Batch size 1 decoding leaves the GPU mostly idle. Batching requests and split-K/flash-decoding create parallelism |
| **Precision** | FP16 halves bytes; FP16 accumulation error 3.6e-2 vs 4.4e-6 (FP32) at K = 16,384, and overflow to inf | FP16/BF16 storage + FP32 accumulation is the standard; quantization (INT8/INT4) pushes the bytes down further for memory-bound decode |
| **Cache** | rows that fit in the 4 MB L2 ran above "100% of DRAM peak"; softmax v3 re-read rows from DRAM once its working set exceeded L2; v2 GEMM 32×1 used 35% less DRAM | Working-set size relative to L2 decides whether re-reads are free. Tiling and kernel design should keep reuse inside the cache |
| **Occupancy** | v2 GEMM at 49.7% occupancy matched 98.7%; v4 at 66% was the fastest FP32 kernel; but vector add with 32-thread blocks (50% occupancy) lost 15% | Occupancy hides latency when a kernel is latency-bound; it is not a goal in itself |
| **Bandwidth** | vector add 82%, LayerNorm 72%, softmax 76% of peak; decode GEMM 205–224 GB/s | Decode tokens/second is roughly bandwidth ÷ bytes per token (weights + KV cache): this is why quantization and KV-cache compression matter |

---

## 6. From kernel to inference performance

```
CUDA kernel            matmul_register_kernel, softmax_block_kernel, add_layernorm_kernel, attention_fused_kernel
   ↓ runs as
GPU execution          blocks on 40 SMs; warps stalled on memory, shared memory, or issue (measured stall reasons)
   ↓ implements
AI operation           GEMM, softmax, LayerNorm, residual add, attention
   ↓ forms
Transformer layer      QKV projection → attention → output projection → add+norm → MLP (2 GEMMs) → add+norm
   ↓ determines
Inference performance  prefill (many tokens):  GEMM-dominated, compute-bound  → Tensor Cores, tiling
                       decode (one token):     GEMV + KV-cache reads, memory-bound, launch-heavy
                                               → bandwidth, quantization, fusion, KV cache, batching
```

Measured instances of each regime: prefill-shaped GEMMs reached 1,979–2,194 GFLOP/s with v4
(compute side); decode-shaped GEMMs were limited to ~205–224 GB/s by bandwidth, and decode
attention by parallelism.

---

## 7. Negative and surprising results (and what they taught)

| Expectation | What happened | Lesson |
|---|---|---|
| Online softmax is faster (fewer passes) | equal or slower; the same DRAM bytes | fewer passes only help if the saved pass would hit DRAM. Online softmax's value is enabling fusion |
| Warp-per-row softmax wins for typical rows | block-per-row won at ≥ 1,024 columns | L2 capacity vs rows in flight decides it (2.77× vs 1.22× DRAM reads) |
| 32×1 blocks lose (low occupancy) | fastest v2 shape | less DRAM traffic beat occupancy |
| Fusion helps small shapes most | 1.04× at 512×768 | the intermediate stayed in L2 anyway |
| Fused attention is faster | slower for non-causal | fusion removes bytes, not instructions; the kernel was issue-bound |
| Grid-stride is "the" pattern | slower for a large vector add (160 blocks) | too few loads in flight to saturate DRAM |
| FP16 on CUDA cores ≈ 2× | ≈ FP32 tiled | the math still runs as FP32 FMAs; the speedup needs Tensor Cores |
| PyTorch SDPA saves memory | not for FP32 on the T4 (438 MB) | measure library behavior; backends depend on dtype and GPU |

**Still unexplained:** ~10% extra shared-memory store wavefronts in the fused attention kernel
(10 §9.4). The next step is the per-instruction Source view.

---

## 8. What would be done next (prioritized by measured impact)

1. **GEMM v5:** shared-memory staging + fragment reuse for WMMA (currently 9% of cuBLAS FP16;
   the largest headroom in the project).
2. **FlashAttention-style tiles:** query × key tiles as small Tensor Core GEMMs (the fused kernel
   is issue-bound).
3. **Flash-decoding:** split the KV cache across blocks for q_len = 1 (0.07 waves today).
4. **GEMM v4:** whole-wave tiling and register capping (2.13 waves, 75% occupancy).
5. **Softmax for long rows:** read once into registers, as in LayerNorm v3.
6. **CUDA Graphs** for the multi-kernel pipelines (launch-bound at small sizes).
