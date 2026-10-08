# 11 — The Optimization Story: Bottleneck → Evidence → Change → Result

This document ties the project together. For every kernel version it states the bottleneck,
the **evidence** (benchmark or profiler counter), the change made, and the **measured** result.
It includes the changes that did *not* help, because explaining those is part of the skill, and
ends with the challenges faced along the way and how they were solved (§9).

All numbers: Tesla T4, CUDA 13.0, FP32 unless noted. Sources:
[12_results.md](12_results.md) (benchmarks) and [10_nsight_profiling.md §9](10_nsight_profiling.md)
(profiler counters, measured at the profiler's locked 585 MHz base clock). §4b (INT8
quantization) was measured on an A100: [12_results.md §9.1–9.2](12_results.md#91-int8-weight-only-quantization-decode-gemv-docs08-9).

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

## 4b. INT8 weight-only quantization: decode GEMV (A100, 7B-class MLP up, 11008 × 4096)

One decoding step is y = W x with one token: no reuse of W, so time should follow the bytes of W.
Weights of 45–180 MB are larger than the A100's 40 MB L2, so these rows measure DRAM.

| Step | Bottleneck | Evidence | Change | Result |
|---|---|---|---|---|
| FP32 weights | DRAM (as intended) | DRAM 89%, L1/TEX 21% | – | 0.1355 ms, 86% of peak |
| FP16 weights | DRAM | DRAM 86%, L1/TEX 51% | 2 bytes per weight | **1.93×** (prediction: 2×) |
| INT8 weights, one row per warp | **L1, not DRAM** | DRAM read exactly 45.2 MB (1 B/weight) but DRAM only 49% busy; **L1/TEX 94%**; 12.7 M L1 sectors for 45 MB of weights | 1 byte per weight + one FP32 scale per row | **2.20×**, not the predicted 4× |
| INT8, 4 rows per warp | **parallelism** | L1 sectors 12.69 M → 4.24 M (model 4.23 M); L1/TEX 94% → 42%; DRAM 49% → 66%; but 344 blocks = 0.64 waves, 34% occupancy (48 registers) | read each chunk of x once for 4 rows | 2.99× (1.36× the one-row INT8 kernel) |
| INT8, **2 rows per warp** | (best balance) | twice the warps of R = 4, half the x traffic of R = 1 | read each chunk of x once for 2 rows | **3.23×** (MLP down: 2.73×) |

**Why the activations dominate:** every weight is multiplied by one FP32 x value that each warp
reads through L1. That costs the same whatever the weight type, so it becomes the largest stream
once the weights shrink, and for FP16/INT8 each 16-byte x load is 32 or 64 bytes from its
neighbour's, so it fills only half a 32-byte sector. That model reproduces the measured L1
sector counts of every kernel exactly (11.27 / 14.09 / 12.68 / 4.23 M).

**Accuracy:** INT8 adds 0.4% relative RMS error on uniform weights (FP16: 0.02%). With 16 outlier
weights, a per-tensor scale gives 20%, per-row scales 1.5%.

**Small layers don't benefit:** GPT-2-sized layers (2–9 MB) take 5–7 µs in every format, and
INT8 is at most 1.10× faster: there, launch and latency set the time, not bytes.

---

## 5. Why inference performance depends on each factor: evidence from this project

| Factor | What we measured | Inference consequence |
|---|---|---|
| **Memory movement** | Fusion: 22.07 → 18.11 bytes/element gave 1.21×. LayerNorm v3 vs v2: 8.90 vs 19.8 bytes/element | Memory-bound ops (norms, softmax, residuals, decode) speed up only by moving fewer bytes: fusion, read-once kernels, lower precision |
| **Compute** | GEMM 61 → 2,236 GFLOP/s through reuse; Tensor Cores 1.49× (ours) and cuBLAS FP16 at 35.9 TFLOP/s vs 3.7 FP32 | Prefill is dominated by GEMMs; Tensor Cores + FP16/BF16 are the main lever |
| **Kernel launch overhead** | vector add: a 2.7 µs floor regardless of size; `cudaLaunchKernel` median 15.9 µs in the profiled program; PyTorch small ops: GPU time ≈ wall time (launch-bound) | Decode runs hundreds of tiny kernels per token. Fusion and CUDA Graphs cut launches |
| **Parallelism** | decode attention: 0.07 waves, 3% SM throughput; softmax of one 50,257-wide row: 7.2 GB/s; GEMM v4 at n = 128: 2.8× slower than v2 | Batch size 1 decoding leaves the GPU mostly idle. Batching requests and split-K/flash-decoding create parallelism |
| **Precision** | FP16 halves bytes; FP16 accumulation error 3.6e-2 vs 4.4e-6 (FP32) at K = 16,384, and overflow to inf | FP16/BF16 storage + FP32 accumulation is the standard; quantization (INT8/INT4) pushes the bytes down further for memory-bound decode. Measured (A100): INT8 weights 3.2× faster than FP32 in the decode GEMV, but only after removing the L1 bottleneck the smaller weights exposed (§4b) |
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
| INT8 weights (¼ the bytes) are 4× faster in decode | 2.2× (A100), then 3.2× after a second kernel | fewer bytes help only while DRAM is the bottleneck; the activation reads through L1 became the limit |
| More rows per warp = more activation reuse = faster | 2 rows beat 4 and 8 | reuse costs warps and registers; at 4 rows only 0.64 waves remained (34% occupancy) |

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
7. **INT8 GEMV:** coalesced activation loads (halve the half-used sectors) and splitting each
   row's K range across warps, so more rows per warp keep enough warps busy (3.2× of a possible
   ~4× today).

---

## 9. Challenges faced, and how they were solved

The problems that cost real time in this project, in one place. Each entry: what went wrong, how
it showed up, what fixed it, and the lesson. Details are in the linked sections.

### 9.1 Correctness

| Challenge | How it showed up | Fix | Lesson |
|---|---|---|---|
| **Variance computed as E[x²] − mean²** (one pass) | in a float demonstration it gave **2.125 instead of 0.086** (catastrophic cancellation) | two-pass variance: mean first, then Σ(x − mean)² from registers (06 §3) | an algebraically equal formula can be numerically wrong; test with large offsets |
| **Online softmax with masked (−∞) scores** | causal attention rows became NaN: with the running max still −∞, the update computed exp(−∞ − (−∞)) = NaN | skip −∞ values in the update (they contribute exp(−∞) = 0 anyway) (07 §5) | masking creates inputs a plain softmax test never sees; test the masked case |
| **Choosing tolerances** | a fixed tolerance either failed correct kernels (summation order differs from the CPU) or passed buggy ones | tolerances derived from the arithmetic (grow with K or row length), checked by CPU emulation of each kernel's summation order; poison values in outputs; a provable error bound for INT8 (scale/2 · Σ\|x\|) | a tolerance needs a reason; an indexing bug gives errors of order 1, rounding gives ~1e-6 |
| **Code written without a local GPU** | every kernel change could only be compiled and run on Colab, one round trip per attempt | CPU checks before each run: g++ for the host code (quantization, references), a PyTorch stand-in for the GPT-2 C++ ops (FP32 logits matched Hugging Face to 2.5e-7 on CPU), lint, and `run_all.sh` refusing to benchmark if any test fails | catch everything that doesn't need a GPU before using the GPU; the INT8, multi-row and GPT-2 code all passed their tests on the first GPU run |

### 9.2 Measurement

| Challenge | How it showed up | Fix | Lesson |
|---|---|---|---|
| **Timing an asynchronous launch** | a CPU timer without synchronization "measured" **0.024 ms** for a 3.07 ms kernel | CUDA events around many back-to-back launches, after a warm-up (09) | measure GPU time with GPU timestamps |
| **Clock throttling on the Colab T4** | one attention row was inflated; five identical GEMM v1 launches took **36.8–92.5 ms** | marked the row †, re-measured with the Nsight Systems timeline (4.12 / 5.08 ms) (12 §6, 10 §9.2) | a single number from a shared cloud GPU can be wrong; repeat and cross-check |
| **Clocks after idle time** (A100) | the first shape of the rows-per-warp sweep timed **28% slower** than the same kernel elsewhere in the same run | marked those values unreliable; the sweep now warms up for 100 launches instead of 5 (12 §9.2) | a GPU that idled during CPU work needs a longer warm-up |
| **L2 cache inflating "bandwidth"** | rows above **100% of DRAM peak** (softmax 4096×512 on the A100: 107%) | only rows whose data exceeds L2 count as DRAM results; 7B-class shapes added for the GEMV (09 §21) | know the cache size before claiming bandwidth |
| **Results that differ between runs** | four A100 runs: most rows within 1%, but cuBLAS FP16 varied 9% and fused attention at seq 2048 measured 7.81 / 6.43 / 7.84 ms | one run per results page, the run named, unstable rows stated (12 §9) | report run-to-run spread instead of hiding it |

### 9.3 Performance: when the expected bottleneck wasn't the real one

| Challenge | How it showed up | Fix | Lesson |
|---|---|---|---|
| **32×1 blocks were fastest** (v2 GEMM), against the "better L1 hit rate" hypothesis | Nsight: L1 hit rate was *worse* (19.8% vs 87.4%) | the real cause: 35% less DRAM traffic and half the queue stalls (10 §9.3, H3) | write the hypothesis down, then let the counters decide |
| **Warp-per-row softmax lost at 1,024 columns** | 1.8× slower than block-per-row | Nsight: it re-read x **2.77×** from DRAM because ~5 MB of rows in flight exceeded the 4 MB L2 (H7) | working-set size vs cache size decides access patterns |
| **Fused attention slower than unfused** (non-causal) | 5.08 vs 4.12 ms at seq 1024 | Nsight: instruction-issue-bound (FMA 60%, DRAM 0.4%), not memory-starved (H12) | fusion removes bytes, not instructions |
| **INT8 weights only 2.2× faster, not 4×** | the GEMV reached only 46–61% of DRAM peak | Nsight: L1 at 94% (activations re-read through L1 for every row); reading them once per 2 rows gave **3.2×** (12 §9.1–9.2) | fewer bytes help only while DRAM is the bottleneck |
| **My own traffic model was off by 2×** | predicted L1 sectors didn't match the counters at 4 rows per warp | FP16/INT8 activation loads fill only half of each 32-byte sector; with that, the model matches all four kernels within 0.2% (10 §9.7, H15) | when a model disagrees with the counters, fix the model, then it explains everything |
| **More rows per warp got slower** (4, 8) | R = 8 on a 4096-row layer was slower than the original kernel | Nsight: 0.64 waves, 34% occupancy at R = 4; R = 8 launched 64 blocks for 108 SMs (12 §9.2) | reuse costs parallelism; sweep the parameter |

### 9.4 Running a real model (GPT-2)

| Challenge | How it showed up | Fix | Lesson |
|---|---|---|---|
| **Weight layout** | Hugging Face GPT-2 stores linear layers as `Conv1D` `[in, out]`; our GEMV expects `[out, in]` | transpose once at load time (14 §3) | check the checkpoint's layout, not the library's usual one |
| **KV cache with spare capacity** | the attention kernel located head h at `h · kv_len · d`, which assumes a packed cache; real caches are preallocated for max_len | `AttentionShape::kv_capacity`, tested with NaN in the unused rows (bit-identical to packed) (14 §3) | kernels written for benchmarks need an interface for real memory layouts |
| **INT8 text diverged on GPT-2 small** | logits 1.8% off, greedy text different after 4 tokens (GPT-2 XL: 32/32) | reported as measured; cause not investigated (14 §4) | check generated text, not only logit error |
| **End-to-end speed didn't follow the kernel speedup** | INT8 and FP16 both 6.5 ms/token on GPT-2 XL | launch floor: ~725 launches × ~8.5 µs ≈ 6.2 ms (14 §5) | at batch 1 the launch count can matter more than the kernels |
| **A misleading 5× over Hugging Face** | our decode loop ~5× faster than Hugging Face eager | Hugging Face FP32 and FP16 took the same time: the gap is Python overhead; documented as "do not claim" | a speedup needs a mechanism before it is a result |

### 9.5 Workflow

| Challenge | How it showed up | Fix | Lesson |
|---|---|---|---|
| **A profiling case silently missing** | the zip had no `ncu_quantization_*` files after a full profiling run | most likely an older copy of the notebook was running in Colab (its case list predated the new case); the case was re-run separately | the notebook in Colab is not updated by uploading new code; check the outputs list, not just "no error" |
| **Results filed under the wrong GPU** | the GPT-2 runs were on an A100-**80GB**, but the command wrote into the 40GB folder | refiled under `benchmarks/NVIDIA_A100_SXM4_80GB/` and noted the different bandwidth (14 §5) | read the GPU name from the output, not the path |
| **Keeping docs and numbers in sync across reruns** | each new A100 run moved numbers slightly; README claims became stale (e.g. "86–88%" when softmax was at 80%) | numbers extracted by script from one run's CSVs; stale claims grep-checked before each commit | write numbers from data, never from memory |
