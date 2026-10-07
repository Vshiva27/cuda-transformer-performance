# 12 — Results

> Every number on this page comes from one real run, saved in
> [`benchmarks/Tesla_T4/`](../benchmarks/Tesla_T4/) (raw output `*.txt`, CSV rows `*.csv`,
> generated `summary.md`). Nothing is estimated. Where a measurement looks unreliable, it is
> marked † and explained. Where a measurement contradicted a prediction in the earlier docs,
> that is stated too.

## Environment

| Field | Value |
|---|---|
| GPU | Tesla T4 (Google Colab), 40 SMs, 14.56 GiB, 4 MB L2 |
| Compute capability | 7.5 (Turing) |
| Driver / CUDA | 580.82.07 / CUDA 13.0 (nvcc 13.0.88) |
| PyTorch | 2.11.0+cu130 |
| Theoretical peak bandwidth | 320.1 GB/s (from device query) |
| Theoretical peak FP32 | 8,141 GFLOP/s (at the 1,590 MHz boost clock; the T4's 70 W power cap usually keeps sustained clocks lower, so real attainable peak is below this) |
| Ridge point | 25.4 FLOP/byte |
| CPU (for CPU baselines) | Colab VM CPU (model not recorded in this run; `run_all.sh` now records `lscpu`) |
| Date | 2026-10-07 |

**Correctness:** all 6 test programs passed (`tests.txt`). The run script refuses to benchmark
otherwise, so every number below belongs to a kernel that was verified first.

**How to read "% of peak bandwidth" above 100%:** the T4 has a 4 MB L2 cache. When a benchmark's
working set is smaller than that (e.g. vector add with n = 262,144: 3 MB), the repeated
iterations read from L2 instead of DRAM, and "bandwidth" exceeds the DRAM peak. Those rows
(marked *) measure L2, not DRAM. Use the large sizes for DRAM claims.

---

## 1. Vector add (FP32, block 256) — Phase 1

| n | CPU ms | naive ms | grid-stride ms | end-to-end ms | naive GB/s | % of peak | kernel vs CPU |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.0002 | 0.0027 | 0.0028 | 0.0232 | 4.5 | 1.4% | 0.1× |
| 16,384 | 0.0047 | 0.0028 | 0.0028 | 0.0950 | 70.8 | 22.1% | 1.7× |
| 262,144 | 0.1270 | 0.0043 | 0.0035 | 0.9242 | 728.0 | 227.5%* | 29.4× |
| 1,048,576 | 0.5163 | 0.0508 | 0.0527 | 2.8464 | 247.5 | 77.3% | 10.2× |
| 16,777,216 | 16.0513 | 0.7836 | 0.8901 | 43.9202 | 256.9 | 80.3% | 20.5× |
| 67,108,864 | 66.8643 | 3.0657 | 3.8779 | 173.7274 | 262.7 | 82.1% | 21.8× |

Block-size sweep (n = 2²⁶): 32 → 220.7 GB/s; 64 → 257.6; 128 → 258.6; 256 → 259.6; 512 → 259.2;
1024 → 258.6. PyTorch `a + b` at n = 2²⁶: 246.4 GB/s.

**Observations**
- Large vectors reach **82% of theoretical DRAM bandwidth**. This is the ceiling for a
  memory-bound kernel (predicted 70–90% in 02 §11 ✔).
- At n = 1,024 the kernel takes 2.7 µs regardless of size: that's the launch-overhead floor.
  The single-thread CPU is 10× faster here ✔.
- **End-to-end (with PCIe copies) is 2.6× slower than the CPU** at n = 2²⁶ (173.7 ms vs 66.9 ms).
  The copies move 805 MB in about 170 ms ≈ 4.7 GB/s (pageable memory). The kernel itself is only
  3 ms. This is the measured reason inference keeps weights and activations on the GPU.
- The deliberately wrong "no-sync" CPU timing reported 0.024 ms for a 3.07 ms kernel. It measured
  only the launch.
- Block size barely matters once it is ≥ 64. Block 32 is 15% slower: with a limit of 16 blocks
  per SM, 32-thread blocks fill at most 512 of the SM's 1,024 thread slots (50% occupancy).
- The grid-stride version was *slower* at large n (3.88 vs 3.07 ms). Using only 160 blocks
  (40 SMs × 4) left too few loads in flight to saturate DRAM.

---

## 2. GEMM v1–v4, FP32 — Phases 2–3

**n = 1024** (GFLOP/s; % of the 8,141 GFLOP/s theoretical peak):

| Version | ms | GFLOP/s | % peak | vs previous | vs v1 | vs CPU |
|---|---|---|---|---|---|---|
| CPU (1 thread) | 394.75 | 5.4 | – | – | – | 1× |
| v1 naive | 35.067 | 61.2 | 0.8% | – | 1× | 11× |
| v2 coalesced | 3.689 | 582.1 | 7.2% | 9.51× | 9.51× | 107× |
| v3 tiled-32 | 2.476 | 867.4 | 10.7% | 1.49× | 14.2× | 159× |
| v4 register 4×4 | 0.961 | 2,235.6 | 27.5% | 2.58× | **36.5×** | **411×** |
| cuBLAS FP32, TF32 off (PyTorch) | 0.576 | 3,729.9 | 45.8% | – | – | – |
| cuBLAS FP16 in / FP32 acc (PyTorch) | 0.060 | 35,884.1 | – | – | – | – |

**v4 reaches 60% of cuBLAS FP32** at n = 1024 (57% at n = 2048: 2,221.0 vs 3,890.8 GFLOP/s).

| n | v1 | v2 | v3 | v4 | cuBLAS FP32 |
|---|---|---|---|---|---|
| 128 | 47.8 | **495.5** | 426.7 | 175.8 | 216.9 |
| 256 | 55.9 | 732.7 | **835.4** | 729.3 | 1,622.4 |
| 512 | 60.5 | 815.1 | 915.4 | 1,898.4 | 5,019.9* |
| 1024 | 61.2 | 582.1 | 867.4 | 2,235.6 | 3,729.9 |
| 2048 | 61.6 | 453.7 | 874.2 | 2,221.0 | 3,890.8 |

**Transformer shapes (GPT-2 small), GFLOP/s** (decode rows: GB/s in brackets):

| Shape | M×N×K | v1 | v2 | v3 | v4 | cuBLAS FP32 |
|---|---|---|---|---|---|---|
| QKV projection, prefill 512 | 512×2304×768 | 61.2 | 463.3 | 789.3 | 2,193.6 | 4,209.8 |
| MLP up, prefill 512 | 512×3072×768 | 61.4 | 455.2 | 819.3 | 2,189.8 | 4,076.4 |
| MLP down, prefill 512 | 512×768×3072 | 61.0 | 485.8 | 818.9 | 1,978.5 | 3,940.1 |
| QKV projection, decode 1 | 1×2304×768 | 19.5 (39 GB/s) | **102.5 (205 GB/s)** | 25.3 (51) | 20.8 (42) | 119.7 (240 GB/s) |
| MLP up, decode 1 | 1×3072×768 | 21.5 (43) | **111.6 (224)** | 22.7 (46) | 25.9 (52) | 121.2 (243) |

Block shapes, v2 at n = 1024 (GFLOP/s): 32×1 **598.8**, 32×4 474.5, 32×8 530.4, 32×16 566.9,
32×32 577.9, 16×16 449.2, 8×8 318.2, 8×32 336.6.
Tile sizes, v3 at n = 1024: 8 → 524.0, 16 → 783.1, 32 → 814.5.

**Observations**
- Each step gave the improvement its memory analysis predicted. Coalescing gave **9.5×**
  (v1 → v2, same arithmetic). Shared-memory tiling gave 1.5×. Register tiling gave **2.6×**,
  breaking v3's shared-memory bottleneck.
- **Optimizations only pay off above a problem size.** At n = 128, v2 is fastest and v4 is 2.8×
  slower than v2: a 64×64 block tile gives only 4 blocks for 40 SMs (predicted in 04 §22 ✔).
- **Decode GEMMs (M = 1) invert the ranking:** the simple coalesced v2 reaches 205–224 GB/s
  (64–70% of DRAM peak, 86–92% of cuBLAS), while the tiled v3/v4 waste 31 of 32 (or 63 of 64)
  tile rows and run 4–5× slower. Decode needs different (GEMV) kernels.
- cuBLAS FP32 at n = 512 (5,020 GFLOP/s*) beats its own n = 1024 result. The 3 MB working set fits
  in L2.
- *Contradicted prediction:* 04 §7 expected the 32×1 block shape to suffer from low occupancy.
  It was the **fastest** v2 shape. Occupancy is not the whole story; finding out why is a
  Nsight Compute task (Phase 10).

---

## 3. Precision — Phase 7

**Speed, GFLOP/s** (verified against an exact double-precision product first, n ≤ 1024):

| Variant | n = 256 | n = 512 | n = 1024 | n = 2048 | vs FP32 v4 (1024) | scaled error (1024) |
|---|---|---|---|---|---|---|
| FP32 v3 tiled-32 | 834.1 | 1,019.5 | 900.1 | 827.9 | 0.41× | 1.49e-6 |
| FP32 v4 register | 729.9 | 2,188.5 | 2,203.0 | 2,076.7 | 1× | 1.49e-6 |
| FP16 tiled, FP32 acc | 930.7 | 1,088.8 | 983.5 | 838.6 | 0.45× | 1.48e-6 |
| FP16 tiled, FP16 acc | 988.6 | 1,121.4 | 1,085.0 | 924.1 | 0.49× | **1.06e-2** |
| **v5 WMMA (Tensor Cores), FP32 acc** | **2,526.7** | **3,185.0** | **3,289.5** | **2,406.7** | **1.49×** | 2.88e-6 |
| cuBLAS FP16 (PyTorch) | 2,322.7 | 17,581.2 | 35,884.1 | 20,038.3 | – | – |

**Accuracy vs K** (M = N = 64; error / max |exact|; "arith" = vs the FP16-rounded inputs):

| K | FP32 v4 | FP16 in, FP32 acc: total / arith | FP16 in, **FP16 acc**: arith | WMMA: total / arith |
|---|---|---|---|---|
| 64 | 2.78e-7 | 2.59e-4 / 3.44e-7 | 2.34e-3 | 2.59e-4 / 2.83e-7 |
| 256 | 5.06e-7 | 2.34e-4 / 5.82e-7 | 3.59e-3 | 2.34e-4 / 6.62e-7 |
| 1024 | 8.43e-7 | 2.87e-4 / 1.27e-6 | 7.63e-3 | 2.86e-4 / 2.47e-6 |
| 4096 | 2.65e-6 | 2.57e-4 / 2.36e-6 | 1.58e-2 | 2.57e-4 / 9.19e-6 |
| 16384 | 5.76e-6 | 2.97e-4 / 4.35e-6 | **3.62e-2** | 3.00e-4 / 2.95e-5 |

**Overflow** (inputs in [0, 8], K = 8192, largest exact output 133,839 > FP16 max 65,504):
FP16 accumulator: **256 of 256 outputs inf**. All FP32-accumulating variants: 0.

Transformer prefill shapes (GFLOP/s), FP32 v4 → WMMA: QKV 2,075.0 → 3,094.4; MLP up
2,028.6 → 2,648.7; MLP down 1,942.8 → 2,832.7.

**Observations**
- **FP16 accumulation is ~8,000× less accurate** than FP32 accumulation at K = 16,384
  (3.6e-2 vs 4.4e-6) and **overflows** on large sums. That's why inputs are stored in FP16 but
  accumulated in FP32. The CPU emulation in 08 §5 predicted 2.8e-2 and 4.1e-6 ✔.
- With FP16 inputs and FP32 accumulation, the **total** error (~2.6e-4) comes almost entirely from
  rounding the inputs to FP16; the kernel's own arithmetic adds only ~1e-6.
- Tensor Core accumulation measured ~7× less accurate than CUDA-core FP32 at K = 16,384
  (2.95e-5 vs 4.35e-6), still far better than FP16 accumulation.
- FP16 on CUDA cores (tiled) is no faster than FP32 tiled: the math still runs as FP32 FMAs
  (predicted in 08 §6 ✔). **Tensor Cores gave 1.49×** over the best FP32 kernel.
- **Our WMMA kernel reaches only ~9% of cuBLAS FP16** (3,290 vs 35,884 GFLOP/s at 1024). Each
  warp loads its fragments straight from global memory, with no shared-memory staging and no
  reuse between warps (08 §7 limitations). That's the clearest example in the project of "the
  instruction is fast, the data supply is not".

---

## 4. Softmax, FP32 — Phase 5

GB/s = 8 bytes per element / time.

| Shape | CPU ms (double) | v1 thread/row | v2 block/row | v3 warp/row | v4 online | PyTorch | best % of peak |
|---|---|---|---|---|---|---|---|
| 1024×128 | 2.419 | 9.9 | 53.8 | **285.4*** | 221.0 | 132.1 | (L2-resident) |
| 4096×512 | 42.900 | 32.8 | 158.2 | 195.8 | 152.0 | **219.1** | 68% (PyTorch) |
| 4096×1024 | 76.851 | 31.9 | **239.0** | 127.4 | 126.2 | 213.0 | 75% |
| 12288×1024 | 232.980 | 31.8 | **243.2** | 137.0 | 129.9 | 216.2 | 76% |

Row-length sensitivity (GB/s, ~16M elements):

| Shape | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 524288×32 | 11.6 | 15.2 | **218.0** | 160.4 |
| 65536×256 | 5.6 | 106.5 | **227.1** | 200.5 |
| 16384×1024 | 32.9 | **243.8** | 138.2 | 130.0 |
| 4096×4096 | 33.0 | **202.4** | 108.1 | 121.8 |
| 256×65536 | 2.4 | **112.4** | 83.1 | 69.1 |
| 334×50257 | 3.9 | **95.9** | 76.4 | 69.9 |
| 1×50257 | 0.1 | **7.2** | 1.0 | 0.6 |

v2 block size at 4096×1024: 32 → 177.8, 64 → 191.1, 128 → 227.2, **256 → 238.6**,
512 → 133.0, 1024 → 51.8 GB/s.

**Observations**
- **The best kernel depends on the row length.** One warp per row wins up to 256 columns. One
  block per row wins from 1,024 columns upward, including the attention shape 12288×1024, where
  our v2 is **12% faster than PyTorch** (243.2 vs 216.2 GB/s).
- A single 50,257-wide row (LLM decode logits) reaches only 7.2 GB/s with any design: one block
  can use only one SM (predicted in 05 §11 ✔).
- *Contradicted prediction:* online softmax (v4) was **not** faster than v3. It was equal or
  slower. Its saved pass over x was mostly an L1/L2 hit anyway, and its per-element work is
  higher (a branch, and an extra exp whenever the max changes). **Online softmax's value is
  enabling fusion** (FlashAttention, §6), not speeding up standalone softmax.
- *Contradicted prediction:* 05 §11 expected warp-per-row to win at 16384×1024. Block-per-row
  won by 1.8×. The likely cause is per-row parallelism and memory-level parallelism (32 lanes
  looping 32 times per pass vs 256 threads looping 4 times), but this is unverified. It's a
  Phase 10 Nsight question.
- Block size 1024 collapses v2 to 52 GB/s: one block per SM, and 10 barrier levels per reduction.

---

## 5. LayerNorm and fusion, FP32 — Phase 6

GB/s = 8 bytes per element / time.

| Shape | CPU ms (double) | v1 thread/row | v2 warp/row | v3 block/row regs | PyTorch | v3 % of peak |
|---|---|---|---|---|---|---|
| 512×768 | 1.498 | 4.7 | **394.7*** | 350.7* | 220.8 | (L2-resident) |
| 2048×768 | 5.859 | 17.8 | 146.4 | **234.6** | 224.3 | 73% |
| 4096×1024 | 16.213 | 33.0 | 135.1 | **236.4** | 224.5 | 74% |
| 8192×4096 | 127.583 | 34.2 | 110.1 | **230.1** | 166.2 | 72% |

**Fusion: residual add + LayerNorm** (GB/s = 16 bytes per element / time):

| Shape | Unfused, ours (2 kernels) | Fused, ours (1 kernel) | Fusion speedup | PyTorch add + layer_norm | Ours fused vs PyTorch |
|---|---|---|---|---|---|
| 512×768 | 0.0302 ms | 0.0290 ms | 1.04× | 0.0330 ms | 1.14× |
| 2048×768 | 0.1295 ms | 0.1076 ms | 1.20× | 0.1358 ms | 1.26× |
| 4096×1024 | 0.3434 ms | 0.2832 ms | **1.21×** | 0.3587 ms | 1.27× |
| 8192×4096 | 2.6401 ms | 2.1892 ms | **1.21×** | 3.2794 ms | **1.50×** |

**Observations**
- Keeping the row in registers (v3, x read once) gives **72–74% of peak bandwidth** on large
  shapes. That's **7× faster than v1**, and **1.38× faster than PyTorch's `layer_norm`** at the
  LLaMA-7B hidden size (230.1 vs 166.2 GB/s).
- **Fusion gave 1.20–1.21× on large shapes, close to the 1.25× upper bound** from byte counting
  (20 → 16 bytes per element, 06 §8 ✔). The fused kernel is 1.27–1.50× faster than eager
  PyTorch's two-kernel pipeline.
- *Contradicted prediction:* 06 §8 expected the *small* shape to gain most (saved launch). It
  gained least (1.04×). At 512×768, the intermediate h (1.5 MB) stays in the 4 MB L2, so the
  unfused read-back is an L2 hit. And back-to-back launches overlap their launch overhead in
  event timing. Fusion pays when the intermediate would otherwise go to DRAM.
- v2 (warp per row) beats v3 only on the L2-resident small shape; on DRAM-bound shapes, v3's
  single read wins clearly.

---

## 6. Attention, FP32 (12 heads, d = 64) — Phase 8

Non-causal (same shapes as PyTorch); ms:

| seq | ours unfused (QKᵀ / softmax / PV) | ours fused | unfused / fused | our workspace | PyTorch naive (extra MB) | PyTorch SDPA (extra MB) |
|---|---|---|---|---|---|---|
| 128 | 0.066 (0.028 / 0.006 / 0.031) | 0.089 | 0.74× | 0.8 MB | 0.062 (1.9) | 0.101 (2.1) |
| 512 | 1.027 (0.599 / 0.137 / 0.508) | 1.821 | 0.56× | 12 MB | 0.442 (25.5) | 0.600 (29.5) |
| 1024 † | 9.252 (2.344 / 0.750 / 1.998) | 12.052 | 0.77× | 48 MB | 1.946 (99.0) | 2.243 (111.0) |
| 2048 | 18.383 (8.295 / 3.058 / 7.246) | 29.347 | 0.63× | 192 MB | 7.697 (390.0) | 8.834 (438.0) |

Causal (decoder models); ms:

| seq | ours unfused | ours fused | unfused / fused |
|---|---|---|---|
| 128 | 0.072 | 0.076 | 0.95× |
| 512 | 1.012 | 0.963 | 1.05× |
| 1024 | 4.243 | 3.419 | 1.24× |
| 2048 | 18.172 | 15.015 | 1.21× |

**KV cache** (one decoding step, fused kernel):

| context L | with cache (q_len = 1) | recompute attention for all L | ratio |
|---|---|---|---|
| 128 | 0.0312 ms | 0.0772 ms | 2.5× |
| 512 | 0.1060 ms | 0.7288 ms | 6.9× |
| 1024 | 0.2588 ms | 3.4768 ms | 13.4× |
| 2048 | 0.5032 ms | 15.0928 ms | **30.0×** |

KV cache size for GPT-2 small in FP16: 36 KB per token, 72 MB per 2,048-token sequence.

**Observations**
- **Memory:** the fused kernel needs no score matrix. The unfused version needs 192 MB at
  seq 2048 (ours stores S once and runs softmax in place; PyTorch's naive path allocated 390 MB).
- **Speed:** as 07 §6 warned, our fused kernel is **slower** for non-causal attention
  (0.56–0.77×). It does one key at a time with warp shuffles instead of GEMM-style tiles. With
  causal masking it **skips** future keys and becomes **1.2× faster** than unfused at seq ≥ 1024.
  Causal fused costs about half of non-causal fused (15.0 vs 29.3 ms at 2048), as predicted ✔.
  Unfused causal costs the same as non-causal (it computes everything, then masks) ✔.
- PyTorch's naive attention is 2.4× faster than our unfused version at seq 2048. Its two GEMMs
  are cuBLAS batched GEMMs, versus our v3-style tiled kernel.
- **KV cache:** a decoding step with the cache is **30× cheaper** at a 2,048-token context, and the
  ratio grows roughly linearly with context length ✔. (The real gap is larger, because
  recomputing without a cache would also redo the K/V projection GEMMs.)
- *Correction to earlier docs:* **PyTorch SDPA did not save memory in this configuration** (FP32
  inputs on a T4). It allocated 438 MB at seq 2048, more than the naive path, and was slower.
  Its memory-saving fused kernels evidently were not used for FP32 on this GPU, so it most likely
  fell back to a "math" implementation; which backend was chosen was not verified. The
  statements in 09 §8 and 07 have been corrected. `python/benchmark.py` now also measures FP16
  SDPA.
- † **Unreliable row (seq 1024, non-causal):** the unfused total (9.25 ms) is far above the sum
  of its separately timed kernels (5.09 ms), while at seq 2048 the two agree (18.38 vs 18.60 ms).
  The fused time (12.05 ms) also breaks the ~4× per doubling scaling (1.82 ms at 512, 29.3 ms at
  2048). The causal-1024 and KV-cache-1024 rows, measured minutes later, are consistent. The
  likely cause is clock or power throttling during that measurement (only 10 iterations). **Rerun
  `./build/bench_attention` before quoting the seq-1024 non-causal numbers.**

---

## 7. Summary table

| Operation | Size | Precision | CPU (1 thread) | PyTorch | CUDA basic | CUDA optimized | Basic → optimized | GPU |
|---|---|---|---|---|---|---|---|---|
| Vector add | n = 2²⁶ | FP32 | 66.86 ms | 3.268 ms | 3.066 ms (naive) | – (memory-bound: 82% of peak already) | – | T4 |
| GEMM | 1024³ | FP32 | 394.75 ms | 0.576 ms (cuBLAS) | 35.07 ms (v1) | 0.961 ms (v4) | **36.5×** | T4 |
| GEMM | 1024³ | FP16 in / FP32 acc | – | 0.060 ms (cuBLAS) | 2.184 ms (tiled, CUDA cores) | 0.653 ms (v5 WMMA) | 3.3× | T4 |
| Softmax | 12288×1024 | FP32 | 232.98 ms | 0.466 ms | 3.161 ms (v1) | 0.414 ms (v2) | **7.6×** | T4 |
| LayerNorm | 8192×4096 | FP32 | 127.58 ms | 1.615 ms | 7.856 ms (v1) | 1.167 ms (v3) | **6.7×** | T4 |
| Add + LayerNorm | 8192×4096 | FP32 | – | 3.279 ms (2 kernels) | 2.640 ms (ours, 2 kernels) | 2.189 ms (fused) | 1.21× | T4 |
| Attention (causal) | 12×2048×64 | FP32 | – | – | 18.17 ms (unfused) | 15.02 ms (fused) | 1.21× | T4 |
| Attention decode | 1 × 2048 ctx | FP32 | – | – | 15.09 ms (recompute) | 0.503 ms (KV cache) | **30×** | T4 |

## 8. Open questions for profiling (Phase 10)

These are measured effects whose cause is not yet confirmed:
1. Why is v2 GEMM fastest with 32×1 blocks, despite the lower occupancy limit?
2. Why does warp-per-row softmax lose to block-per-row at 1,024 columns?
3. How far is v4 from the T4's *actual* sustained clock, and is it limited by shared memory or
   by instruction issue? (v4 = 60% of cuBLAS.)
4. Confirm that the WMMA kernel is limited by global-memory loads (expected: high L2/DRAM
   throughput, low Tensor Core utilization).
5. Rerun attention at seq 1024 to settle the † row.
