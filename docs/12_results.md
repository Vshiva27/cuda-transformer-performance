# 12 — Results

> Only numbers from real runs go here. Each table records the GPU, CUDA version and date it
> came from. Run `bash scripts/run_all.sh` (or the Colab notebook), then copy the values from
> `benchmarks/<GPU>/summary.md`. Its "Headline numbers" section already computes the ratios
> below. The raw output stays in `benchmarks/<GPU>/`.

## Environment

| Field | Value |
|---|---|
| GPU | |
| Compute capability | |
| CUDA runtime / driver | |
| Theoretical bandwidth (GB/s) | |
| CPU (for the CPU baseline) | |
| Date | |

## Vector add (FP32, block 256) — Phase 1

| n | CPU (ms) | CUDA naive (ms) | CUDA grid-stride (ms) | End-to-end (ms) | Achieved GB/s | % of peak | Kernel speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 2^20 | | | | | | | |
| 2^24 | | | | | | | |

Block-size sweep (n = ____):

| Block size | ms | GB/s |
|---|---|---|
| 32 | | |
| 64 | | |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |

Observations (write your own conclusions after running):
-

## GEMM v1–v4, FP32 — Phases 2–3

Peak FP32 of this GPU (from device query): ______ GFLOP/s

n = 1024 (repeat the table for 512 and 2048):

| Version | ms | GFLOP/s | % of peak | vs previous | vs v1 | vs CPU |
|---|---|---|---|---|---|---|
| CPU | | | – | – | – | 1× |
| v1 naive | | | | – | 1× | |
| v2 coalesced | | | | | | |
| v3 tiled-32 | | | | | | |
| v4 register 4×4 | | | | | | |
| cuBLAS FP32, TF32 off (PyTorch) | | | | – | | |
| cuBLAS FP16 in / FP32 acc (PyTorch) | | | | – | | |

v4 as a fraction of cuBLAS FP32: ____ %

## Softmax, FP32 — Phase 5

GB/s = 8 bytes per element (read x once + write y once) / time.

| Shape | CPU (ms) | v1 thread/row | v2 block/row | v3 warp/row | v4 online | PyTorch | best % of peak BW |
|---|---|---|---|---|---|---|---|
| 1024×128 | | | | | | | |
| 4096×1024 | | | | | | | |
| 12288×1024 | | | | | | | |

Row-length sensitivity (GB/s), ~16M elements:

| Shape | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 524288×32 | | | | |
| 16384×1024 | | | | |
| 256×65536 | | | | |
| 1×50257 | | | | |

## LayerNorm, FP32 — Phase 6

GB/s = 8 bytes per element / time.

| Shape | CPU (ms) | v1 thread/row | v2 warp/row | v3 block/row regs | PyTorch | best % of peak BW |
|---|---|---|---|---|---|---|
| 512×768 | | | | | | |
| 4096×1024 | | | | | | |
| 8192×4096 | | | | | | |

Fusion: residual add + LayerNorm (GB/s = 16 bytes per element / time)

| Shape | Unfused, ours (ms) | Fused, ours (ms) | Speedup | PyTorch add + layer_norm (ms) |
|---|---|---|---|---|
| 512×768 | | | | |
| 4096×1024 | | | | |
| 8192×4096 | | | | |

## Precision — Phase 7

Speed, n = 1024 and 2048 (GFLOP/s):

| Variant | n = 1024 | n = 2048 | vs FP32 v4 |
|---|---|---|---|
| FP32 v3 tiled-32 | | | |
| FP32 v4 register | | | 1× |
| FP16 tiled, FP32 acc | | | |
| FP16 tiled, FP16 acc | | | |
| v5 WMMA (Tensor Cores), FP32 acc | | | |
| cuBLAS FP16 (PyTorch) | | | |

Accuracy, M = N = 64 (error / max |exact|; "arith" = vs FP16-rounded inputs):

| K | FP32 v4 | FP16 in, FP32 acc (arith) | FP16 in, FP16 acc (arith) | WMMA (arith) | FP16 in, FP32 acc (total) |
|---|---|---|---|---|---|
| 64 | | | | | |
| 1024 | | | | | |
| 16384 | | | | | |

Overflow (inputs [0, 8], K = 8192): FP16-accumulator inf outputs: ____ / 256; FP32-accumulator: ____ / 256

## PyTorch baseline: launch overhead (from python/benchmark.py, vector_add)

| n | GPU ms (events) | wall ms | Launch-bound? (wall ≈ GPU and flat across sizes) |
|---|---|---|---|
| 2^10 | | | |
| 2^20 | | | |
| 2^26 | | | |

## Attention, ours — Phase 8 (FP32, 12 heads, d = 64)

| seq | causal | unfused ms (QKᵀ / softmax / PV) | fused ms | unfused / fused | workspace MB | PyTorch naive ms | PyTorch SDPA ms |
|---|---|---|---|---|---|---|---|
| 512 | no | | | | | | |
| 2048 | no | | | | | | |
| 512 | yes | | | | | – | – |
| 2048 | yes | | | | | – | – |

KV cache (one decoding step, fused kernel):

| context L | with cache (ms) | recompute all (ms) | ratio |
|---|---|---|---|
| 512 | | | |
| 2048 | | | |

## Attention memory (PyTorch, FP32, 12 heads, d = 64)

| seq | naive extra MB | SDPA extra MB | naive ms | SDPA ms |
|---|---|---|---|---|
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |

Transformer shapes (GPT-2 small), GFLOP/s per version:

| Shape | M | N | K | v1 | v2 | v3 | v4 | v4 GB/s |
|---|---|---|---|---|---|---|---|---|
| QKV projection, prefill 512 | 512 | 2304 | 768 | | | | | |
| MLP up, prefill 512 | 512 | 3072 | 768 | | | | | |
| MLP down, prefill 512 | 512 | 768 | 3072 | | | | | |
| QKV projection, decode 1 | 1 | 2304 | 768 | | | | | |
| MLP up, decode 1 | 1 | 3072 | 768 | | | | | |

v3 tile sizes (n = 1024): tile 8: ____ GFLOP/s, tile 16: ____, tile 32: ____

Observations:
-

## Summary table (filled in over all phases)

| Operation | Size | Precision | CPU | PyTorch | CUDA Basic | CUDA Optimized | Speedup (Basic→Optimized) | GPU |
|---|---|---|---|---|---|---|---|---|
| Vector add | | FP32 | | | | | | |
| GEMM | | FP32 | | | | | | |
| GEMM | | FP16 in / FP32 acc | | | | | | |
| Softmax | | FP32 | | | | | | |
| LayerNorm | | FP32 | | | | | | |
| Attention | | FP32 | | | | | | |
