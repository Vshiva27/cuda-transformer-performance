# Resume Bullets

Every number below was measured on an NVIDIA Tesla T4 (CUDA 13.0) and traces to
`benchmarks/Tesla_T4/` or `profiling/reports/Tesla_T4/`. Before using a bullet, make sure you can answer
the "defend it" questions without notes. An interviewer will pick one number and dig.

**Project line:**
> **CUDA-Accelerated Transformer Inference Performance Benchmark** — C++17, CUDA, PyTorch, Nsight Systems/Compute

---

## Option A — GEMM optimization and profiling (best for GPU performance / kernel roles)

> Optimized an FP32 GEMM kernel in CUDA from 61 to 2,236 GFLOP/s (36.5×, 60% of cuBLAS) on an
> NVIDIA T4 through memory coalescing, shared-memory tiling and register tiling, validating each
> step with Nsight Compute counters (global-load sectors/request 16.5 → 4.0, DRAM reads 433 → 33 MB).

**Defend it:**
- the thread → data mapping in v1 vs v2, and why the sector count is exactly (32 + 1)/2;
- the k-tile loop and both `__syncthreads()`;
- the outer product in v4 and why it's 0.5 shared loads per FMA;
- why it's 60% and not 100% of cuBLAS: tail effect (2.13 waves), 72 registers → 75% occupancy,
  latency;
- why v4 is *slower* than v2 at n = 128.

## Option B — memory-bound kernels and fusion (best for inference / runtime roles)

> Wrote memory-bound Transformer kernels (softmax, LayerNorm, residual add) reaching 72–76% of
> peak DRAM bandwidth; a register-resident LayerNorm ran 1.38× faster than PyTorch at LLaMA-7B's
> hidden size, and a fused residual-add + LayerNorm cut DRAM traffic from 22.1 to 18.1
> bytes/element (Nsight-verified), giving 1.21× over the unfused kernels and 1.5× over eager PyTorch.

**Defend it:**
- the roofline: why these ops are memory-bound and judged in GB/s;
- the two-pass variance and why E[x²] − mean² fails;
- the hierarchical block reduction (shuffles → shared → shuffles, 2 barriers);
- why fusion's ceiling is 20/16 = 1.25×;
- why the small shape gained only 1.04× (L2);
- the scope of the PyTorch comparison: FP32, eager mode, these shapes, one GPU.

## Option C — precision, attention and inference (best for AI systems / accelerator roles)

> Implemented FP16 Tensor Core GEMM via WMMA with FP32 accumulation (1.49× over the best FP32
> kernel), quantified FP16-accumulation error (~8,000× higher at K = 16K, plus overflow), and
> built causal attention with a FlashAttention-style online-softmax fused kernel and KV-cache
> decoding (30× cheaper per generated token at a 2K context), all gated by CPU-reference
> correctness tests.

**Defend it:**
- FP16 range and precision;
- why accumulation stays FP32, and that an FP16 × FP16 product is exact in FP32;
- what a WMMA fragment is, and why your WMMA kernel is only ~9% of cuBLAS FP16 (no shared-memory
  staging; LG-throttle stalls);
- the online-softmax rescaling of m, l, o;
- why your fused kernel is slower for non-causal attention (instruction-bound) but faster for
  causal;
- what the KV cache stores and its memory cost.

---

## Short version (two bullets, if space is tight)

> - Built and profiled CUDA kernels for Transformer inference (GEMM, softmax, LayerNorm,
>   attention) from naive to optimized; GEMM reached 2,236 GFLOP/s (36.5× over naive, 60% of
>   cuBLAS) on an NVIDIA T4, with each optimization confirmed by Nsight Compute counters.
> - Fused residual-add + LayerNorm (1.21×, DRAM traffic −18%), FP16 Tensor Core GEMM (1.49× over
>   FP32), and FlashAttention-style attention with KV-cache decoding (30× cheaper per token at 2K
>   context), verified against CPU references and benchmarked against PyTorch.

---

## Do NOT claim

| Tempting claim | Why not |
|---|---|
| "faster than cuBLAS" / "near-cuBLAS" | 60% of cuBLAS FP32; WMMA is ~9% of cuBLAS FP16 |
| "411× faster than CPU" without context | that's vs a **single-threaded**, unvectorized-by-design CPU reference. If used, say "vs a single-thread CPU reference" |
| "faster than PyTorch" (in general) | only for specific ops and shapes, in FP32 eager mode, on a T4. PyTorch's attention and GEMMs were faster than ours |
| "implemented FlashAttention" | it's FlashAttention-*style* (online softmax, no score matrix), without the tiling and Tensor Cores that make FlashAttention fast |
| "optimized Tensor Core GEMM" | it uses Tensor Cores, but the profiler shows they are starved of data |
| "LLM inference engine" / "served a model" | no model weights, tokenizer or serving; these are the kernels such engines use |
| "INT8/INT4 quantization" | only INT8 weight-only for the decode GEMV is implemented, no INT4; quote speedups only after the GPU run has measured them |
| any number from the † attention row | it was throttled; use the Nsight Systems numbers (4.12 / 5.08 ms at seq 1024) |
| bandwidth numbers above 100% of peak | those rows measured the L2 cache, not DRAM |
