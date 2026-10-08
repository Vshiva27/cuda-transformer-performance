# 13 — Interview Questions and Answers

How to use this page: read the question, answer **out loud** without looking, then compare.
Every answer is grounded in this project, and the numbers are your own measurements on a
Tesla T4 ([12_results.md](12_results.md), [10_nsight_profiling.md §9](10_nsight_profiling.md)),
except where an answer says **A100**: the A100 results and all INT8 quantization numbers are in
[12_results.md §9](12_results.md#9-nvidia-a100-sxm4-40gb). If you can't explain a number, don't
say it in the interview.

---

## Part A — The project pitch

**"Tell me about this project." (≈ 90 seconds)**

> "I built and profiled the GPU kernels that dominate Transformer inference — GEMM, softmax,
> LayerNorm, residual add and attention — in CUDA from scratch, each from a naive version to an
> optimized one, compared against a CPU reference and PyTorch. Correctness came first: every
> kernel is tested against a double-precision CPU reference, and my run script refuses to
> benchmark if a test fails. On a T4, my GEMM went from 61 to 2,236 GFLOP/s across four
> versions — coalescing, shared-memory tiling, register tiling — reaching 60% of cuBLAS, and I
> used Nsight Compute to confirm each step: sectors per request went from 16.5 to 2.5 to 4, DRAM
> reads from 433 MB to 33 MB. For memory-bound ops I reached 72–76% of peak bandwidth; my
> LayerNorm was 1.38× faster than PyTorch's at LLaMA-7B's hidden size, and a fused residual-add +
> LayerNorm cut DRAM traffic from 22 to 18 bytes per element, matching its measured 1.21×
> speedup. I also did FP16 with FP32 accumulation on Tensor Cores via WMMA, a FlashAttention-style
> fused attention, and measured that a KV-cache decode step is 30× cheaper than recomputing
> attention at a 2K context. Then, on an A100, I added INT8 weight-only quantization for the
> decode matrix-vector product. INT8 weights first ran only 2.2× faster than FP32, not the 4×
> the bytes predict; Nsight Compute showed the activation reads had made L1 the bottleneck, and
> reading each activation once for two rows brought it to 3.2×."

**"What was the hardest part?"** → Pick one you can go deep on. Good candidates:
(a) GEMM indexing and tiling: the k-tile loop and two barriers; (b) online softmax and its
NaN edge cases with masked −∞ scores; (c) the profiler showing that my 32×1-block hypothesis was
wrong (it was DRAM traffic, not L1); (d) the INT8 GEMV, where each fix exposed the next
bottleneck: DRAM → L1 → too few warps. All challenges, with how each was solved:
[11_optimization.md §9](11_optimization.md#9-challenges-faced-and-how-they-were-solved).

**"What would you do next?"** → [11_optimization.md §8](11_optimization.md): WMMA with
shared-memory staging (9% of cuBLAS FP16 today), FlashAttention-style tiles, flash-decoding,
and for the INT8 GEMV coalesced activation loads and splitting rows across warps.

---

## Part B — CUDA

**What is a CUDA kernel?**
A function marked `__global__` that runs on the GPU and is executed by many threads in parallel.
It is launched from the CPU with `kernel<<<grid, block>>>(args)`. The launch is asynchronous: it
only queues the work. A kernel returns `void`; results go to memory.

**What is a thread?**
The smallest unit of execution. Each thread runs the kernel body once, with its own registers
and its own `threadIdx`/`blockIdx`, which it uses to choose its data. Example: in vector add,
`i = blockIdx.x * blockDim.x + threadIdx.x` = threads in earlier blocks + my position.

**What is a warp?**
32 threads that the hardware schedules together and that execute the same instruction
simultaneously (SIMT). Memory coalescing, divergence and shuffles are all defined per warp. In a
2D block, the warp spreads along `threadIdx.x`, which is why mapping `x` to columns made GEMM
coalesced.

**What is a block?**
A group of up to 1024 threads that runs on one SM, can share `__shared__` memory, and can
synchronize with `__syncthreads()`. Blocks are independent and can run in any order.

**What is a grid?**
All blocks of one launch; up to 3D. I used `gridDim.z` to batch attention heads in one launch.

**What is occupancy?**
Active warps per SM ÷ the maximum (32 on T4). It's limited by registers, shared memory, threads
and blocks per SM. **It's a means of hiding latency, not a goal:** my v2 GEMM ran equally fast at
49.7% and 98.7% occupancy, and my fastest FP32 GEMM (v4) runs at 66%, limited by 72 registers per
thread. But 32-thread blocks in vector add (50% occupancy) did cost 15%, because that kernel *is*
latency-bound.

**What is coalesced memory access?**
When the 32 threads of a warp access consecutive addresses, the hardware serves them with the
minimum number of 32-byte sectors: 4 sectors for 32 floats. Measured: naive GEMM 16.5 sectors per
request (strided), coalesced 2.5, tiled 4.0.

**What is shared memory?**
On-chip memory per SM (64 KB on T4), shared by a block's threads, managed explicitly by the
program. It's used to stage data for reuse: GEMM tiles, reduction scratch, attention K/V tiles.
It is organized in 32 banks.

**What are registers?**
The fastest storage, private to each thread. A T4 SM has 65,536 32-bit registers. More registers
per thread means fewer resident warps. Arrays stay in registers only with compile-time indices
(`#pragma unroll`, `constexpr`); otherwise they spill to local memory. I verified 0 bytes of
local memory for all kernels with `cuobjdump`.

**Why is shared memory faster than global memory?**
It's on-chip, next to the SM's execution units, while global memory is off-chip DRAM (hundreds
of cycles away, behind L1/L2). Shared memory has much lower latency and much higher bandwidth per
SM. And because it's programmer-managed, data stays until you overwrite it, unlike a cache.

**What is warp divergence?**
When threads of one warp take different branches, the warp executes both paths one after the
other with some threads masked off. In my kernels it only occurs at bounds checks and in
reduction trees (where most threads idle at the last levels). One reason warp shuffles beat the
shared-memory tree.

**What is `__syncthreads()`?**
A barrier for all threads of a block: none continues until all arrive. It's needed when threads
communicate through shared memory. In tiled GEMM: once after loading the tile (so it's complete)
and once after computing (so nobody overwrites it early). It must be reached by all threads, so
never put it inside a divergent `if`, and inactive threads must not `return` early. In the fused
attention kernel, inactive warps keep running for exactly this reason.

**What is `__shfl_sync()`?**
A warp-level instruction for reading another lane's register directly, with no shared memory and
no barrier. I used `__shfl_down_sync` for tree reductions (offsets 16, 8, 4, 2, 1) and
`__shfl_sync(mask, v, 0)` to broadcast lane 0's result. The mask `0xffffffff` promises that all
32 lanes participate.

**Why use tiling?**
To create data reuse. Without it, each GEMM thread re-reads its row and column from global
memory. With 32×32 tiles each value loaded is used 32 times, which raises arithmetic intensity
from 0.25 to 8 FLOP/byte. Measured: DRAM reads fell from 199 MB (v2) to 147 MB (v3), and to 33 MB
with register tiling (v4).

**What causes poor CUDA performance?**
- uncoalesced access (16.5 sectors/request in my v1);
- no data reuse;
- too little parallelism (decode attention: 0.07 waves);
- launch overhead for tiny kernels (a ~2.7 µs floor);
- PCIe copies (end-to-end vector add was 2.6× slower than the CPU);
- shared-memory bank conflicts;
- register spills;
- excessive synchronization;
- low precision accumulation (correctness, not speed).

---

## Part C — GPU architecture

**What is an SM?**
A Streaming Multiprocessor: the GPU's core unit, with its own schedulers (4 on Turing),
registers, shared memory/L1, FP32/INT units, special-function units and Tensor Cores. Blocks are
assigned to SMs. A T4 has 40.

**What is memory bandwidth?**
Bytes per second that memory can deliver. T4: 320 GB/s theoretical, from 2 × memory clock × bus
width. I reached 82% with vector add. Memory-bound kernels are judged in GB/s against this.

**What is arithmetic intensity?**
FLOPs per byte moved. Vector add: 1/12. Ideal 1024³ GEMM: about 170. Compared with the ridge
point (peak FLOP/s ÷ peak bandwidth = 25.4 on T4), it tells you whether a kernel is memory- or
compute-bound.

**Compute-bound vs memory-bound?**
Below the ridge point, performance is limited by bandwidth (vector add, softmax, LayerNorm,
decode GEMV), so move fewer bytes. Above it, it's limited by math throughput (large GEMMs), so use
faster math: Tensor Cores, FP16. Nsight Compute's Speed of Light shows which.

**What is cache?**
Hardware-managed fast memory holding recently used data: per-SM L1 and a GPU-wide L2 (4 MB on
T4). Measured effects: small working sets that fit in L2 showed "bandwidth" above DRAM peak (728
GB/s for vector add at 3 MB), and warp-per-row softmax re-read rows from DRAM because ~5 MB of
rows in flight exceeded L2.

**Why does memory access pattern matter?**
Memory is moved in 32-byte sectors and 128-byte lines; scattered access wastes most of each
transfer. The same arithmetic with a different thread mapping was 9.5× faster (GEMM v1 → v2).
Inside shared memory, the pattern decides bank conflicts: padding `[32][33]` gave 0 conflicts
on a transposed store.

---

## Part D — AI / Transformer inference

**What is GEMM?**
General matrix multiply, C = A·B, with 2·M·N·K FLOPs. `C[row][col] = Σ_k A[row][k]·B[k][col]`;
row-major indices `A[row*K+k]`, `B[k*N+col]`, `C[row*N+col]`.

**Why is GEMM important in Transformers?**
Almost all FLOPs are GEMMs: Q/K/V projections, QKᵀ, PV, output projection, two MLP layers. In
prefill they are large and compute-bound; in decode (M = 1) they become matrix-vector products
and memory-bound. Measured: my coalesced v2 reached 205–224 GB/s on decode shapes, while the
tiled versions wasted 31 of 32 tile rows.

**What is attention?**
`softmax(QKᵀ/√d)·V` per head. Each token's query is scored against every key; softmax turns the
scores into weights; the output is the weighted average of the values. √d keeps the scores near
unit size. Causal masking hides future tokens. The score matrix is seq², which is 192 MB at seq
2048 for 12 heads in my unfused version.

**What is KV cache?**
In autoregressive decoding, keys and values of past tokens don't change under causal masking, so
they're stored per layer and reused. Each new token computes q, k, v only for itself and attends
to the cache: `q_len = 1`. Cost: 2 × layers × hidden × bytes per token (36 KB in FP16 for GPT-2
small; 512 KB for LLaMA-7B). Measured: one decode step with the cache was **30× cheaper** than
recomputing attention at a 2,048-token context.

**What is softmax?**
`exp(x_i)/Σ exp(x_j)` over a row: positive outputs summing to 1. It must subtract the row max
first, because exp overflows FP32 above about 88.7 and FP16 above about 11. Shifting by a
constant doesn't change the result. On the GPU it's two reductions (max, sum) plus an
elementwise step.

**What is LayerNorm?**
Per token: subtract the mean, divide by √(variance + ε), multiply by γ and add β (both learned,
per feature). Use the two-pass variance: the one-pass E[x²] − mean² gave 2.125 instead of 0.086
in my float demonstration (catastrophic cancellation). ε prevents 0/0 on constant rows.

**Why FP16?**
Half the bytes: memory capacity (a 7B model is 14 GB instead of 28 GB), up to 2× for
memory-bound ops, and Tensor Core throughput (cuBLAS FP16 measured 35.9 TFLOP/s vs 3.7 TFLOP/s
FP32 on the T4). Cost: about 3 decimal digits, and a maximum value of 65,504.

**Why FP32 accumulation?**
The running sum grows; in FP16 its rounding step grows with it, so small terms get lost
(1000 + 0.25 = 1000 in FP16), and large sums overflow. Measured at K = 16,384: FP16 accumulation
error 3.6e-2 vs 4.4e-6 with FP32, and 256/256 outputs became inf in my overflow test. A product of
two FP16 values is exact in FP32, so only the additions round.

**What is quantization?**
Storing values as small integers plus a scale: q = round(w/scale), w ≈ q·scale. INT8 is 1 byte
per weight, INT4 is half a byte. Per-channel or per-group scales handle outliers. Weight-only
quantization dequantizes inside the GEMM and mainly speeds up memory-bound decode. I implemented it
in an INT8 weight-only decode GEMV: same kernel for FP32, FP16 and INT8 weights, one scale per
output row applied once after the sum, and an outlier experiment comparing per-row with per-tensor
scales. **A100**, 7B-class layer (11008 × 4096): FP16 weights 1.9× faster than FP32, INT8 2.2×
with the first kernel and **3.2×** after fixing its L1 bottleneck.

**What is the difference between weight-only quantization (W8A16) and W8A8?**
Weight-only stores the weights in INT8 but converts them back to floating point inside the kernel,
so the math and the activations stay FP16/FP32. It saves memory and bandwidth, which is what
decode needs. W8A8 also quantizes the activations and runs the math on INT8 (or FP8) Tensor
Cores, which speeds up compute-bound prefill but needs activation calibration. Mine is weight-only
with FP32 activations (W8A32), on CUDA cores.

**How did you quantize the weights?**
Symmetric INT8 on the CPU, once, before inference: for each output row, scale = max|w| / 127,
q = round(w / scale) clamped to ±127. Checked by hand: w = [0.12, −0.5, 0.31, 0.02] →
q = [30, −127, 79, 5], scale = 0.5/127. An all-zero row gets scale 0. Real engines also quantize
offline; the kernel only reads q and the scales.

**Why apply the scale after the sum instead of dequantizing each weight?**
Within a row the scale is constant, so Σ x·(q·s) = s·Σ x·q exactly. The kernel accumulates x·q in
FP32 and multiplies by s once per output: K − 1 fewer multiplies per row, with the same result.
INT8 → float conversion is exact, so the only rounding is the FP32 accumulation.

**Why did INT8 weights give only 2.2×, not 4×, over FP32?**
Measured on the A100. DRAM bytes did drop exactly 4× (180.4 → 45.2 MB), so the weights were read
once. But every weight is multiplied by an FP32 activation that each warp reads through L1, and
that traffic doesn't shrink with the weights. Nsight Compute: L1/TEX throughput 21% (FP32) →
51% (FP16) → **94% (INT8)**, while DRAM fell to 49%. The bottleneck moved from DRAM to L1. A
simple model of the sectors (activation loads use half of each 32-byte sector for FP16/INT8)
matched the measured counts for all three kernels exactly.

**How did you fix it, and why did 2 rows per warp beat 4 or 8?**
Each warp computes R outputs and reads each chunk of activations once for all R rows, so the
activation traffic drops by R. At R = 4, L1 sectors fell from 12.69 M to 4.24 M (my model said
4.23 M), L1 load from 94% to 42%, and DRAM use rose to 66%. But 4× fewer warps at 48 registers
each left only 0.64 of a wave and 34% occupancy: too few loads in flight. R = 2 kept enough
warps and was fastest on every shape: 3.23× over FP32 (MLP up) and 2.73× (MLP down). At R = 8 the
4096-row layer launched 64 blocks for 108 SMs and was slower than the original kernel.

**Per-row or per-tensor scales: does it matter?**
Only when there are outliers. With uniform weights both gave 0.4% relative RMS error, because
every row had the same max. With 16 outlier weights (|w| = 50 among weights ≤ 1), one per-tensor
scale gave every weight a step of 0.39 and **20%** error; per-row scales confined the damage to
the 16 affected rows: 1.5% RMS error. That's why real methods use per-channel or per-group scales
(and why GPTQ/AWQ/SmoothQuant exist).

**How did you verify the INT8 kernel?**
Two references. Against the exact double product of q·scale, which isolates the kernel's own
arithmetic (tolerance 2e-5 of the output scale). And a provable bound against the original
FP32 weights: each weight is off by at most scale/2, so |y_int8 − y_exact| ≤ scale/2 · Σ|x| must
hold for every output. Both run on 11 shapes, including rows not a multiple of 16 bytes and row
counts not a multiple of the rows per warp.

---

## Part E — Performance engineering

**How did you identify the bottleneck?**
Roofline first (memory- vs compute-bound), then Nsight Compute: Speed of Light, sectors per
request, DRAM bytes, bank conflicts, and stall reasons. Example: v3 GEMM's dominant stall was MIO
Throttle (the shared-memory queue), which pointed to register tiling. After it, cycles per
instruction fell from 37.9 to 12.95 and GFLOP/s rose 2.6×.

**How did you measure kernel latency?**
CUDA events around many back-to-back launches after a warm-up, averaged. End-to-end time with a
wall clock plus synchronization. I recorded the environment, ran correctness tests first, and
wrote every measurement to CSV so reported numbers trace back to rows.

**Why use CUDA events?**
Launches are asynchronous, so a CPU timer without synchronization only measures the launch. My
benchmark shows this deliberately: 0.024 ms "measured" for a 3.07 ms kernel. Events are
timestamps recorded by the GPU in its queue, so they measure GPU time without CPU noise. Caveat:
for tiny kernels, the span includes idle gaps between launches, so you measure launch overhead.

**Why can the GPU appear faster or slower depending on workload size?**
Fixed costs (launch ~ µs, PCIe transfer) dominate small problems. The GPU also needs enough
parallel work to fill its SMs. Measured: vector add at n = 1,024 runs at about 0.1× the CPU's
speed (a 2.7 µs launch-bound kernel against a sub-microsecond CPU loop); at 2²⁶ the kernel is
21.8× faster. GEMM v4 loses to v2 at n = 128 (4 blocks for 40 SMs) but wins
2.6× at 1,024. Cache also changes with size: small inputs fit in L2.

**Why doesn't increasing threads always improve performance?**
Once memory bandwidth is saturated, more threads only queue more requests. Vector add was flat at
about 259 GB/s for blocks of 64–1024. More threads also mean fewer registers or less shared memory
per thread, and more contention: v2 GEMM with half the warps (32×1) used 35% less DRAM traffic and
had half the stall cycles per instruction.

**How did Nsight Compute help?**
It confirmed or refuted 16 written hypotheses (13 on the T4, 3 on the A100). It confirmed the coalescing math exactly (16.5 →
2.5 sectors/request), 0 bank conflicts in the tiled kernels, and no spills. It explained the
fusion speedup by DRAM bytes (22.07 → 18.11 per element ≈ 1.21×). It overturned my guesses: 32×1
blocks won through less DRAM traffic, not a better L1 hit rate (which was worse); softmax
warp-per-row lost because it re-read 2.77× its input from DRAM; fused attention was
instruction-bound, not starved. It also showed the WMMA kernel issuing an instruction in only
5.5% of cycles, with 110 of every 124 cycles per instruction spent waiting for global loads:
the Tensor Cores were starved of data. On the A100 it explained the INT8 GEMV: weight bytes were
exactly 1 per weight, but L1 was at 94%, so the activation reads, not DRAM, were the limit.

**Why is your v4 only 60% of cuBLAS?**
Measured: a tail effect (2.13 waves; the profiler estimates up to 33%), occupancy capped at 75%
by 72 registers per thread, and latency (0.86 eligible warps per scheduler). cuBLAS also uses
vectorized loads, double-buffering and auto-tuned tile shapes.

**Why is your fused attention slower than unfused for non-causal attention?**
It's instruction-issue-bound (FMA pipe 60% busy, DRAM 0.4%). Processing one key at a time costs
shuffles and exponentials per key, while the unfused path uses tiled GEMMs. Fusion removes bytes,
not instructions. Real FlashAttention processes query × key tiles as Tensor Core GEMMs. With
causal masking mine is 1.2× faster, because it skips future keys.

**Did you test the kernels inside a real model?**
Yes: GPT-2 from Hugging Face, decoding with every linear layer, LayerNorm and the attention on my
kernels (PyTorch only for bias, GELU, residual and cache writes). On an A100, FP32 logits matched
Hugging Face to 1.4e-6 of the logit scale at every prompt position, and all 32 greedy tokens were
identical, for GPT-2 small and GPT-2 XL. FP16 weights also matched all 32. INT8 weights changed
GPT-2 small's logits by 1.8% and its text diverged after 4 tokens; on GPT-2 XL the change was 0.8%
and all 32 tokens matched. One kernel needed a change: the attention kernel assumed a packed KV
cache, and real caches are preallocated, so it got a capacity (stride) parameter.

**Why didn't INT8 make GPT-2 faster end to end?**
Decoding at batch 1 is launch-bound: about 15 kernel launches per layer, ~8.5 µs each including
dispatch. GPT-2 small: ~185 launches take ~1.6 ms, while reading all 494 MB of FP32 weights takes
~0.25 ms, so 4× fewer weight bytes changed the time by under 4%. GPT-2 XL (48 layers): ~725
launches give a floor of ~6.2 ms. FP32 (7.5 ms) was above it, so FP16 helped (6.5 ms, 1.16×), but
INT8 (6.5 ms) just stayed on the floor: its 0.9 ms of weight reads are hidden behind the launches.
So the next step is fewer launches (CUDA Graphs, fusing bias/GELU/residual into my kernels), not a
faster GEMV. My version was also ~5× faster than Hugging Face, but Hugging Face FP32 and FP16 took
the same time, so that's Python overhead, not kernels.

**How did you verify correctness?**
CPU references in double precision. Tolerances derived from the arithmetic (they grow with K or
the row length), validated by CPU emulation of each kernel's summation order. Poison values in
outputs, non-square and edge shapes (65×63×9, 255×257×129, M = 1, K = 0), hand-computed examples
(4×4 GEMM, LayerNorm, attention), stability cases (softmax of values near 1000), and for INT8 an
error bound that must hold for every output. 7 test programs, all passing on the A100 (the T4
run had 6; quantization was added later).

---

## Part F — C++

**Pointers.** A variable holding an address. `const float* a` = pointer to read-only floats.
Pointer arithmetic moves in elements: `x + row * cols` is the start of a row. Device pointers
(from `cudaMalloc`) must not be dereferenced on the CPU.

**References.** An alias for an existing object; it can't be null or reseated. Used to avoid
copies (`const std::vector<float>&`) and in `float (&vals)[N]` (a reference to an array), which
keeps the array size in the type so loops unroll and the array stays in registers.

**const.** A promise not to modify: `const float*` inputs, `const` member functions
(`data() const`), `constexpr` for compile-time constants (tile sizes).

**RAII.** Resource Acquisition Is Initialization: the constructor acquires, the destructor
releases. `DeviceBuffer<T>` calls `cudaMalloc` in its constructor and `cudaFree` in its
destructor, so GPU memory can't leak. Copy is deleted (it would double-free); move transfers
ownership. `GpuTimer` does the same for CUDA events, `CsvLog` for files.

**Classes.** Types bundling data with operations and controlling access (`private` members).
`DeviceBuffer`, `GpuTimer`, `CpuTimer`, `CsvLog`.

**Templates.** Compile-time generic code. `DeviceBuffer<T>` works for `float` and `__half`;
`matmul_tiled_kernel<TILE>` makes the tile size a compile-time constant, which shared array
sizes and loop unrolling require; `matmul_tiled_fp16_kernel<TILE, AccT>` switches the
accumulator type. Each instantiation is a separately compiled kernel.

**Memory management.** Host: `std::vector` (RAII). Device: `cudaMalloc`/`cudaFree` wrapped in
RAII; copies with `cudaMemcpy`. Keep data on the GPU: PCIe was about 55× slower than the kernel
in the vector-add test. PyTorch uses a caching allocator, so measure memory with its allocator
statistics, not `nvidia-smi`.

**Compilation.** `nvcc` splits `.cu` files: device code is compiled for a GPU architecture
(`sm_75` for T4; CMake's `native`), host code by the host compiler (`g++`). Release mode `-O3`;
`-lineinfo` maps SASS to source lines for Nsight without slowing the code.

**Linking.** Object files are combined into executables. All kernels and utilities form one
static library (`libctp_core.a`), linked into each test and benchmark, so kernels are compiled
once. The CUDA runtime is linked automatically by CMake for CUDA targets.

---

## Part G — Rapid-fire (one sentence each)

- Ceiling division for the grid size? `(n + block − 1) / block`, plus a bounds check.
- Why 256 threads per block? A multiple of 32, enough warps to hide latency, several blocks per
  SM; measured flat from 64 to 1024 for vector add.
- Bank conflict fix for a transposed tile? Pad to `[32][33]`.
- Why two barriers per tile? Data ready before use; tile fully used before overwrite.
- Why does `ncu` report longer times? It locks clocks to base (585 MHz on the T4) and flushes
  caches; compare percentages.
- Why is GPU-Util in `nvidia-smi` misleading? It means "a kernel was running", not that the SMs
  were busy.
- Why can bandwidth exceed 100% of peak in a benchmark? The working set fit in L2.
- What is the tail effect? A partial last wave of blocks leaves SMs idle (v4: 2.13 waves).
- Why quantize weights for decode? Decode reads every weight once per token: fewer bytes per
  weight means fewer bytes per token.
- INT8 range in symmetric quantization? −127 to 127 (−128 unused, so the range is symmetric).
- Why didn't GPT-2-sized layers speed up with INT8? 2–9 MB of weights take 5–7 µs: launch and
  latency, not bytes, set the time.
- Why can a smaller data type stop helping? The bottleneck moves: INT8 weights moved it from DRAM
  to L1 (the activation reads).
