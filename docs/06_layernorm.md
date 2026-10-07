# 06 — LayerNorm and Kernel Fusion

Files covered: `kernels/layernorm.cu`, `kernels/layernorm.cuh`, `kernels/warp_reduce.cuh`
(`block_reduce_sum`), `src/cpu/cpu_ops.cpp` (`cpu::layernorm`, `cpu::add_layernorm`),
`tests/test_layernorm.cu`, `src/benchmark/bench_layernorm.cu`.

New concepts: mean/variance as reductions, numerical cancellation, hierarchical (block-level)
reduction, caching a row in registers, **kernel fusion**.

---

## 1. What problem are we solving?

Inside a Transformer, each token is represented by a vector of `hidden` numbers (768 for GPT-2
small, 4096 for LLaMA-7B). As this vector passes through many layers, its values can drift to
very large or very small magnitudes, which makes the model unstable. **Layer normalization**
rescales each token's vector to a standard range before the next sub-layer uses it.

For one row x (one token, `cols` = hidden size):

| Term | Formula | Meaning |
|---|---|---|
| **mean** μ | (1/cols) · Σ xᵢ | the average value of the row |
| **variance** σ² | (1/cols) · Σ (xᵢ − μ)² | the average squared distance from the mean: how spread out the row is |
| **epsilon** ε | a tiny constant, 10⁻⁵ | added to the variance so we never divide by 0 (a constant row has σ² = 0) |
| **normalize** | x̂ᵢ = (xᵢ − μ) / √(σ² + ε) | the row now has mean 0 and variance ≈ 1 |
| **gamma** γ | learned vector, `cols` values | per-feature **scale** the model learned during training |
| **beta** β | learned vector, `cols` values | per-feature **shift** the model learned |
| **output** | yᵢ = x̂ᵢ · γᵢ + βᵢ | |

γ and β are the same for every row (every token). They belong to the layer, not to the input.
The variance divides by `cols` (the "population" variance), which is what PyTorch's LayerNorm
does.

### Where LayerNorm appears in inference

A GPT-style block (pre-LN), executed for every layer and every token:

```
h1 = x  + Attention( LayerNorm1(x)  )        ← residual add, then…
h2 = h1 + MLP      ( LayerNorm2(h1) )        ← …LayerNorm of the result
```

So every layer runs "residual add → LayerNorm" twice. That pair is exactly what our fused
kernel implements. (LLaMA-family models use **RMSNorm**, which is the same idea without
subtracting the mean: y = x / √(mean(x²) + ε) · γ. Everything in this document applies to it.)

---

## 2. Hand example (first test in `tests/test_layernorm.cu`)

x = [1, 2, 3, 4], γ = [1, 2, 1, 0.5], β = [0, 0, 1, −1], ε = 10⁻⁵

```
mean        = (1 + 2 + 3 + 4) / 4                 = 2.5
x − mean    = [−1.5, −0.5, 0.5, 1.5]
squared     = [2.25, 0.25, 0.25, 2.25]
variance    = 5 / 4                                = 1.25
1/√(1.25 + 0.00001)                                = 0.8944236
x̂           = [−1.3416355, −0.4472118, 0.4472118, 1.3416355]
x̂ · γ       = [−1.3416355, −0.8944236, 0.4472118, 0.6708178]
y = x̂·γ + β = [−1.3416355, −0.8944236, 1.4472119, −0.3291823]
```

---

## 3. Numerical accuracy: two lessons measured in this project

### Lesson 1: never compute variance as E[x²] − mean²

Mathematically, σ² = mean(x²) − mean(x)². This "one-pass" formula is tempting, because a
single loop can accumulate Σx and Σx² together. In float it can be badly wrong.
`test_layernorm` prints this demonstration (1024 values in [999.5, 1000.5]; true variance
≈ 0.0862):

```
two-pass  mean((x − mean)²)  = 0.086159
one-pass  mean(x²) − mean²   = 2.125000      ← 25× too large
```

**Why:** x² ≈ 1,000,000. A float keeps only about 7 significant digits, so each x² carries a
rounding error around 0.06. mean(x²) and mean² are both ≈ 1,000,000 and differ by only 0.086.
Subtracting two large, nearly equal numbers leaves mostly rounding error. This is called
**catastrophic cancellation**.

The **two-pass** method first computes the mean, then sums (x − mean)². Those differences are
small, so there is no cancellation. It costs a second pass over the data. **v3 makes that second
pass free**, because the row is already in registers (§7).

(A third option, **Welford's algorithm**, computes a stable variance in one pass by updating
a running mean, much like the running max in online softmax, 05 §10. We don't need it here.)

### Lesson 2: the order of summation changes accuracy

A CPU emulation of each kernel's exact float summation order, compared with the double-precision
reference, gave these largest output errors:

| Input range | v1 (one thread adds sequentially) | v2 (32 partial sums + tree) | v3 (256–1024 partial sums + tree) |
|---|---|---|---|
| [−2, 2] | 3.1e-6 | 4.8e-7 | 4.8e-7 |
| [50, 52] (large mean) | **4.4e-4** | 2.2e-5 | 1.3e-5 |
| [−300, 300] | 3.8e-6 | 4.8e-7 | 4.8e-7 |

A sequential sum adds each small value to an ever-growing total and loses the low digits every
time. A tree adds numbers of similar size, so less is lost. **Parallel reductions are faster and
also more accurate.** The test tolerances are based on these measurements: 5e-5 in general, and
2e-3 for the [50, 52] range, so that v1's legitimate error passes.

---

## 4. CPU version

`cpu::layernorm`: two-pass in `double`. `cpu::add_layernorm`: computes `h = x + residual` in
**float** (exactly what the GPU does, so h can be compared bit-for-bit), then LayerNorm of h.

---

## 5. GPU idea

The same structure as softmax: rows are independent; inside a row there are **two reductions**
(sum for the mean, then sum of squared differences for the variance) and then an **elementwise**
step.

| Version | Work per row | Reduction | Reads of x |
|---|---|---|---|
| v1 | 1 thread | sequential | 3 (uncoalesced) |
| v2 | 1 warp | shuffles (`warp_reduce_sum`) | 3 (coalesced) |
| v3 | 1 block | warp shuffles + shared memory (`block_reduce_sum`) | **1**, then registers |
| fused | 1 block | as v3 | reads x and residual once, writes h and y |

v1 and v2 are the same patterns as softmax v1 and v3 (05 §5, §8). Their line-by-line
explanation carries over, with "max" replaced by "sum". The new ideas are in v3.

---

## 6. Block-level reduction (`block_reduce_sum<BLOCK>`)

A warp shuffle only reaches 32 threads. v3 uses up to 1024 threads per row, so we need a
reduction over the whole **block**. It has two levels:

```
BLOCK = 256 threads = 8 warps, each thread holds a partial sum v

Level 1 (shuffles, inside each warp):
  warp 0: lanes 0..31  ──warp_reduce_sum──► total w0  (in every lane)
  warp 1: lanes 0..31  ──warp_reduce_sum──► total w1
  ...
  warp 7               ──warp_reduce_sum──► total w7

  lane 0 of each warp writes its total:  shared = [w0 w1 w2 w3 w4 w5 w6 w7]
  __syncthreads()

Level 2 (shuffles again, in EVERY warp):
  each lane l < 8 reads shared[l], lanes 8..31 use 0
  warp_reduce_sum → w0 + w1 + … + w7 = block total, in every thread
  __syncthreads()
```

Line by line:

```cpp
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "...");
```
- `static_assert` checks a condition **at compile time**. A wrong BLOCK becomes a compile
  error instead of a silent bug.

```cpp
    constexpr int NUM_WARPS = BLOCK / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
```
- After `warp_reduce_sum`, every lane of the warp holds the warp total (05 §7). Only one lane
  needs to store it, so we use lane 0.
- The barrier makes sure all `NUM_WARPS` totals are in shared memory before anyone reads them.

```cpp
    v = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    v = warp_reduce_sum(v);
    __syncthreads();
    return v;
```
- **Every** warp (not only warp 0) loads the ≤ 32 warp totals and reduces them. So every thread
  gets the block total without a separate broadcast step. The redundant work is a few
  instructions.
- Lanes beyond `NUM_WARPS` contribute 0, which adds nothing.
- The final barrier: the function is called twice in a row (for the mean, then the variance).
  Without this barrier, a fast warp could start the second call and overwrite `shared[warp]`
  while a slow warp is still reading the first call's values.

**Compared with softmax v2's pure shared-memory tree** (05 §6): a 256-thread tree needs 8
levels, each with a barrier. This version needs **2 barriers**. Most of the work happens in
registers.

**Rule:** the function contains `__syncthreads()`, so every thread of the block must call it,
never from inside a divergent `if`.

---

## 7. v3: the row cached in registers

### Choosing the configuration

```cpp
if      (cols <= 1024)  layernorm_block_kernel<256, 4> <<<rows, 256 >>>(...);
else if (cols <= 4096)  layernorm_block_kernel<512, 8> <<<rows, 512 >>>(...);
else if (cols <= 16384) layernorm_block_kernel<1024,16><<<rows, 1024>>>(...);
```
- `BLOCK × PER_THREAD` = how many columns fit in registers. Each thread holds `PER_THREAD`
  floats.
- `PER_THREAD` must be a **compile-time** constant, because the array `vals[PER_THREAD]` must
  live in registers (04 §21). So we compile three versions (template instantiations) and pick
  one at run time.
- Example: hidden size 768 → `<256, 4>` → 1024 slots, of which 768 are used (each thread
  holds 3 real values and 1 padding zero). Hidden size 4096 → `<512, 8>` → exactly full.

### Loading the row

```cpp
float vals[PER_THREAD];
for (int i = 0; i < PER_THREAD; ++i) {
    const int c = threadIdx.x + i * BLOCK;
    vals[i] = (c < cols) ? x_row[c] : 0.0f;
}
```
Thread `tid` holds columns `tid, tid + BLOCK, tid + 2·BLOCK, …`. At each `i`, the block reads
BLOCK consecutive floats: **coalesced** ✔. Positions past the end of the row hold 0.

Example, cols = 768, BLOCK = 256: thread 5 holds columns 5, 261, 517 and a padding 0
(5 + 768 = 773 ≥ 768).

### `normalize_row_from_registers`

```cpp
template <int BLOCK, int PER_THREAD>
__device__ __forceinline__ void normalize_row_from_registers(float (&vals)[PER_THREAD], ...)
```
`float (&vals)[PER_THREAD]` reads as "**vals is a reference (`&`) to an array of PER_THREAD
floats**". The parentheses are needed: without them, `float& vals[N]` would mean "an array of
references", which is not allowed. A reference means the function uses the caller's array
directly, with no copy. And because the size is part of the type, the compiler can unroll every
loop over it, so the array stays in registers.

Inside:
1. **Mean**: each thread sums its `PER_THREAD` values (padding zeros add nothing) →
   `block_reduce_sum` → divide by `cols`.
2. **Variance**: each thread sums `(vals[i] − mean)²` **only for real columns**
   (`if (c < cols)`). A padding zero would contribute (0 − mean)², which is not zero, so it
   must be skipped. The values come from **registers**: the two-pass stable method costs no
   extra memory traffic.
3. `inv_std = rsqrtf(var + eps)`: `rsqrtf` computes 1/√x in one fast hardware instruction
   (about 2 ulp accurate), cheaper than `1.0f / sqrtf(x)`.
4. **Write**: `y = (vals[i] − mean) · inv_std · gamma[c] + beta[c]`. `gamma`/`beta` reads are
   coalesced, and since every row reads the same γ and β, they stay hot in L1/L2.

### Bytes moved per element

| Version | Global reads of x | Writes | Total |
|---|---|---|---|
| v1, v2 | 3 × 4 B | 4 B | 16 B (repeat reads may hit the cache) |
| v3 | 1 × 4 B | 4 B | **8 B = the minimum** |

LayerNorm is memory-bound (about 7 FLOPs per element against 8 bytes), so v3 should be the
fastest. Its GB/s (computed with the minimum 8 B/element) is the number to compare against peak
bandwidth.

**Limits of v3:**
- A row longer than 16,384 does not fit in registers (the launcher stops with an error; use v2).
- Only one block per row: with very few rows (e.g. 1 token during decoding), most SMs are idle.
  The same issue as softmax (05 §11).
- `<1024, 16>` uses many registers per thread. Check occupancy in Nsight (Phase 10).

---

## 8. Kernel fusion: residual add + LayerNorm in one kernel

### The unfused pipeline (what eager PyTorch does)

```
kernel 1 (vector_add):   read x, read residual  →  write h          12 B/element
kernel 2 (layernorm):    read h                 →  write y           8 B/element
                                                            total   20 B/element, 2 launches
```

h is written to global memory by kernel 1, then immediately read back by kernel 2.

### The fused kernel (`add_layernorm_kernel`)

```cpp
if (c < cols) {
    vals[i] = x[offset + c] + residual[offset + c];   // the add, in registers
    h[offset + c] = vals[i];                           // h is still written (the next layer needs it)
} else {
    vals[i] = 0.0f;
}
normalize_row_from_registers<BLOCK, PER_THREAD>(vals, gamma, beta, y + offset, cols, eps, shared);
```

```
fused kernel:  read x, read residual → (h in registers) → write h, write y      16 B/element, 1 launch
```

- **Saved:** one read of h (4 B/element, 20%) and one kernel launch.
- **Not saved:** writing h. The next layer's residual connection needs it (§1).
- The rest of the kernel is v3 unchanged. That is why `normalize_row_from_registers` is a shared
  helper: the fused kernel only differs in the load step.

### What speedup to expect

- From bytes alone: up to 20/16 = **1.25×** for large shapes, where the kernels are
  bandwidth-bound.
- For small shapes (e.g. 512 × 768), each kernel lasts only a few µs, so the saved **launch** is a
  noticeable share of the time. The gain can be larger than 1.25×.

Read the actual number from Experiment B. If it is smaller than 1.25×, think about what else
limits the fused kernel (one block per row, register use → occupancy) and check in Phase 10.

This exact fusion exists in production inference engines (for example "fused add + RMSNorm" in
TensorRT-LLM and vLLM), together with many others (bias + activation, QKV split + rotary
embedding). Fusion is one of the most effective inference optimizations for memory-bound ops.

---

## 9. Synchronization summary

| Version | Barriers per row | Why |
|---|---|---|
| v1 | 0 | one thread per row |
| v2 | 0 | one warp per row; shuffles need no barriers |
| v3 / fused | 4 (2 per `block_reduce_sum` call × 2 calls) | totals pass between warps through shared memory |

## 10. Common mistakes

1. Computing variance as E[x²] − mean² (catastrophic cancellation, §3).
2. Forgetting ε: a constant row gives 0/0 = NaN. The test uses constant rows.
3. Including padding slots in the variance sum (adds (0 − mean)² per padding slot).
4. Calling `block_reduce_sum` from inside a divergent `if` (it contains barriers).
5. Two consecutive block reductions without the trailing barrier (shared memory overwritten
   while being read).
6. Indexing `gamma`/`beta` with the flattened index instead of the column index (`gamma[c]`).
   They are per feature, not per element.
7. Register arrays with run-time sizes or indices → spill to local memory.
8. Fusing the add but forgetting to write h. The next layer needs the residual stream.

## 11. Interview explanation (~90 seconds)

> "LayerNorm normalizes each token's hidden vector: subtract the mean, divide by
> √(variance + ε), then apply learned γ and β. On the GPU it is two row reductions plus an
> elementwise step. I compute variance with the two-pass method, because E[x²] − mean² suffers
> catastrophic cancellation; I measured it giving 2.1 instead of 0.086 for values around 1000.
> My best kernel uses one block per row and keeps the row in registers, so the input is read from
> global memory exactly once and the second pass is free. The block reduction is hierarchical:
> warp shuffles first, then one shared-memory slot per warp, then shuffles again, which is only
> two barriers. I also fused the residual add into the same kernel: the add happens in registers,
> h is written once for the next layer, and the read-back of h is gone. That cuts traffic from 20
> to 16 bytes per element and saves a launch, which is the same kind of fusion production
> engines use. I also found that the tree reductions are more accurate than a sequential sum,
> which I measured against a double-precision reference."

## 12. What to remember

- mean, variance (population), ε against division by zero, γ/β per feature.
- Two-pass variance; avoid E[x²] − mean².
- Block reduction = warp shuffles + one shared slot per warp + shuffles: 2 barriers.
- Keeping the row in registers turns a 3-read kernel into a 1-read kernel.
- Fusion removes intermediate global-memory round trips and launches. It is the main lever for
  memory-bound ops.
- Tree reductions are also more accurate than sequential sums.

---

## 13. File guide

### `kernels/layernorm.cu`
1. **What:** LayerNorm v1–v3, the fused add + LayerNorm, and launchers. 2. **Why:** a second
reduction workload, register caching, and the project's fused operation. 3. **Inputs:** device
`x` (rows × cols), `gamma`, `beta` (cols), `eps`; for fused also `residual`. 4. **Outputs:** `y`;
for fused also `h`. 5. **Data flow (v3):** global → registers → (warp shuffle → shared → warp
shuffle) → registers → global. 6. **Functions:** `layernorm_naive_kernel`,
`layernorm_warp_kernel`, `normalize_row_from_registers`, `layernorm_block_kernel<BLOCK, PER_THREAD>`,
`add_layernorm_kernel<BLOCK, PER_THREAD>`, `gpu::layernorm_*`, `gpu::add_layernorm`.
7. **Variables:** `vals`, `mean`, `inv_std`, `shared`, `offset`. 8. **CUDA concepts:** block
reduction, register arrays, `rsqrtf`, templates, `static_assert`. 9. **Memory:** §7 table.
10. **Thread mapping:** thread `tid` ↔ columns `tid + i·BLOCK`. 11. **Sync:** §9.
12. **Performance:** memory-bound; v3 moves the minimum bytes. 13. **Mistakes:** §10.

### `kernels/warp_reduce.cuh` (extended)
`block_reduce_sum<BLOCK>(v, shared)`: §6.

### `kernels/layernorm.cuh`
Launchers, `LAYERNORM_EPS = 1e-5f`, `layernorm_versions()`.

### `src/cpu/cpu_ops.cpp`
`cpu::layernorm` (two-pass, double) and `cpu::add_layernorm` (float add, then LayerNorm).

### `tests/test_layernorm.cu`
The one-pass variance demonstration, the hand example, constant rows (σ² = 0 → y = β), 13 shapes
(including 768, 4096, all three capacity tiers and their boundaries 1024/1025, 4096/4097,
16384) × 3 value ranges for each version, and the fused kernel (h bit-exact, y within tolerance).
**Verified before any GPU run:** a CPU emulation of the float summation order of v1, v2 and v3
produced the error table in §3, and the hand example's expected values.

### `src/benchmark/bench_layernorm.cu`
Experiment A (versions vs CPU, GB/s vs peak) and Experiment B (unfused 2-kernel pipeline vs fused
kernel, both verified first).

### `python/benchmark.py` (extended)
Adds `torch add + layer_norm`, the eager PyTorch two-kernel pipeline, as the framework baseline
for the fused kernel.
