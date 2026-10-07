# 08 — Precision: FP32, FP16, FP32 Accumulation, Tensor Cores, Quantization

Files covered: `kernels/precision.cuh`, `kernels/matmul_fp16.cu`, `kernels/matmul_wmma.cu`
(GEMM v5), `src/utils/precision_utils.cuh`, `src/cpu/cpu_ops.cpp` (`cpu::matmul_f64`),
`tests/test_precision.cu`, `src/benchmark/bench_precision.cu`.

---

## 1. What problem are we solving?

So far every number was a 32-bit `float`. Production inference almost never runs in FP32: model
weights are stored in 16 bits (FP16 or BF16), often in 8 or 4 bits. This phase answers:

1. What exactly is lost when we use fewer bits? (§2, §3)
2. Why is it worth it? (§4)
3. Why do we still **add up** in FP32 when inputs are FP16? (§5)
4. How do Tensor Cores make FP16 matrix multiply much faster? (§7, GEMM v5)
5. What is quantization? (§9)

---

## 2. Floating point, from scratch

### The idea: scientific notation in base 2

In decimal scientific notation, 6,020 = 6.02 × 10³: a few **significant digits** (6.02) and a
**power** (10³) that sets the size. A floating-point number does the same in binary:

```
value = (−1)^sign × 1.mantissa × 2^(exponent − bias)
```

- **sign** (1 bit): + or −.
- **exponent**: sets the scale (the "power of 2"). More exponent bits → a larger **range**
  (bigger maximum, smaller minimum).
- **mantissa** (also called fraction or significand): the significant digits after the leading
  "1.". More mantissa bits → more **precision** (finer steps between neighboring numbers).
- **bias**: a constant subtracted from the stored exponent, so that negative powers can be
  stored with unsigned bits (127 for FP32, 15 for FP16).

### The formats

```
FP32  │s│ exponent: 8 │        mantissa: 23            │   32 bits
FP16  │s│ exp: 5 │ mantissa: 10 │                          16 bits
BF16  │s│ exponent: 8 │ mant: 7 │                          16 bits
```

| Format | Exponent bits | Mantissa bits | Largest value | Smallest normal | Machine epsilon (step after 1.0) | ≈ decimal digits |
|---|---|---|---|---|---|---|
| FP32 | 8 | 23 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 2⁻²³ ≈ 1.2 × 10⁻⁷ | 7 |
| TF32 (Tensor Cores only) | 8 | 10 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 2⁻¹⁰ ≈ 9.8 × 10⁻⁴ | 3 |
| **FP16** (`__half`) | 5 | 10 | **65,504** | 6.1 × 10⁻⁵ | 2⁻¹⁰ ≈ 9.8 × 10⁻⁴ | 3 |
| BF16 | 8 | 7 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 2⁻⁷ ≈ 7.8 × 10⁻³ | 2 |

- **FP16** has decent precision but a tiny range: anything above 65,504 becomes infinity.
- **BF16** ("brain float") keeps FP32's range but has very little precision. It's popular for
  training because overflow is the bigger danger there.
- **Machine epsilon** is the gap between 1.0 and the next representable number. It is the
  relative precision: every rounding can change a value by up to half of this, relative to its
  size.

### The spacing grows with the size

Between 2ᵉ and 2ᵉ⁺¹, FP16 has exactly 1024 evenly spaced values, so the step is 2ᵉ⁻¹⁰:

| Range | Step between neighbors (FP16) |
|---|---|
| [1, 2) | 0.000977 |
| [2, 4) | 0.00195 |
| [1024, 2048) | 1 |
| [2048, 4096) | **2** |
| [32768, 65504] | 32 |

### Real roundings (`test_precision` checks all of these on your GPU)

The values were computed with a CPU emulation of IEEE FP16 rounding (round to nearest, ties to
even):

| float input | FP16 result | What happened |
|---|---|---|
| 1.0 | 1.0 | exactly representable |
| 0.1 | 0.0999755859375 | **rounding**: 0.1 has no exact binary form (not in FP32 either) |
| 3.14159 | 3.140625 | rounding to a step of 0.00195 |
| 2049 | 2048 | the step is 2 here; 2049 is exactly halfway, and **ties go to the even** neighbor |
| 65504 | 65504 | the largest FP16 value |
| 70000 | **inf** | **overflow** |
| 1e-5 | 1.0013580322265625e-05 | **subnormal**: below the smallest normal value, stored with reduced precision |
| 1e-8 | **0** | **underflow**: smaller than the smallest subnormal (5.96 × 10⁻⁸) |

---

## 3. Kinds of numerical error

| Error | Example | Where it bites in inference |
|---|---|---|
| **Rounding** | 0.1 → 0.09998 | every operation, a tiny amount |
| **Overflow** | 70000 → inf in FP16 | large activations, sums of many products, softmax without max-subtraction |
| **Underflow** | 1e-8 → 0 | tiny probabilities, small gradients (training) |
| **Swamping** | 1000 + 0.25 = **1000** in FP16 | adding small terms to a large running sum |
| **Accumulation** | errors of K additions pile up | long dot products (K = 4096+) |
| **Cancellation** | E[x²] − mean² (06 §3) | subtracting nearly equal numbers |

**Swamping, explained:** near 1000, FP16's step is 0.5. The exact sum 1000.25 is not
representable, and rounds back to 1000. The 0.25 is simply lost. (1000 + 0.3 rounds up to
1000.5, so the error is 0.2 instead.) In a dot product, the running sum grows, its step grows,
and later small products get partly or fully lost.

---

## 4. Why use lower precision for inference at all?

1. **Memory capacity.** A 7-billion-parameter model needs 28 GB in FP32 and 14 GB in FP16. That's
   the difference between fitting on a GPU or not.
2. **Memory bandwidth.** Half the bytes means memory-bound operations (decode GEMMs, softmax,
   LayerNorm, the KV cache, 03 §5) run up to **2× faster**. For LLM decoding this is the main
   benefit.
3. **Compute.** **Tensor Cores** run FP16 matrix math many times faster than FP32 on regular
   cores. NVIDIA's published peaks: T4 ≈ 65 TFLOP/s FP16 Tensor Core vs ≈ 8 TFLOP/s FP32;
   A100 ≈ 312 vs ≈ 19.5.
4. **Cache.** Twice as many values fit in L2 cache and shared memory.
5. **Neural networks tolerate noise.** They are trained with noise (dropout, data
   augmentation, stochastic gradients), so 3 significant digits in weights and activations
   barely change their outputs. Some operations still need more care (§5, §10).

---

## 5. Why FP16 inputs but FP32 accumulation?

A GEMM output is a sum of K products. The **inputs** only need to be stored. The **running
sum** keeps growing, and in FP16 its rounding step grows with it (§2), so swamping and
accumulation errors pile up. It can also overflow.

A CPU emulation of exactly our tiled kernel's arithmetic, M = N = 64, inputs in [−1, 1] rounded
to FP16, gave this error (as a fraction of the largest output):

| K | FP32 accumulator | FP16 accumulator | largest output |
|---|---|---|---|
| 64 | 2.2 × 10⁻⁷ | 2.2 × 10⁻³ | 10.9 |
| 256 | 5.7 × 10⁻⁷ | 5.2 × 10⁻³ | 19.6 |
| 1024 | 9.3 × 10⁻⁷ | 5.9 × 10⁻³ | 42.6 |
| 4096 | 1.8 × 10⁻⁶ | 1.6 × 10⁻² | 90.2 |
| 16384 | 4.1 × 10⁻⁶ | **2.8 × 10⁻²** | 157 |

FP16 accumulation is about **10,000× less accurate**, and the gap grows with K. At K = 16384 the
worst output is off by about 3% of the output's scale.

**Overflow:** with inputs in [0, 8] and K = 8192, the true sums are about 133,000. That is above
65,504, so the FP16 accumulator becomes **inf**, while the FP32 accumulator is fine.
`bench_precision` Experiment C shows this on your GPU.

**The resulting rule** (used by cuBLAS, Tensor Cores and every inference engine): **store and
multiply in low precision, accumulate in FP32, convert the final result.** It costs almost
nothing. Tensor Cores are built to multiply FP16 and add in FP32.

Two more details make FP16 × FP16 → FP32 especially clean:
- A product of two FP16 numbers (11 significant bits each) has at most 22 significant bits, so it
  fits **exactly** in a float (24 bits). Only the additions round.
- Our benchmark compares against two references: "total" error (vs the original FP32 inputs,
  including the loss from rounding inputs to FP16) and "arith" error (vs the FP16-rounded
  inputs, only the kernel's own error). For FP16 inputs with FP32 accumulation, the total
  error is dominated by **input rounding**, not by the arithmetic.

---

## 6. The FP16 tiled GEMM (`kernels/matmul_fp16.cu`)

This is GEMM v3 (04 §16) with two changes, and nothing else, so differences in speed or error
come only from precision:

```cpp
template <int TILE, typename AccT>
__global__ void matmul_tiled_fp16_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    __shared__ __half As[TILE][TILE];   // 2 bytes per element instead of 4
    __shared__ __half Bs[TILE][TILE];
    ...
    AccT sum = AccT(0.0f);              // float or __half
    ...
        multiply_add(sum, As[ty][k], Bs[k][tx]);
    ...
    C[row * N + col] = to_float(sum);   // output always FP32, so errors can be measured
```

- `__half` is CUDA's FP16 type, from `<cuda_fp16.h>`.
- `typename AccT` is a **type template parameter**: the same kernel source is compiled twice,
  once with `AccT = float` and once with `AccT = __half`.
- `AccT(0.0f)`: constructs a zero of type AccT (`__half` has a constructor from float).
- `__float2half(0.0f)` is used for the zero padding in the tiles.

### `multiply_add`: function overloading

```cpp
__device__ void multiply_add(float& acc, __half a, __half b)  { acc = fmaf(__half2float(a), __half2float(b), acc); }
__device__ void multiply_add(__half& acc, __half a, __half b) { acc = __hfma(a, b, acc); }
```

Two functions with the **same name** and different parameter types. The compiler picks the one
matching `sum`'s type. This is called **overloading**.
- The float version converts both inputs to float (exact), then does an FP32 FMA.
- The `__half` version uses `__hfma` (half FMA): it computes a·b + acc and rounds the result to
  FP16. This rounding of the running sum is where the error in §5 comes from.

### Conversion kernels

`float_to_half_kernel` (`__float2half`, round to nearest even) and `half_to_float_kernel`
(`__half2float`, always exact). Both use grid-stride loops (02 §8) with at most 4096 blocks.
`round_to_half_on_gpu` (in `precision_utils.cuh`) runs float → half → float on the GPU, so tests
see exactly the values the FP16 kernels see.

**Speed expectation:** the FP16 tiled kernel still does FP32 math on regular CUDA cores (after
converting). It saves global-memory and shared-memory **bytes**, but v3 was not limited by
global memory (04 §19). So expect it to be similar to FP32 v3, not 2× faster. The big gain
needs Tensor Cores.

---

## 7. GEMM v5: Tensor Cores with WMMA (`kernels/matmul_wmma.cu`)

### What a Tensor Core is

A regular CUDA core executes one FMA (one multiply-add on one pair of numbers) per instruction.
A **Tensor Core** is a separate unit inside the SM that multiplies **small matrices** in one
operation. A single warp-level instruction computes a whole 16 × 16 × 16 tile: 4,096
multiply-adds. The SM has a few Tensor Cores next to its CUDA cores (compute capability ≥ 7.0:
V100, T4, A100, L4, H100…).

### What "warp-level" means here

With **WMMA** (Warp Matrix Multiply-Accumulate, header `<mma.h>`), you do not program individual
threads. The **32 threads of a warp act together as one unit**:

- A **fragment** is a 16 × 16 tile held *collectively* in the registers of the 32 threads. Each
  thread holds a few elements, but **which** elements is decided by the hardware and
  deliberately hidden. You only use fragments through WMMA functions.
- Every WMMA function ends in `_sync`: **all 32 lanes must call it together** (like the
  shuffles, 05 §7).

### Line by line

```cpp
using namespace nvcuda;
```
WMMA lives in the namespace `nvcuda::wmma`. This line lets us write `wmma::…`.

```cpp
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WARPS_X = 2, WARPS_Y = 2;          // 4 warps per block, arranged 2 × 2
constexpr int THREADS_PER_BLOCK = 128;
```
For FP16 inputs, WMMA supports the tile shape 16 × 16 × 16 (and some others). Each warp
computes one 16 × 16 tile of C, so a block of 4 warps covers a 32 × 32 tile of C.

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
```
`__CUDA_ARCH__` is defined by nvcc while compiling GPU code, to the target architecture
(750 for sm_75). The WMMA instructions only exist from 700 (Volta) onward. If we ever compile for
an older GPU, the kernel body is left empty instead of failing the build. The launcher also
checks at run time (`wmma_supported()`).

```cpp
const int warp_id  = threadIdx.x / 32;        // 0..3
const int warp_row = warp_id / WARPS_X;       // 0..1
const int warp_col = warp_id % WARPS_X;       // 0..1
const int tile_row = (blockIdx.y * WARPS_Y + warp_row) * WMMA_M;
const int tile_col = (blockIdx.x * WARPS_X + warp_col) * WMMA_N;
if (tile_row >= M || tile_col >= N) return;
```
The same "global index" idea as always, with **warps** as the unit and **16 × 16 tiles** as the
output element. Example: block (1, 0), warp 3 → warp_row 1, warp_col 1 →
tile_row = (0·2 + 1)·16 = 16, tile_col = (1·2 + 1)·16 = 48. This warp computes
C[16..31][48..63]. The early return exits whole warps only, so the `_sync` functions are safe.

```cpp
wmma::fragment<wmma::matrix_a,    16, 16, 16, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b,    16, 16, 16, __half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float>                   c_frag;
```
Template arguments: the role (A operand, B operand, accumulator), the tile shape M × N × K, the
element type (FP16 inputs, **FP32 accumulator**), and for A/B the memory layout (our matrices
are row-major).

```cpp
wmma::fill_fragment(c_frag, 0.0f);
```
Accumulator = 0 (like `float sum = 0.0f`).

```cpp
for (int k = 0; k < K; k += WMMA_K) {
    wmma::load_matrix_sync(a_frag, A + tile_row * K + k, K);
    wmma::load_matrix_sync(b_frag, B + k * N + tile_col, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```
- `A + tile_row * K + k`: the address of A[tile_row][k], the **top-left corner** of the 16 × 16
  A tile (the same flat-index formula as 03 §3).
- The last argument, `K`, is the **leading dimension**: how many elements one row of A has in
  memory, so WMMA knows how far to jump to reach the next row of the tile.
- For B: the top-left is B[k][tile_col]; the leading dimension is N.
- `mma_sync(d, a, b, c)` computes d = a × b + c on the Tensor Cores. Passing `c_frag` as both
  c and d means "accumulate in place".
- The loop has the same structure as the tiled GEMM: walk along K in steps of the tile depth.

```cpp
wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
```
Write the 16 × 16 FP32 result to C[tile_row..][tile_col..], with row length N.

### Requirements, and why

- **Compute capability ≥ 7.0.** Older GPUs have no Tensor Cores.
- **M, N, K multiples of 16** (the launcher checks and stops otherwise). Tiles cannot be partial
  in this simple version, and WMMA requires 32-byte-aligned tile addresses. With multiples of
  16, every tile address is a multiple of 16 elements from the `cudaMalloc` base, which is
  aligned. Production code **pads** matrices to the tile size, or handles edge tiles separately.
  `test_precision` shows padding works: the 4 × 4 hand example, zero-padded to 16 × 16, comes out
  exact.

### Limitations of this simple v5 (be ready to discuss them)

- **No shared-memory staging.** The two warps in the same block row load the *same* A tile from
  global memory separately; the same holds for B tiles in a block column. Fast Tensor Core
  kernels (cuBLAS, CUTLASS) load tiles into shared memory once per block, double-buffer them
  (`cp.async`), and give each warp several 16 × 16 tiles to reuse fragments, exactly the reuse
  ideas of 04 §15–22, one level up.
- **Decode (M = 1) does not fit.** A 1-row GEMM would be padded to 16 rows, wasting 15/16 of the
  Tensor Core work. Decode GEMMs are memory-bound anyway (03 §5), so inference engines use
  specialized matrix-vector (GEMV) kernels there.

---

## 8. Experiments in `bench_precision`, and how to read them

- **A (speed):** compare FP32 v4 against WMMA. This is the Tensor Core effect. The FP16 tiled
  kernels show what you get *without* Tensor Cores (expect little gain, §6). FP16 vs FP32
  accumulation in the tiled kernel: little speed difference, a large accuracy difference.
- **B (accuracy vs K):** check your numbers against §5's emulated table. "total" vs "arith" for
  FP16-input variants shows how much error comes from rounding the inputs and how much from the
  kernel's arithmetic.
- **C (overflow):** the FP16-accumulator kernel should report inf outputs; FP32 accumulation
  should not.
- **D (Transformer prefill shapes):** realistic GEMM sizes. Compare with PyTorch's
  `cuBLAS fp16 (fp32 acc)` rows from `python/benchmark.py` to see how far this simple WMMA kernel
  is from production.

---

## 9. Quantization (concept)

**Quantization** stores numbers as small **integers** plus a **scale**, instead of floats.

### Symmetric INT8, by hand

INT8 stores whole numbers from −127 to 127 in 1 byte. To store weights w = [0.12, −0.5, 0.31, 0.02]:

```
scale = max|w| / 127 = 0.5 / 127 = 0.003937
q = round(w / scale)      = round([30.48, −127, 78.74, 5.08]) = [30, −127, 79, 5]     ← stored, 1 byte each
dequantize: q × scale     = [0.11811, −0.5, 0.31102, 0.019685]
error                     = [0.0019, 0, 0.0010, 0.0003]
```

- 1 byte per weight instead of 4 (FP32) or 2 (FP16), plus one scale per group of weights.
- **Per-tensor** scale: one scale for the whole matrix. A single large **outlier** value makes
  the scale big, and then every small weight loses precision.
- **Per-channel / per-group** scale: one scale per output row, or per 64–128 weights. More
  accurate, slightly more storage.
- **Weight-only quantization** (e.g. "W8A16", "W4A16"): weights are stored in INT8 or INT4 and
  converted back to FP16 inside the GEMM kernel, just before multiplying, while activations stay
  FP16. Because **decode is memory-bound** (03 §5), reading 4× fewer bytes per weight can make
  token generation substantially faster. This is why INT4 LLMs are popular on consumer GPUs.
- **Weight + activation quantization** (W8A8, FP8 on Hopper/Ada GPUs): the math itself also runs
  in 8 bits, on INT8/FP8 Tensor Cores, which helps compute-bound prefill.
- **Cost:** accuracy. Each method trades bits for error. It needs calibration data and outlier
  handling (methods such as GPTQ, AWQ, SmoothQuant).

This project implements FP16 and explains quantization conceptually, as planned. Adding an INT8
weight-only GEMM would be a natural extension: same tiled structure, but loading int8 weights
and multiplying by the scale.

---

## 10. Precision of the other operations in inference

- **Softmax and LayerNorm** in FP16 models: inputs and outputs are FP16 (to save bandwidth), but
  **the reductions (max, sum, mean, variance) are computed in FP32** inside the kernel (load
  half → convert to float → reduce → convert the result to half). The same reasoning as §5
  applies: sums and variances are where precision and range get lost.
- **Attention scores** can be large, so the max-subtraction of softmax (05 §2) is even more
  important in FP16 (the maximum is 65,504, and exp overflows FP16 already at about 11).
- **KV cache** (07): stored in FP16 or quantized to INT8/FP8 to save memory and bandwidth.

---

## 11. Common mistakes

1. Accumulating long sums in FP16 → error grows with K, and overflow above 65,504.
2. Leaving TF32 enabled and calling it "FP32" (09 §6).
3. Measuring FP16 error against the wrong reference. Separate input-rounding error from
   arithmetic error ("total" vs "arith").
4. Expecting FP16 to be 2× faster on CUDA cores. The big compute gain needs Tensor Cores.
5. Calling WMMA functions from only some lanes of a warp.
6. WMMA with sizes that aren't multiples of the tile, or misaligned pointers → wrong results
   or faults.
7. Assuming all GPUs have Tensor Cores. Check the compute capability at run time.
8. Quantizing with a single per-tensor scale when there are outliers.

## 12. Interview explanation (~90 seconds)

> "FP16 has 10 mantissa bits, so about 3 significant digits, and a maximum of 65,504, compared
> with FP32's 7 digits and a range up to about 10³⁸. For inference, FP16 halves memory and
> bandwidth, which directly speeds up memory-bound work like decode, and it unlocks Tensor
> Cores. But accumulation must stay in FP32. I emulated and then measured it: with an FP16
> accumulator, GEMM error grows to about 3% of the output scale at K = 16K, roughly 10,000×
> worse than FP32 accumulation, and a sum of products in [0, 8] over K = 8192 overflows to
> infinity. So I kept the inputs in FP16 and accumulated in FP32. That's what cuBLAS and Tensor
> Cores do; the product of two FP16 values is even exact in FP32. My v5 GEMM uses WMMA: each warp
> cooperatively loads 16×16 FP16 fragments and issues `mma_sync` with an FP32 accumulator
> fragment. It's a warp-level operation, since the fragment is spread across the 32 lanes'
> registers in a hardware-defined layout. It's a simple version without shared-memory staging,
> so I know where the remaining gap to cuBLAS comes from. Quantization takes this further:
> INT8 or INT4 weights with per-channel scales, dequantized inside the GEMM, which mainly speeds
> up memory-bound decode."

## 13. What to remember

- Exponent bits = range, mantissa bits = precision. FP16: max 65,504, ~3 digits. BF16: FP32's
  range, ~2 digits.
- Spacing grows with magnitude, which causes swamping (1000 + 0.25 = 1000 in FP16).
- Low precision helps: memory capacity, bandwidth (2×), Tensor Cores, cache.
- Store and multiply in FP16; **accumulate in FP32**.
- Tensor Cores: warp-level 16 × 16 × 16 matrix operations through fragments (`load_matrix_sync`,
  `mma_sync`, `store_matrix_sync`).
- Quantization: integers + scale; weight-only quantization speeds up memory-bound decode.

---

## 14. File guide

### `kernels/precision.cuh`
Declarations: `float_to_half`, `half_to_float`, `matmul_tiled_fp16_acc32`, `matmul_tiled_fp16_acc16`,
`matmul_wmma`, `wmma_supported`. Includes `<cuda_fp16.h>` for `__half`.

### `kernels/matmul_fp16.cu`
1. **What:** conversion kernels plus the tiled FP16-input GEMM with a selectable accumulator.
2. **Why:** isolate the effect of accumulator precision. 3. **Inputs:** FP16 A, B (device).
4. **Output:** FP32 C. 5. **Data flow:** global (half) → shared (half) → convert → register
accumulator (float or half) → global (float). 6. **Functions:** `float_to_half_kernel`,
`half_to_float_kernel`, `multiply_add` (2 overloads), `to_float` (2 overloads),
`matmul_tiled_fp16_kernel<TILE, AccT>`, `launch_tiled_fp16<AccT>`. 7. **Variables:** `As`,
`Bs`, `sum`, `zero`. 8. **CUDA concepts:** `__half`, `__float2half`, `__half2float`, `__hfma`,
type templates, overloading. 9. **Memory:** half the bytes of v3 in global and shared memory.
10. **Thread mapping:** identical to v3. 11. **Sync:** identical to v3. 12. **Performance:** CUDA
cores, so similar to FP32 v3. 13. **Mistakes:** §11.

### `kernels/matmul_wmma.cu`
1. **What:** GEMM v5 on Tensor Cores. 2. **Why:** the warp-level operation behind production
inference GEMMs. 3–4. FP16 A, B → FP32 C; M, N, K multiples of 16. 5. **Data flow:** global →
fragments (registers spread across the warp) → Tensor Core MMA → fragment → global.
6. **Functions:** `matmul_wmma_kernel`, `gpu::wmma_supported`, `gpu::matmul_wmma`.
7. **Variables:** `warp_id`, `tile_row`, `tile_col`, `a_frag`, `b_frag`, `c_frag`.
8. **CUDA concepts:** Tensor Cores, WMMA fragments, `_sync` warp-collective calls,
`__CUDA_ARCH__`, leading dimension. 9. **Memory:** loads directly from global with no
shared-memory reuse between warps (§7 limitations). 10. **Thread mapping:** one warp ↔ one
16 × 16 tile of C. 11. **Sync:** warp-collective calls; no block barriers. 12. **Performance:**
Tensor Core throughput, limited by redundant global loads. 13. **Mistakes:** §11 (5–7).

### `src/utils/precision_utils.cuh`
`round_to_half_on_gpu` (exact FP16 rounding of host data, done by the GPU) and `measure_error`
(max error as a fraction of max |reference|, plus a count of inf/NaN outputs).

### `src/cpu/cpu_ops.cpp` — `cpu::matmul_f64`
GEMM with double accumulation and output: the "exact" reference for error measurements.

### `tests/test_precision.cu`
Conversion table (§2), the WMMA padded hand example (exact), and 9 shapes × 3 kernels against the
exact product of the FP16-rounded inputs, with tolerances from §5 (FP32 accumulation: 2e-5 tiled,
1e-4 WMMA; FP16 accumulation: 2e-2). WMMA cases are skipped automatically on GPUs without Tensor
Cores, and use only multiples of 16.
**Verified before any GPU run:** the expected conversion values and the error table in §5 come
from a CPU emulation of IEEE FP16 rounding and of `__hfma` accumulation.

### `src/benchmark/bench_precision.cu`
Experiments A–D (§8).
