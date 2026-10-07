# 05 — Softmax

Files covered: `kernels/softmax.cu`, `kernels/softmax.cuh`, `kernels/warp_reduce.cuh`,
`src/cpu/cpu_ops.cpp` (`cpu::softmax`), `tests/test_softmax.cu`,
`src/benchmark/bench_softmax.cu`.

New concepts: numerical stability, reductions, shared-memory tree reduction, warp shuffles
(`__shfl_down_sync`, `__shfl_sync`), online softmax.

---

## 1. What problem are we solving?

**Softmax** turns a row of arbitrary real numbers ("scores", also called "logits") into
**probabilities**: every output is between 0 and 1, and the outputs of a row add up to 1.
Bigger scores get bigger probabilities.

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

`exp(x)` = e^x, with e ≈ 2.71828. It is always positive, and grows very fast.

**Hand example** (this is the first test in `tests/test_softmax.cu`):

```
x            = [ 1,       2,       3      ]
exp(x)       = [ 2.71828, 7.38906, 20.08554 ]     sum = 30.19288
softmax(x)   = [ 0.09003, 0.24473, 0.66524 ]       sum = 1
```

### Where softmax appears in Transformer inference

1. **Attention**: every row of the score matrix `QKᵀ/√d` goes through softmax. Row i then
   says how much token i "attends to" each other token. Shape: (heads × seq) rows of length seq.
   This is our main benchmark shape.
2. **Next-token probabilities**: the model's final output is one score per vocabulary word
   (50,257 for GPT-2). Softmax turns them into probabilities for sampling. In decoding, that is
   1 row × 50,257 columns per sequence.

We apply softmax to each **row** of a row-major matrix independently: `rows × cols`, with
softmax along `cols`.

---

## 2. Numerical stability: why we subtract the maximum

### What can go wrong

`float` (FP32) can represent numbers up to about 3.4 × 10³⁸. Since e^88.7 ≈ 3.4 × 10³⁸:
- `exp(x)` for **x > 88.7** gives **infinity** (overflow).
- `exp(x)` for **x < −103** gives **0** (underflow, below the smallest float).

Attention scores of 100 or more are not unusual in real models. With x = [1000, 1001, 1002]:

```
exp(x) = [inf, inf, inf]      sum = inf
inf / inf = NaN               → the whole row becomes NaN ("not a number")
```

The test program prints this on purpose (`unstable_softmax_row`), so you can see it happen.

With x = [−1000, −1001], all exponentials underflow to 0, and the result is 0/0 = NaN.

### The fix: shift by the row maximum

For any constant c:

```
exp(x_i − c) / Σ_j exp(x_j − c)  =  (exp(x_i)·e^−c) / (Σ_j exp(x_j)·e^−c)  =  exp(x_i) / Σ_j exp(x_j)
```

The `e^−c` appears in both the top and the bottom and cancels. So softmax **does not change**
when you shift a row by a constant. Choose `c = max(x)`:

- The largest shifted value is exactly 0 → `exp(0) = 1`. **No overflow is possible.**
- The sum is at least 1 (the max element contributes 1) → **no division by zero**.
- Tiny values may still underflow to 0, but that only happens for probabilities below 10⁻⁴⁵,
  which are 0 for every practical purpose.

```
x = [1000, 1001, 1002],  max = 1002
x − max = [−2, −1, 0]
exp     = [0.13534, 0.36788, 1]      sum = 1.50321
softmax = [0.09003, 0.24473, 0.66524]   ← same as softmax([1, 2, 3]) ✔
```

**Cost:** one extra pass over the row to find the max. That is the price of correctness. Every
production softmax pays it, except in the online version (§10), which hides it.

---

## 3. CPU version (`cpu::softmax`)

Three passes per row: **max → sum of exp(x − max) → divide**. It computes in `double` (about 15
significant digits instead of 7), so its result can serve as "ground truth" for testing the
float GPU kernels.

---

## 4. The GPU idea, and the new problem: reductions

Rows are independent: different rows can run on different threads, warps or blocks.

Inside one row, though, the work is **not** independent. The max and the sum each combine *all*
elements of the row into a single number. That is called a **reduction**. Reductions are the
key new skill in this phase. LayerNorm (Phase 6) needs them too.

### Sequential vs tree reduction

Sequential (one thread): `((((a+b)+c)+d)+…)` → n − 1 steps, one after another.

Tree (many threads): add pairs at the same time, then pairs of pairs, and so on:

```
step 0:  3   1   4   1   5   9   2   6        (8 values)
          \ /     \ /     \ /     \ /
step 1:   4       5       14      8           4 additions at the same time
            \     /         \     /
step 2:       9               22              2 additions
                \            /
step 3:              31                       1 addition  → log2(8) = 3 steps
```

n values take **log₂(n)** steps instead of n − 1. For 1024 values: 10 steps instead of 1023.
(The order of additions differs from sequential, so float results can differ in the last bits.
That's why tests use a tolerance.)

The three GPU designs differ in **who** does a row's reduction:

| Version | Threads per row | Reduction done with |
|---|---|---|
| v1 | 1 thread | a sequential loop |
| v2 | 1 block (256 threads) | each thread loops over part of the row, then a tree in **shared memory** |
| v3, v4 | 1 warp (32 threads) | each lane loops over part of the row, then a tree with **warp shuffles** |

---

## 5. v1: one thread per row (`softmax_naive_kernel`)

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;   // the 02 §4 formula: thread → row
if (row >= rows) return;
const float* x_row = x + static_cast<size_t>(row) * cols; // pointer to x[row][0]
```
- `x + row * cols`: **pointer arithmetic**. Adding an integer to a pointer moves it forward by
  that many *elements* (not bytes). Row `row` starts `row * cols` floats after the beginning
  (row-major, 03 §3).
- `static_cast<size_t>(row)`: multiply in 64 bits, because `rows * cols` can exceed 2³¹.
- `return` is safe here: this kernel has no `__syncthreads()`.

```cpp
float row_max = -INFINITY;
for (int c = 0; c < cols; ++c) row_max = fmaxf(row_max, x_row[c]);
```
- `-INFINITY`: starting value for a max. Any real number is bigger than it.
- `fmaxf(a, b)`: the larger of two floats. The `f` suffix means the float version of `fmax`.

```cpp
float row_sum = 0.0f;
for (int c = 0; c < cols; ++c) row_sum += expf(x_row[c] - row_max);
const float inv_sum = 1.0f / row_sum;
for (int c = 0; c < cols; ++c) y_row[c] = expf(x_row[c] - row_max) * inv_sum;
```
- `expf`: float exponential, accurate to about 1–2 units in the last place. (`__expf` is a
  faster hardware approximation; we stay with `expf` for accuracy.)
- `inv_sum = 1/sum` once, then **multiply**: a division is several times more expensive than a
  multiplication, and there are `cols` of them.
- `exp` is computed twice per element (pass 2 and pass 3). Recomputing it is cheaper than
  storing it, because storing would cost an extra write and read of global memory.

**Why v1 is slow:**
1. **Uncoalesced.** Neighboring threads handle neighboring *rows*. At the same `c`, a warp reads
   `x[row][c]`, `x[row+1][c]`, … which are `cols` floats apart (03 §2, pattern 3).
2. **Too little parallelism.** 1024 rows means only 1024 threads = 32 warps for the whole GPU.
   A T4 can run 1,280 warps at once (40 SMs × 32 warps).
3. **Long sequential loops.** Each thread does 3 × cols steps alone.

---

## 6. v2: one block per row, shared-memory tree (`softmax_block_kernel<BLOCK>`)

```cpp
__shared__ float scratch[BLOCK];
const int row = blockIdx.x;                 // grid = rows blocks: block b handles row b
const int tid = threadIdx.x;
```

### Step 1: each thread reduces part of the row (a block-stride loop)

```cpp
float local_max = -INFINITY;
for (int c = tid; c < cols; c += BLOCK) local_max = fmaxf(local_max, x_row[c]);
```
Thread `tid` looks at columns `tid, tid + 256, tid + 512, …`. In each step, the 256 threads read
256 **consecutive** columns, so the reads are **coalesced** ✔. Each thread ends with the max of
its own share.

### Step 2: combine the 256 partial results with a tree

```cpp
scratch[tid] = local_max;
__syncthreads();
for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    __syncthreads();
}
const float row_max = scratch[0];
__syncthreads();
```

The same tree as §4, with an example for BLOCK = 8:

```
scratch:   [3  1  4  1  5  9  2  6]
stride 4:  tid 0..3:  s[0]=max(3,5)  s[1]=max(1,9)  s[2]=max(4,2)  s[3]=max(1,6)
           [5  9  4  6  .  .  .  .]
stride 2:  tid 0..1:  s[0]=max(5,4)  s[1]=max(9,6)
           [5  9  .  .  .  .  .  .]
stride 1:  tid 0:     s[0]=max(5,9)
           [9  .  .  .  .  .  .  .]   → row_max = 9
```

The three kinds of `__syncthreads()`:
1. **After `scratch[tid] = local_max`**: all 256 values must be written before anyone reads
   them.
2. **Inside the loop, after each level**: level s + 1 reads results written in level s by
   *other* threads. The barrier is **outside** the `if`, so all threads reach it (04 §14
   explains why this matters).
3. **After reading `scratch[0]`**: the sum pass reuses `scratch`. Without this barrier, a fast
   thread could overwrite `scratch[0]` before a slow thread has read the max.

Then the same pattern computes the sum, and finally every thread normalizes its own columns.

**Cost of the tree:** in the last 5 levels (stride 16, 8, 4, 2, 1), fewer than 32 threads do
any work, but the whole block still waits at every barrier. A block of 256 threads needs
8 levels × 2 reductions = 16 barriers per row. For short rows, this overhead is a large part of
the total time.

---

## 7. Warp shuffles

A **shuffle** lets a thread read a **register** of another thread **in the same warp**,
directly, in one instruction. No shared memory and no `__syncthreads()` are needed, because
the data never leaves the registers.

### `__shfl_down_sync(mask, value, offset)`

"Give me `value` from the lane `offset` positions above me."
- **lane** = a thread's position inside its warp, 0…31 (`threadIdx.x % 32`).
- Lane `i` receives lane `i + offset`'s `value`.
- If `i + offset ≥ 32`, there is no such lane. Lane `i` receives **its own** value back.
- `mask = 0xffffffff`: 32 bits, one per lane, all set. It promises the hardware that **all 32
  lanes** execute this instruction together. Since the Volta generation, lanes of a warp can be
  scheduled independently. The `_sync` suffix plus the mask makes the warp line up at the shuffle.

### Warp sum (`warp_reduce_sum` in `kernels/warp_reduce.cuh`)

```cpp
for (int offset = 16; offset > 0; offset /= 2)
    v += __shfl_down_sync(FULL_WARP_MASK, v, offset);
return __shfl_sync(FULL_WARP_MASK, v, 0);
```

Here it is with a pretend 8-lane warp (offsets 4, 2, 1), using the values 3 1 4 1 5 9 2 6:

```
lane:            0    1    2    3    4    5    6    7
start v:         3    1    4    1    5    9    2    6
offset 4: +v[i+4] 8   10    6    7    x    x    x    x     (x = lanes whose result we never use)
offset 2: +v[i+2] 14  17    x    x    …
offset 1: +v[i+1] 31   x    …
```

Lane 0 ends with the total, 31. With 32 lanes, the offsets are 16, 8, 4, 2, 1 (5 steps =
log₂ 32). The "x" lanes compute garbage (some received their own value and added it to
itself), but lane 0's result never depends on them.

### `__shfl_sync(mask, value, src_lane)` — broadcast

"Give me `value` from lane `src_lane`." With `src_lane = 0`, every lane gets lane 0's total, so
every lane can use the result.

(There is also `__shfl_xor_sync`, which leaves the result in all lanes directly. We use
down + broadcast because it is easier to draw and to explain.)

### Why shuffles beat shared memory for small reductions

| | Shared-memory tree (v2) | Warp shuffle (v3) |
|---|---|---|
| Data path | register → shared memory → register | register → register |
| Barriers | `__syncthreads()` at every level | none |
| Instructions per level | store + barrier + load + op | 1 shuffle + op |
| Scope | the whole block | one warp (32 lanes) only |

---

## 8. v3: one warp per row (`softmax_warp_kernel`)

```cpp
const int warp_id = threadIdx.x / WARP_SIZE;              // 0..7 inside a 256-thread block
const int lane    = threadIdx.x % WARP_SIZE;              // 0..31
const int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
if (row >= rows) return;
```
- A block of 256 threads holds 8 warps, and each warp takes one row. So block b handles rows
  8b … 8b + 7. This is the 02 §4 index formula again, with *warps* as the unit instead of threads.
- The early `return` is safe: all 32 lanes of a warp have the same `row`, so the warp exits as a
  whole and the shuffles always see all 32 lanes. There is no `__syncthreads()` to deadlock on.

```cpp
float local_max = -INFINITY;
for (int c = lane; c < cols; c += WARP_SIZE) local_max = fmaxf(local_max, x_row[c]);
const float row_max = warp_reduce_max(local_max);
```
Lane `l` reads columns `l, l + 32, l + 64, …`. The 32 lanes read 32 consecutive floats per step,
so the reads are coalesced ✔. Then 5 shuffle steps produce the row max in every lane. The sum
and the normalize passes work the same way.

**v3 vs v2:**
- v3 has no shared memory and no barriers, and a tiny reduction (5 shuffle steps).
- v3 has more rows in flight: 8 rows per block instead of 1.
- But only 32 threads work on a row. For very long rows (65,536 columns), each lane loops 2,048
  times, and a row is processed by a single warp. For few, long rows, v2 (256 threads per row)
  should win. Experiment B measures where the crossover is.

---

## 9. How many bytes does each version move?

Minimum possible: read x once, write y once → **8 bytes per element**.

| Version | Reads of x | Writes of y | Bytes / element requested |
|---|---|---|---|
| v1, v2, v3 | 3 (max, sum, normalize) | 1 | 16 |
| v4 | 2 | 1 | 12 |

The repeated reads often hit the L1/L2 cache: a row of 1024 floats is only 4 KB. They are not
free, but cheaper than DRAM. Softmax does ~5 FLOPs per element against ≥ 8 bytes, so it is
deeply **memory-bound** (03 §4). The goal is GB/s close to the peak bandwidth, which is why the
benchmark reports **GB/s based on the minimal 8 bytes/element**. A version that reads x three
times shows up as lower "effective" GB/s.

---

## 10. v4: online softmax (one pass for max and sum)

### The idea

Can we compute the max and the sum **in the same pass**, before we know the final max?

Keep a running max `m` and a running sum `s = Σ exp(xⱼ − m)` over the values seen so far. When
a new value v arrives:
- **v ≤ m**: just add it: `s += exp(v − m)`.
- **v > m**: the max changes. Every term already in `s` was computed relative to the *old* m.
  Multiplying by `exp(m_old − v)` converts all of them to the new max at once, because
  `exp(xⱼ − m_old) · exp(m_old − v) = exp(xⱼ − v)`. Then add the new element, `exp(v − v) = 1`:

```cpp
if (v > m) { s = s * expf(m - v) + 1.0f;  m = v; }
else       { s += expf(v - m); }
```

### Worked example: one lane sees [1, 3, 2]

| step | v | case | s after | m after | check: s = Σ exp(xⱼ − m) |
|---|---|---|---|---|---|
| start | – | – | 0 | −∞ | empty |
| 1 | 1 | v > m | 0·e^(−∞) + 1 = **1** | 1 | e^(1−1) = 1 ✔ |
| 2 | 3 | v > m | 1·e^(1−3) + 1 = **1.13534** | 3 | e^(−2) + e^0 ✔ |
| 3 | 2 | v ≤ m | 1.13534 + e^(2−3) = **1.50321** | 3 | e^(−2) + e^0 + e^(−1) ✔ |

The output is `exp(x − 3) / 1.50321` = [0.0900, 0.6652, 0.2447]. That is softmax([1, 2, 3])
reordered, as expected.

### Merging two partial results (for the warp shuffle tree)

Two lanes have (m_a, s_a) and (m_b, s_b). Rescale both sums to the common max:

```
m = max(m_a, m_b)
s = s_a · exp(m_a − m) + s_b · exp(m_b − m)
```

Example: lane A saw [1, 3] → (3, 1.13534); lane B saw [2] → (2, 1).
m = 3; s = 1.13534 · e^0 + 1 · e^(2−3) = 1.13534 + 0.36788 = 1.50321 ✔ (the same as one lane
doing everything).

`online_merge` does exactly this, inside the same shuffle tree as §7, shuffling both `m` and `s`.

**Edge case:** if a row has fewer than 32 columns, some lanes see no elements: (m = −∞, s = 0).
Merging two empty lanes would compute `exp(−∞ − (−∞)) = exp(NaN)` → NaN. The guard
`if (m_new == -INFINITY) return;` skips that case. The tests cover rows of 1, 5 and 31 columns.

### Why this matters beyond softmax

This running-max-with-rescaling trick is the core of **FlashAttention**. FlashAttention
computes softmax over a row of attention scores that is processed in tiles and is never stored
in full, fixing up earlier partial results whenever a new maximum appears. We use it again in
Phase 8.

---

## 11. Performance: what to expect (verify with your run)

**Experiment A** (attention-like shapes): v1 should be far behind (uncoalesced, little
parallelism). v2 → v3 shows the cost of barriers and the shared-memory tree versus shuffles.
v3 → v4 shows the effect of one fewer pass over x (smaller if the rows fit in cache).

**Experiment B** (16M elements, rows getting longer):

| Shape | Threads working per row | Who should win, and why |
|---|---|---|
| 524,288 × 32 | v2: 256 threads per 32 columns, 224 idle | v3/v4: one warp = exactly one row |
| 16,384 × 1,024 | – | v3/v4 |
| 256 × 65,536 | v3/v4: 256 warps → low occupancy, long loops | v2: more threads per row |
| 1 × 50,257 (decode logits) | everyone: one block or one warp **for the whole GPU** | nobody is good |

The last row is a real problem in LLM decoding. One long row can only use one SM with
these designs. Production kernels split such a row across **many blocks** (each computes a
partial max and sum, and a second small kernel merges them with the §10 rule).

**Experiment C:** block size for v2. Small blocks → fewer threads per row and more rows per SM.
Large blocks → more barrier levels and many idle threads during the tree.

### Measured on a Tesla T4 (12_results §4): what held, and what didn't

- ✔ v1 is far behind everywhere (≈ 32 GB/s, 10% of peak). The 1 × 50,257 row is terrible for
  every design (best 7.2 GB/s). Warp-per-row wins for short rows (32 and 256 columns: about
  220 GB/s vs 15 and 107 for block-per-row).
- ✘ **The crossover comes earlier than predicted.** From 1,024 columns upward, block-per-row (v2)
  wins: 243.8 vs 138.2 GB/s at 16,384 × 1,024. That includes the attention-like shape
  12,288 × 1,024, where v2 was 12% faster than PyTorch's softmax. The cause is not yet
  confirmed; it's a Phase 10 profiling question.
- ✘ **Online softmax (v4) was not faster than v3.** It was equal or slower (e.g. 221 vs 285 GB/s
  at 1024 × 128). The pass it saves was mostly an L1/L2 hit anyway, and it does more work per
  element: a branch, plus an extra `expf` every time the running max changes. **Lesson: online
  softmax earns its place by enabling fusion (FlashAttention, 07 §6), not as a standalone
  speedup.** Fewer bytes only help when those bytes would have come from DRAM.
- Block size: 256 was best for v2 at 4096 × 1024 (238.6 GB/s); 1024 collapsed to 51.8 GB/s
  (one block per SM, 10 barrier levels per reduction).

---

## 12. Common mistakes

1. **Forgetting the max subtraction** → inf/NaN for large scores (the test demonstrates this).
2. **`__syncthreads()` inside `if (tid < stride)`** → deadlock. Put it outside the `if`.
3. **Missing the barrier after reading `scratch[0]`** before reusing `scratch`.
4. **Shuffling with a full mask when some lanes have exited** (for example
   `if (c >= cols) return;` per lane) → undefined behavior. Exit only whole warps.
5. **Online merge without the empty-lane guard** → NaN for rows shorter than 32.
6. **Dividing per element** instead of multiplying by `1/sum`.
7. **Using `exp`/`fmax` (double versions) in device code by accident** → slow double-precision
   math. Use `expf`/`fmaxf`.
8. **Testing only on nice inputs**: always include large values, equal values, and rows shorter
   than a warp.

## 13. Interview explanation (~90 seconds)

> "Softmax is a row-wise reduction followed by an elementwise op. I subtract the row max first,
> because exp overflows float above about 88. Shifting by a constant doesn't change softmax, and
> after the shift the largest term is exp(0) = 1. My naive kernel used one thread per row, which
> is uncoalesced and has too little parallelism. The block-per-row version strides threads across
> the row, so loads are coalesced, and reduces partial results with a shared-memory tree, which
> needs a `__syncthreads()` at every level. The warp-per-row version replaces that with
> `__shfl_down_sync`: five register-to-register steps, then `__shfl_sync` to broadcast lane 0's
> result, with no barriers. Finally I implemented online softmax: each lane keeps a running max and
> a sum that is rescaled by exp(old_max − new_max) when the max changes, so one pass gives both
> reductions. That cuts reads of the input from three to two, and it is the same trick
> FlashAttention uses. Softmax is memory-bound, so I compare versions by effective bandwidth
> against peak. Warp-per-row wins for typical attention rows; block-per-row wins for few very long
> rows like vocabulary logits."

## 14. What to remember

- Subtract the max: `softmax(x) = softmax(x − max)`, and it prevents overflow.
- Reductions use a tree: log₂(n) steps.
- Shared-memory tree: needs a barrier per level, outside any `if`.
- Warp shuffle: register-to-register, no barriers. Out-of-range lanes get their own value; lane 0
  ends up with the result, then broadcast it.
- Match the work split to the row length: warp per row for short rows, block per row for long
  ones, multiple blocks per row for huge single rows.
- Online softmax: rescale the running sum by exp(m_old − m_new). This is the FlashAttention trick.
- Softmax is memory-bound: judge it in GB/s.

---

## 15. File guide

### `kernels/warp_reduce.cuh`
`warp_reduce_sum`, `warp_reduce_max`, `FULL_WARP_MASK`. These are `__device__` functions:
callable from kernels, not from the CPU. `__forceinline__` asks the compiler to paste the body
into the caller (no function-call overhead). Requirement: all 32 lanes call them together.
Reused in Phase 6.

### `kernels/softmax.cu`
1. **What:** 4 softmax kernels plus launchers. 2. **Why:** compare reduction strategies.
3. **Inputs:** device `x` (rows × cols), sizes. 4. **Output:** device `y`, same shape.
5. **Data flow:** global → registers (partial max/sum) → shared memory or shuffles (combine) →
registers → global. 6. **Functions:** `softmax_naive_kernel`, `softmax_block_kernel<BLOCK>`,
`softmax_warp_kernel`, `online_merge`, `softmax_online_kernel`, `gpu::softmax_*`.
7. **Variables:** `row`, `tid`/`lane`/`warp_id`, `local_max`, `local_sum`, `row_max`, `inv_sum`,
`scratch`, `m`, `s`. 8. **CUDA concepts:** reductions, `__shared__`, `__syncthreads()`, shuffles,
templates. 9. **Memory:** v1 strided; v2–v4 coalesced; v4 one fewer read. 10. **Thread mapping:**
§4 table. 11. **Sync:** §6 (three barrier kinds); none in v3/v4. 12. **Performance:** §9, §11.
13. **Mistakes:** §12.

### `kernels/softmax.cuh`
Launchers and `softmax_versions()`, the same pattern as `gemm_versions()`.

### `src/cpu/cpu_ops.cpp` — `cpu::softmax`
Three-pass stable softmax in double precision. Ground truth.

### `tests/test_softmax.cu`
Hand example, shifted (stability) example, unstable-formula demonstration, all-equal rows,
14 shapes × 2 value ranges, and the row-sum = 1 check, for all versions plus 5 extra v2 block
sizes. Tolerance `rtol = 1e-5 + 1e-7·cols` (the sum's rounding error grows with cols),
`atol = 1e-7`.
**Verified before any GPU run:** a CPU emulation of v2 (shared-memory tree) and v4 (online
merge with exact `__shfl_down_sync` semantics, including out-of-range lanes) passed 72/72
cases, including 1 × 1, 1 × 5 and 2 × 50,257.

### `src/benchmark/bench_softmax.cu`
Experiments A (shared shapes, with CPU), B (row length sensitivity), C (v2 block size).
Every version is verified against `cpu::softmax` before it is timed.
