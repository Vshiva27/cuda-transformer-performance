# 04 — Matrix Multiplication (GEMM)

Part 1 (Phase 2): the problem, indexing, CPU version, **v1 naive** and **v2 coalesced**.
Part 2 (Phase 3): **v3 shared-memory tiling**, **v4 register tiling** (starts at §13).
v5, the warp-level Tensor Core version, is in [08_precision.md](08_precision.md) (Phase 7)
because it needs FP16 inputs; the reason is explained in §24.

Prerequisite: [03_memory_hierarchy.md](03_memory_hierarchy.md) (row-major layout, sectors,
coalescing, arithmetic intensity).

Files covered: `kernels/matmul.cuh`, `kernels/matmul_basic.cu`, `src/cpu/cpu_ops.cpp`
(`cpu::matmul`), `tests/test_matmul.cu`, `src/benchmark/bench_matmul.cu`.

---

## 1. What problem are we solving?

**GEMM** = GEneral Matrix Multiply: `C = A × B`.

```
A[M][K]  ×  B[K][N]  =  C[M][N]

     K               N                 N
  ┌─────┐         ┌─────┐           ┌─────┐
M │  A  │   ×   K │  B  │   =     M │  C  │
  └─────┘         └─────┘           └─────┘
```

- M = rows of A and rows of C
- K = columns of A = rows of B (the "inner" or "shared" dimension; it disappears in the result)
- N = columns of B and columns of C

**The one formula to remember:**

```
C[row][col] = Σ (k = 0 … K−1)  A[row][k] × B[k][col]
```

In words: element (row, col) of C is the **dot product** of **row `row` of A** with **column
`col` of B**.

```
            column col of B
                 ↓
              ┌─┬─┬─┐
              │ │█│ │
              │ │█│ │  B (K×N)
              │ │█│ │
              └─┴─┴─┘
 row of A → ███████    → C[row][col]
```

**Cost:** M·N outputs × K multiply-adds each = M·N·K multiply-adds = **2·M·N·K FLOPs**
(a multiply-add counts as 2 operations). For n = 1024: 2.1 billion FLOPs.

### Why GEMM matters for Transformers

Almost all the FLOPs of a Transformer are GEMMs. For a layer with hidden size d, processing T
tokens:

| Transformer step | GEMM shape (M × K) × (K × N) |
|---|---|
| Q, K, V projections | (T × d) × (d × 3d) |
| Attention scores QKᵀ | (T × d_head) × (d_head × T), per head |
| Scores × V | (T × T) × (T × d_head), per head |
| Output projection | (T × d) × (d × d) |
| MLP up / down | (T × d) × (d × 4d), then (T × 4d) × (4d × d) |

`bench_matmul` Experiment B runs these shapes for GPT-2 small (d = 768).

---

## 2. A concrete 4×4 example (computed by hand)

This exact example is also the first test in `tests/test_matmul.cu`.

```
A (4×4)              B (4×4)
 1  2  3  4           1  0  2  0
 5  6  7  8           0  1  0  0
 9 10 11 12           1  0  0  1
13 14 15 16           0  1  0  1
```

**C[0][0]** = row 0 of A · column 0 of B
= A[0][0]·B[0][0] + A[0][1]·B[1][0] + A[0][2]·B[2][0] + A[0][3]·B[3][0]
= 1·1 + 2·0 + 3·1 + 4·0 = **4**

**C[2][1]** = row 2 of A · column 1 of B
= 9·0 + 10·1 + 11·0 + 12·1 = **22**

**C[1][2]** = row 1 of A · column 2 of B
= 5·2 + 6·0 + 7·0 + 8·0 = **10**

Doing all 16:

```
C (4×4)
 4  6  2  7
12 14 10 15
20 22 18 23
28 30 26 31
```

### The same example in flat (row-major) memory

```
A (K = 4 columns):  A[row][k]  → A[row*4 + k]
index: 0 1 2 3 | 4 5 6 7 | 8  9 10 11 | 12 13 14 15
value: 1 2 3 4 | 5 6 7 8 | 9 10 11 12 | 13 14 15 16

B (N = 4 columns):  B[k][col]  → B[k*4 + col]
index: 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
value: 1 0 2 0 | 0 1 0 0 | 1 0  0  1 |  0  1  0  1
```

C[2][1] in flat indices (row = 2, col = 1):

| k | A index `2*4+k` | A value | B index `k*4+1` | B value | product | running sum |
|---|---|---|---|---|---|---|
| 0 | 8 | 9 | 1 | 0 | 0 | 0 |
| 1 | 9 | 10 | 5 | 1 | 10 | 10 |
| 2 | 10 | 11 | 9 | 0 | 0 | 10 |
| 3 | 11 | 12 | 13 | 1 | 12 | **22** |

Result goes to C index `2*4 + 1 = 9`. ✔

Notice the pattern as k increases:
- **A index increases by 1** (8, 9, 10, 11): walking **along a row** of A → consecutive memory.
- **B index increases by N** (1, 5, 9, 13): walking **down a column** of B → strided memory.

---

## 3. CPU version (`cpu::matmul`)

```cpp
for (int row = 0; row < M; ++row) {
    float* c_row = C + row * N;               // pointer to the start of row `row` of C
    for (int col = 0; col < N; ++col) c_row[col] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a = A[row * K + k];       // one element of A, kept in a register
        const float* b_row = B + k * N;       // pointer to row k of B
        for (int col = 0; col < N; ++col) {
            c_row[col] += a * b_row[col];     // add a × (row k of B) into row `row` of C
        }
    }
}
```

The textbook order is `row → col → k` (compute one dot product at a time). We use
`row → k → col` instead. It produces exactly the same sums in the same k order, but the
innermost loop walks along **rows** of B and C, which are consecutive in memory. That is the CPU
cache's version of "coalescing": a 64-byte CPU cache line holds 16 floats, and a consecutive walk
uses all of them. The textbook order walks down a column of B, using only 1 float of each
cache line before jumping N floats ahead.

(`static_cast<std::size_t>(row) * K` in the real code converts to a 64-bit type before
multiplying, so large matrices cannot overflow a 32-bit `int`.)

---

## 4. The GPU idea: one thread per output element

Every C[row][col] is independent of every other. So launch **one thread per element of C**:
M·N threads, each computing one dot product of length K.

C is two-dimensional, so we use a **2D grid of 2D blocks**. Then each thread has an
x-coordinate and a y-coordinate, and one of them selects the row while the other selects the
column.

### `dim3`

```cpp
dim3 block(32, 8);   // 32 threads in x, 8 in y  → 256 threads per block (z defaults to 1)
dim3 grid(gx, gy);   // gx blocks in x, gy blocks in y
kernel<<<grid, block>>>(...);
```

`dim3` is a small CUDA struct with fields `.x`, `.y`, `.z`. Inside the kernel, the built-ins
`threadIdx`, `blockIdx`, `blockDim`, `gridDim` all have `.x/.y/.z`.
Limits: `block.x * block.y * block.z ≤ 1024`; `grid.y` and `grid.z` ≤ 65535; `grid.x` ≤ 2³¹−1.

### How a 2D block is cut into warps

The hardware numbers the threads of a block with **x changing fastest**:

```
linear thread id = threadIdx.y * blockDim.x + threadIdx.x
warp number      = linear thread id / 32
```

With `block(32, 8)`, every warp is one full row of the block: same `threadIdx.y`,
`threadIdx.x` = 0…31. **So inside a warp, only `threadIdx.x` changes.** Whatever `threadIdx.x`
is mapped to (rows or columns of C) is the direction the warp spreads in, and that decides the
memory pattern. Remember this; it is the whole difference between v1 and v2.

---

## 5. Version 1: naive (`matmul_naive_kernel`)

```cpp
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Line by line

**`int row = blockIdx.x * blockDim.x + threadIdx.x;`**
The same formula as vector add (02, §4), applied to the x-dimension. "Threads in all earlier
blocks along x + my x position in my block." Here the result is used as the **row** of C. It is
a natural first guess ("x first, x = row"), and it is the reason this version is slow.

**`int col = blockIdx.y * blockDim.y + threadIdx.y;`**
The same formula on the y-dimension gives the **column**.

**`if (row < M && col < N)`**
A 2D bounds check. The grid is rounded up in *both* directions, so the blocks on the right edge
and the bottom edge can have threads outside the matrix. `&&` means both conditions must be
true. Note there is no check on K: `k` comes from our own loop, which stops at K.

**`float sum = 0.0f;`**
An accumulator in a **register**. The `f` suffix makes `0.0f` a `float` literal. Without it,
`0.0` is a `double`, which can cause slow double-precision math on the GPU.

**`for (int k = 0; k < K; ++k)`**
Walk along the shared dimension, exactly like the formula's Σ.

**`sum += A[row * K + k] * B[k * N + col];`**
- `A[row * K + k]` = A[row][k] (A has K columns).
- `B[k * N + col]` = B[k][col] (B has N columns).
- Multiply and add into the register. The compiler turns `sum += a * b` into a single
  **FMA** (fused multiply-add) instruction, which computes `a*b + sum` with one rounding.

**`C[row * N + col] = sum;`**
C[row][col] (C has N columns). It is written once, at the end. Accumulating directly into
global memory inside the loop would cost K global reads and writes instead of one write.

### Thread mapping on the 4×4 example (block 2×2, grid 2×2)

Grid: x covers rows, `(4+1)/2 = 2` blocks; y covers columns, 2 blocks.

```
                    col 0      col 1   │  col 2      col 3
                ┌──────────────────────┼──────────────────────┐
         row 0  │ blk(0,0)   blk(0,0)  │ blk(0,1)   blk(0,1)  │
                │ t(0,0)     t(0,1)    │ t(0,0)     t(0,1)    │
         row 1  │ blk(0,0)   blk(0,0)  │ blk(0,1)   blk(0,1)  │
                │ t(1,0)     t(1,1)    │ t(1,0)     t(1,1)    │
                ├──────────────────────┼──────────────────────┤
         row 2  │ blk(1,0)   blk(1,0)  │ blk(1,1)   blk(1,1)  │
                │ t(0,0)     t(0,1)    │ t(0,0)     t(0,1)    │
         row 3  │ blk(1,0)   blk(1,0)  │ blk(1,1)   blk(1,1)  │
                │ t(1,0)     t(1,1)    │ t(1,0)     t(1,1)    │
                └──────────────────────┴──────────────────────┘
      blk(bx,by) = blockIdx (x,y);  t(tx,ty) = threadIdx (x,y)
```

Example: block (1,0), thread (0,1) → row = 1·2 + 0 = **2**, col = 0·2 + 1 = **1** → computes
**C[2][1] = 22**.

Inside block (0,0), in hardware order (x fastest): thread (0,0) → C[0][0], thread (1,0) →
C[1][0], thread (0,1) → C[0][1], thread (1,1) → C[1][1]. **Consecutive threads go DOWN a
column of C.**

### Memory pattern of one warp (block 32×8, M = N = K = 1024)

Warp 0 of a block = threads with `threadIdx.y = 0`, `threadIdx.x = 0…31`. They all have the
**same col** and **32 consecutive rows** r … r+31. At one step k of the loop:

```
A[row*K + k]  rows r..r+31, same k      → addresses 1024 floats apart → 32 sectors  ✘ strided
B[k*N + col]  same k, same col          → 1 address                   → 1 sector    ✔ broadcast
C[row*N + col] (final write)            → addresses 1024 floats apart → 32 sectors  ✘ strided
```

```
A in memory (each row is 1024 floats long):
row r    : [ . . . k . . . ]   ← thread 0 reads here
row r+1  : [ . . . k . . . ]   ← thread 1 reads here (4 KB further)
row r+2  : [ . . . k . . . ]   ← thread 2
 …
→ the warp touches 32 different sectors and uses 4 bytes of each 32-byte sector
```

**Why it is not 8× slower in practice:** the sector fetched for `A[row][k]` also contains
`A[row][k+1 … k+7]`. If it is still in the L1 cache when the next 7 iterations need it, those
iterations hit the cache. So caches hide part of the damage. They cannot hide all of it, and
the strided C write is never helped.

---

## 6. Version 2: coalesced (`matmul_coalesced_kernel`)

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

**This is the only change.** x now selects the **column**, y selects the row. The loop body is
identical. The launch function also swaps the grid: `grid.x` covers N (columns), `grid.y`
covers M (rows).

### Thread mapping on the 4×4 example (block 2×2)

Inside block (0,0), in hardware order: thread (0,0) → C[0][0], thread (1,0) → **C[0][1]**,
thread (0,1) → C[1][0], thread (1,1) → C[1][1]. **Consecutive threads go ALONG a row of C.**

Same example as before: to compute C[2][1] you now need col = 1 → blockIdx.x = 0,
threadIdx.x = 1; row = 2 → blockIdx.y = 1, threadIdx.y = 0. The loop does exactly the four
steps in the §2 table and produces 22.

### Memory pattern of one warp (block 32×8, n = 1024)

Warp 0 = `threadIdx.y = 0`, `threadIdx.x = 0…31` → **same row**, **32 consecutive columns**.

```
A[row*K + k]   same row, same k        → 1 address                → 1 sector   ✔ broadcast
B[k*N + col]   same k, cols c..c+31    → 32 consecutive floats    → 4 sectors  ✔ coalesced
C[row*N + col] cols c..c+31            → 32 consecutive floats    → 4 sectors  ✔ coalesced
```

| per warp, per k step | v1 naive | v2 coalesced |
|---|---|---|
| A load | 32 sectors (strided) | 1 sector (broadcast) |
| B load | 1 sector (broadcast) | 4 sectors (coalesced) |
| **total** | **33 sectors** | **5 sectors** |
| final C write | 32 sectors | 4 sectors |

v2 asks the memory system for about 6–7× fewer sectors. The *measured* speedup depends on how
much the caches were already rescuing v1. Read it from Experiment A of `bench_matmul`.

### Why v2 is still far from peak (its limitation)

Count the global loads of one thread: 2 per k step (one A, one B) → 2K loads (8K bytes) for
2K FLOPs → **0.25 FLOP per byte requested**. Each element of B is requested again by every row
of C (M times), and each element of A is requested by every warp that covers that row (N/32
times). Most of these requests hit L1/L2 instead of DRAM, but even the caches cannot deliver
data fast enough to keep the math units busy.

The ideal intensity of a 1024³ GEMM is ~170 FLOP/byte (03, §4). v2 achieves only a small
fraction of the GPU's peak FLOP/s (compare the "% peak" column). The cure is **explicit
reuse**: load a tile of A and a tile of B into **shared memory** once, and let every thread in
the block reuse them. That is v3 (Phase 3).

---

## 7. Experiments in `bench_matmul`, and how to read them

**Experiment A (square sizes 32 … 2048):**
- Small n (32–128): far too few threads to fill the GPU (n = 32 → 1024 threads total, while a
  T4 can hold 40 SMs × 1024 = 40,960 threads at once). Launch overhead dominates, and the CPU
  may even win.
- Large n: compare the `coalesced vs naive` column. That is the effect of the access pattern
  alone, because the arithmetic is identical.
- `% peak` shows how far v2 is from the hardware limit. This is the motivation for Phase 3.

**Experiment B (Transformer shapes):** compare the *prefill* rows (M = 512) with the *decode*
rows (M = 1). For M = 1 there are only N threads, so occupancy is low. Also, each weight is
used exactly once, so GFLOP/s will be tiny. The `GB/s` column (minimum bytes / time) is the
meaningful metric there, because decode is memory-bound (03, §5).

**Experiment C (block shapes):** for each shape, work out the warp's footprint: with
`block.x = 32` a warp covers 1 row × 32 columns; with `block.x = 8` it covers 4 rows × 8
columns. Then predict the sector count per load before looking at the time. Also note
`32×1` (32 threads per block): an SM can hold only a limited number of blocks (16 on T4,
32 on A100), so tiny blocks cannot fill the SM. That is low occupancy.

---

## 8. Correctness testing (`tests/test_matmul.cu`)

- **Hand example** (exact match, tolerance 0): small whole numbers and their sums are exactly
  representable in FP32, so there is no rounding.
- **Random shapes**, including:
  - M ≠ N ≠ K (e.g. 3×5×7, 255×257×129): a swapped M/N/K in any index formula produces wrong
    values or out-of-range accesses.
  - Sizes that are not multiples of 32 or 8: these exercise the bounds check on the edge blocks.
  - M = 1 (the decode shape), N = 1, K = 1, K = 0 (C must be all zeros), M = 0.
  - 5 block shapes, because the grid computation depends on the block shape.
- **Tolerance** `gemm_tolerance(K) = 1e-6·K + 1e-5`: GPU FMA rounds once where the CPU rounds
  twice, so tiny differences are legitimate, and they grow with the number of additions K.
  Verified locally: at K = 129 the CPU result differed from a double-precision reference by
  ~5e-6, while the tolerance is 1.4e-4. An indexing bug produces errors of order 0.1–10.
- **Poison** output (−999) so an element that is never written cannot pass.
- A `std::function` list of versions (`all_versions()`) means Phase 3 kernels are added to
  every test with one line each.

---

## 9. Common mistakes

1. Wrong row length in a flat index: `A[row * N + k]` instead of `A[row * K + k]`. It works on
   square matrices and fails on non-square ones. **Always test non-square.**
2. Swapping grid dimensions: computing `grid.x` from M when the kernel uses x for columns.
   Rows or columns at the edge are then skipped, or the grid is too big.
3. Bounds-checking only one dimension.
4. Mapping `threadIdx.x` to rows (v1) → strided loads and stores.
5. Accumulating into `C[...]` in global memory inside the k loop instead of a register.
6. Using `0.0` (double) instead of `0.0f`.
7. Forgetting that `grid.y` is limited to 65535 blocks.

---

## 10. Interview explanation (~60 seconds)

> "GEMM computes each output C[row][col] as the dot product of row `row` of A and column `col`
> of B, which costs 2MNK FLOPs. My first CUDA version assigns one thread per output with a 2D
> grid. In the naive version I mapped threadIdx.x to the row. Since a warp is 32 threads with
> consecutive threadIdx.x, the warp read A down a column and wrote C down a column, with
> addresses K floats apart, so about 32 sectors per load. Swapping the mapping so threadIdx.x
> selects the column makes A a broadcast, and makes B and C fully coalesced: about 5 sectors
> instead of 33 per warp per step. Same arithmetic, different memory pattern, measurably faster.
> But it still re-reads A and B from cache for every output, so its real arithmetic intensity is
> about 0.25 FLOP/byte, far below what GEMM can reach. That is why the next step is
> shared-memory tiling."

---

## 11. What to remember

- `C[row][col] = Σ_k A[row][k] · B[k][col]`; flat indices `A[row*K+k]`, `B[k*N+col]`,
  `C[row*N+col]`.
- Along k: A moves by 1 (consecutive), B moves by N (strided).
- In a 2D block, x changes fastest, so a warp spreads along whatever `threadIdx.x` maps to.
- v1 → v2 changes only the mapping. `threadIdx.x → col` makes B and C coalesced and A a
  broadcast.
- FLOPs = 2MNK; report GFLOP/s, and compare with peak.
- v2's limit is lack of data reuse. The fix is shared-memory tiling.

---

## 12. File guide

### `kernels/matmul_basic.cu`
1. **What:** GEMM v1 and v2 kernels plus their launchers. 2. **Why:** the baseline GEMMs, and a controlled experiment showing the effect of the memory access pattern alone. 3. **Inputs:** device pointers A (M×K), B (K×N), sizes, block shape. 4. **Output:** C (M×N) in device memory. 5. **Data flow:** global (via L1/L2) → registers (`sum`) → global. 6. **Functions:** `matmul_naive_kernel`, `matmul_coalesced_kernel`, `gpu::matmul_naive`, `gpu::matmul_coalesced`. 7. **Variables:** `row`, `col`, `sum`, `k`. 8. **CUDA concepts:** 2D grids/blocks, `dim3`, warp formation from 2D blocks. 9. **Memory:** v1 strided A and C, broadcast B; v2 broadcast A, coalesced B and C. 10. **Thread mapping:** one thread ↔ one C element (§5, §6). 11. **Sync:** none; threads share nothing. 12. **Performance:** limited by memory/cache traffic due to no explicit reuse. 13. **Mistakes:** §9.

### `kernels/matmul.cuh`
Declares the launchers for all GEMM versions, with default block shape 32×8. Phase 3 adds its versions here.

### `src/cpu/cpu_ops.cpp` — `cpu::matmul`
Reference GEMM, loop order row → k → col for cache-friendly access (§3). Ground truth for tests, and the CPU baseline.

### `tests/test_matmul.cu`
§8. Exits non-zero on any failure, so `ctest` reports it.

### `src/benchmark/bench_matmul.cu`
Experiments A–C (§7). Verifies each kernel against the CPU (or against the verified naive kernel for n = 2048) before timing. `GemmProblem` bundles host and device buffers for one problem size. The number of iterations adapts to the problem cost (`iters_for`).

### `kernels/matmul_tiled.cu`, `kernels/matmul_register.cu`
See the file guide at the end of Part 2 (§28).

---
---

# Part 2 — Reusing data: shared-memory tiling (v3) and register tiling (v4)

## 13. What problem are we solving now?

v2 is coalesced, but it has **no planned data reuse**. Look at one block of 32×8 threads
computing an 8×32 patch of C. All 32 threads in a row of the block need **the same row of A**.
All 8 threads in a column need **the same column of B**. Yet every thread fetches its own copy
from global memory (through L1/L2), K times each. The same numbers travel from memory to the
SM again and again.

If the threads of a block could **load each value once and share it**, global traffic would
drop a lot. That is exactly what **shared memory** is for.

---

## 14. Shared memory and `__syncthreads()`

### `__shared__`

```cpp
__shared__ float As[TILE][TILE];
```

- `__shared__` — a CUDA keyword: this array lives in the SM's on-chip shared memory.
- There is **one copy per block**. All threads of the block see the same array; other blocks
  have their own separate copy.
- It exists only while the block runs. Data must be loaded into it by the threads themselves.
- The size must be a compile-time constant. That is why `TILE` is a **template parameter**
  (`template <int TILE>`): the compiler creates a separate kernel for each tile size we use (8,
  16, 32). This is called a **template instantiation**.
- Size limit: 48 KB per block by default. v3 with TILE = 32 uses 2 × 32 × 32 × 4 B = 8 KB.

### `__syncthreads()`

A **barrier** for all threads of one block: no thread continues past it until **every** thread
of the block has reached it. It only synchronizes threads within the same block. Different
blocks cannot synchronize with each other this way.

Why we need it: in tiling, thread A *writes* a value into shared memory and thread B *reads* it.
Threads do not run in a fixed order (different warps run at different times). Without a
barrier, B might read the slot before A has written it. That is a **race condition**: the
result depends on timing, and it may be wrong only sometimes.

**Rule:** every thread of the block must reach the same `__syncthreads()`. Putting it inside an
`if` that only some threads enter can hang the block (deadlock) or give undefined results.
That is why our bounds checks are written as `cond ? load : 0.0f` (every thread runs this line)
and not as an `if` around the whole loop.

---

## 15. v3 idea, in a picture

A block computes a TILE × TILE square of C. It walks along K in steps of TILE:

```
                         B
                ┌────┬────┬────┐
                │    │ B0 │    │   t = 0: tile B0 (rows 0..T-1 of K)
                ├────┼────┼────┤
                │    │ B1 │    │   t = 1: tile B1
                ├────┼────┼────┤
                │    │ B2 │    │   t = 2: tile B2
                └────┴────┴────┘
       A
┌────┬────┬────┐  ┌────┬────┬────┐
│    │    │    │  │    │    │    │
├────┼────┼────┤  ├────┼────┼────┤
│ A0 │ A1 │ A2 │  │    │ C  │    │   C tile = A0·B0 + A1·B1 + A2·B2
├────┼────┼────┤  ├────┼────┼────┤
│    │    │    │  │    │    │    │
└────┴────┴────┘  └────┴────┴────┘
```

For every step t:
1. **Load**: the block's threads cooperatively copy tile A_t and tile B_t into shared memory.
   Each thread copies exactly one element of each.
2. `__syncthreads()`: wait until both tiles are complete.
3. **Compute**: every thread adds TILE products into its own `sum`, reading only shared memory.
4. `__syncthreads()`: wait until everyone has finished reading before the next load
   overwrites the tiles.

Each element loaded from global memory is now used by TILE threads, so **global memory traffic
drops by a factor of TILE**.

---

## 16. v3 line by line

```cpp
template <int TILE>
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
```
Two shared tiles: `As` holds a TILE × TILE piece of A (rows of this block, a window of K),
`Bs` holds a piece of B (a window of K, columns of this block).

```cpp
    const int tx = threadIdx.x;   // column inside the tile
    const int ty = threadIdx.y;   // row inside the tile
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
```
The same mapping as v2: x → column, y → row (coalesced). `blockDim` equals `TILE` here, so we
write `TILE` directly (a compile-time constant). `const` means the value never changes after
it is set.

```cpp
    float sum = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;
```
The register accumulator, and the number of K-steps, rounded up (ceiling division, as in 02 §5).

```cpp
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
```
At step t, the K-window is `t*TILE … t*TILE + TILE − 1`.
- This thread loads **A[row][a_col]**: its own row, and the column of the window given by its `tx`.
- It loads **B[b_row][col]**: the row of the window given by its `ty`, and its own column.

```cpp
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
```
`cond ? x : y` is the **ternary operator**: if `cond` is true it gives `x`, otherwise `y`. Outside
the matrix (the last partial tile, or edge blocks) we store **0**. A zero contributes
`0 × something = 0` to the sum, so partial tiles need no special code in the compute loop.

Global access pattern: in a warp (same `ty`, `tx` = 0…31), A is read at `row*K + t*TILE + 0…31`
(consecutive, coalesced ✔) and B at `b_row*N + col…col+31` (consecutive, coalesced ✔).

```cpp
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
```
This is the same dot-product formula as before, restricted to the current window. `As[ty][k]`
is "row `ty` of the A tile" and `Bs[k][tx]` is "column `tx` of the B tile". `#pragma unroll`
(in the real code) asks the compiler to copy the loop body TILE times. That removes the loop
counter and branch, and lets it schedule the loads early.

```cpp
    if (row < M && col < N) C[row * N + col] = sum;
```
Write the result once. The bounds check is here, *after* the loop, so that out-of-range
threads still took part in loading tiles and in every `__syncthreads()`.

---

## 17. Step-by-step trace: the 4×4 example with TILE = 2

Same A and B as §2. TILE = 2 → grid = 2×2 blocks of 2×2 threads; `num_tiles = 4/2 = 2`.

Follow **block (blockIdx.x = 0, blockIdx.y = 1)**. It computes rows 2–3, columns 0–1 of C:

```
thread (tx,ty) → row = 1*2 + ty, col = 0*2 + tx
(0,0) → C[2][0]    (1,0) → C[2][1]
(0,1) → C[3][0]    (1,1) → C[3][1]
```

### Step t = 0 (K-window k = 0, 1)

Who loads what (`a_col = 0 + tx`, `b_row = 0 + ty`):

| thread (tx,ty) | loads A[row][a_col] → As[ty][tx] | loads B[b_row][col] → Bs[ty][tx] |
|---|---|---|
| (0,0) | A[2][0] = 9 → As[0][0] | B[0][0] = 1 → Bs[0][0] |
| (1,0) | A[2][1] = 10 → As[0][1] | B[0][1] = 0 → Bs[0][1] |
| (0,1) | A[3][0] = 13 → As[1][0] | B[1][0] = 0 → Bs[1][0] |
| (1,1) | A[3][1] = 14 → As[1][1] | B[1][1] = 1 → Bs[1][1] |

```
After __syncthreads():   As = |  9 10 |     Bs = | 1 0 |
                              | 13 14 |          | 0 1 |
```

Compute (`sum += As[ty][0]*Bs[0][tx] + As[ty][1]*Bs[1][tx]`):

| thread | computes | partial sum |
|---|---|---|
| (0,0) C[2][0] | 9·1 + 10·0 | 9 |
| (1,0) C[2][1] | 9·0 + 10·1 | 10 |
| (0,1) C[3][0] | 13·1 + 14·0 | 13 |
| (1,1) C[3][1] | 13·0 + 14·1 | 14 |

### Step t = 1 (K-window k = 2, 3)

`a_col = 2 + tx`, `b_row = 2 + ty`:

| thread (tx,ty) | As[ty][tx] ← A[row][2+tx] | Bs[ty][tx] ← B[2+ty][col] |
|---|---|---|
| (0,0) | A[2][2] = 11 | B[2][0] = 1 |
| (1,0) | A[2][3] = 12 | B[2][1] = 0 |
| (0,1) | A[3][2] = 15 | B[3][0] = 0 |
| (1,1) | A[3][3] = 16 | B[3][1] = 1 |

```
As = | 11 12 |     Bs = | 1 0 |
     | 15 16 |          | 0 1 |
```

| thread | adds | final sum | expected (§2) |
|---|---|---|---|
| (0,0) C[2][0] | 11·1 + 12·0 = 11 | 9 + 11 = **20** | 20 ✔ |
| (1,0) C[2][1] | 11·0 + 12·1 = 12 | 10 + 12 = **22** | 22 ✔ |
| (0,1) C[3][0] | 15·1 + 16·0 = 15 | 13 + 15 = **28** | 28 ✔ |
| (1,1) C[3][1] | 15·0 + 16·1 = 16 | 14 + 16 = **30** | 30 ✔ |

**What to notice:**
- In each step, the 4 threads loaded 4 values of A and 4 of B (8 global loads total) and then
  did 4 × 2 = 8 multiply-adds from shared memory. In v2, the same 8 multiply-adds would have
  needed 16 global loads. With TILE = 32 the saving is 32×.
- The sum for C[2][1] is the *same four products in the same order* (k = 0, 1, 2, 3) as in §2.
  Tiling changes **where the data comes from**, not the math.

---

## 18. Shared memory banks (why access patterns matter even on-chip)

Shared memory is split into **32 banks**. Each bank serves one 4-byte word per clock cycle.
Word address `w` (a float index into the shared array) lives in bank `w % 32`.

When a warp accesses shared memory:
- 32 threads → 32 **different banks**: served in one cycle ✔
- several threads → the **same word**: one read, sent to all of them (broadcast) ✔
- several threads → **different words in the same bank**: served one after another. This is
  a **bank conflict**. An n-way conflict makes the access n times slower ✘

v3 with TILE = 32 (a warp = one `ty`, `tx` = 0…31):
- `As[ty][k]`: all 32 threads read the same word → broadcast ✔
- `Bs[k][tx]`: words `k*32 + 0…31` → banks 0…31 → conflict-free ✔
- The stores `As[ty][tx]`, `Bs[ty][tx]`: consecutive words → conflict-free ✔

---

## 19. Performance of v3, and its limit

**Global-memory intensity.** Per K-step, a block loads 2·TILE² floats (8·TILE² bytes) and
performs 2·TILE³ FLOPs → **TILE/4 FLOP/byte** (TILE = 32: 8 FLOP/byte, versus 0.25 for v2).

**New bottleneck: shared memory.** In the inner loop, every FMA needs **two shared-memory
loads** (`As[ty][k]`, `Bs[k][tx]`). The SM can issue FMAs faster than it can serve shared loads,
so the math units wait on shared memory. The `__syncthreads()` barriers also make warps
wait for the slowest warp of the block twice per step.

**Tile size trade-offs (Experiment D):**
- TILE = 8: 64 threads per block, reuse only 8×. The per-SM block limit (16 on T4) caps the SM
  at 16 × 64 = 1024 threads, and the global loads are short (8 floats = 32 bytes per row).
- TILE = 16: 256 threads, reuse 16×.
- TILE = 32: 1024 threads per block (the maximum), reuse 32×. On GPUs with 1024 threads per SM
  (T4), only **one block** fits per SM, so during each barrier the whole SM waits.

Which wins depends on the GPU. Measure, then explain the result using these points.

---

## 20. v4 idea: each thread computes a 4×4 group (register tiling)

To beat the shared-memory bottleneck, a thread must do **more FMAs per value it loads**.

Think of the inner loop at a fixed k. Suppose a thread owns a 4×4 group of outputs. It needs
4 values from a column of the A tile, and 4 values from a row of the B tile:

```
                b0    b1    b2    b3        ← 4 values of B (row k of the B tile)
           ┌────────────────────────┐
     a0    │ a0b0  a0b1  a0b2  a0b3 │
     a1    │ a1b0  a1b1  a1b2  a1b3 │   16 multiply-adds into 16 accumulators
     a2    │ a2b0  a2b1  a2b2  a2b3 │
     a3    │ a3b0  a3b1  a3b2  a3b3 │
           └────────────────────────┘
     ↑ 4 values of A (column k of the A tile)
```

This is an **outer product**: 4 + 4 = 8 loads from shared memory, 16 FMAs. That is
**0.5 loads per FMA** instead of v3's 2. Each loaded value is put in a **register** and reused
4 times.

### The tile hierarchy

```
block tile   64 × 64 outputs of C   (one block, 256 threads)
K step       BK = 8: As = 64 × 8 tile of A, Bs = 8 × 64 tile of B (shared memory)
thread tile  4 × 4 outputs per thread (16 accumulators in registers)

256 threads × 16 outputs = 4096 = 64 × 64 ✔
```

### Which 16 outputs does a thread own?

The 256 threads are arranged as a 16 × 16 grid: `thread_col = tid % 16`, `thread_row = tid / 16`.
Split the 64 × 64 block tile into a 4 × 4 arrangement of 16 × 16 sub-tiles. Thread
(`thread_row`, `thread_col`) owns **the same position inside each of the 16 sub-tiles**:

```
rows:    thread_row + 0, +16, +32, +48
columns: thread_col + 0, +16, +32, +48

64×64 block tile; X marks the 16 outputs of thread (thread_row = 1, thread_col = 2):

            cols 0..15    cols 16..31   cols 32..47   cols 48..63
           ┌─────────────┬─────────────┬─────────────┬─────────────┐
 row  1    │   X (col 2) │   X (18)    │   X (34)    │   X (50)    │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
 row 17    │   X         │   X         │   X         │   X         │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
 row 33    │   X         │   X         │   X         │   X         │
           ├─────────────┼─────────────┼─────────────┼─────────────┤
 row 49    │   X         │   X         │   X         │   X         │
           └─────────────┴─────────────┴─────────────┴─────────────┘
 → owns C[1, 17, 33, 49][2, 18, 34, 50]  (plus the block's top-left offset)
```

Example: `tid = 18` → `thread_col = 18 % 16 = 2`, `thread_row = 18 / 16 = 1` → exactly the
picture above.

**Why spread out, and not a 4 × 4 block of neighbors?** (§22 explains in detail)
1. Reading `Bs[k][thread_col + 16j]`: 16 consecutive threads read 16 consecutive words → no
   bank conflicts. With neighbors (`thread_col*4 + j`), threads 0 and 8 would hit the same
   bank (2-way conflict).
2. Writing C: consecutive threads write consecutive columns → coalesced stores.

---

## 21. v4 line by line (the parts that are new)

```cpp
constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
constexpr int THREADS_X = BN / TN;   // 16
constexpr int THREADS_Y = BM / TM;   // 16
constexpr int NUM_THREADS = THREADS_X * THREADS_Y;   // 256
```
`constexpr` means "computed at compile time". Shared array sizes and unrolled loop bounds must
be compile-time constants. They are in an **anonymous namespace** (`namespace { … }`), which
makes them private to this file.

```cpp
const int tid = threadIdx.x;               // 1D block of 256 threads
const int thread_col = tid % THREADS_X;    // remainder: 0..15
const int thread_row = tid / THREADS_X;    // integer division: 0..15
```
This time we launch a **1D** block and make the 2D layout ourselves with `%` and `/`. This is
the same flattening as row-major, in reverse: `tid = thread_row * 16 + thread_col`.

```cpp
float acc[TM][TN] = {};   // 16 accumulators, all zero
float a_reg[TM];
float b_reg[TN];
```
Small local arrays. They stay in **registers only if every index is a compile-time constant**
after unrolling. That is why every loop over `i`, `j`, `k` has `#pragma unroll` with constant
bounds. If the compiler cannot resolve an index at compile time, the array is placed in
**local memory** (which is really global memory) and the kernel becomes much slower. Nsight
Compute reports this as "local memory spilling".

### Cooperative tile loading

```cpp
for (int i = tid; i < BM * BK; i += NUM_THREADS) {     // 512 elements, 256 threads → 2 each
    const int r = i / BK;                               // row inside the A tile (0..63)
    const int c = i % BK;                               // column inside the A tile (0..7)
    ...
    As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
}
```
This is a **block-stride loop**: the grid-stride idea (02 §8) applied inside one block. The
A tile has 512 elements, but there are only 256 threads, so each thread loads element `tid` and
element `tid + 256`.
- `tid = 5`: i = 5 → (r 0, c 5); i = 261 → (r 32, c 5).
- Consecutive `i` → consecutive `c` → consecutive global addresses, so 8 threads read one 32-byte
  sector of a row of A. A warp reads 4 rows × 32 bytes = 4 fully used sectors ✔.
- The B tile works the same way with `r = i / BN`, `c = i % BN`: consecutive threads read
  consecutive columns of B ✔.

### The outer-product loop

```cpp
for (int k = 0; k < BK; ++k) {
    for (int i = 0; i < TM; ++i) a_reg[i] = As[thread_row + i * THREADS_Y][k];   // 4 loads
    for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][thread_col + j * THREADS_X];   // 4 loads
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] += a_reg[i] * b_reg[j];                                      // 16 FMAs
}
```
- `As[thread_row + 16i][k]`: the A values for this thread's 4 rows, at depth k.
- `Bs[k][thread_col + 16j]`: the B values for this thread's 4 columns, at depth k.
- `acc[i][j]` accumulates C[row_i][col_j]. Each `acc[i][j]` still receives its products in the
  order k = 0, 1, 2, …, K − 1, so the math is the same as every earlier version.

### Writing the results

```cpp
row = block_row0 + thread_row + i * THREADS_Y;
col = block_col0 + thread_col + j * THREADS_X;
if (row < M && col < N) C[row * N + col] = acc[i][j];
```
The same ownership formula as §20, plus the block's top-left corner. The bounds check handles
matrices that are not multiples of 64.

---

## 22. v4 memory analysis

**Global memory.** Per K-step a block loads (64·8 + 8·64) floats = 4 KB and does
2·64·64·8 = 65,536 FLOPs → **16 FLOP/byte** (v3 TILE = 32: 8; v2: 0.25).

**Shared memory reads (per warp: tid 0…31 → thread_col 0…15, thread_row 0 or 1):**
- `As[thread_row + 16i][k]`: 2 distinct words, 8 apart (the row length is BK = 8) → banks
  differ by 8 → no conflict; each word is broadcast to 16 threads ✔
- `Bs[k][thread_col + 16j]`: 16 consecutive words (both halves of the warp read the same ones)
  → 16 different banks + broadcast ✔
- If each thread instead owned 4 *adjacent* columns (`thread_col*4 + j`), threads 0 and 8
  would read words 32 apart (the same bank, different words) → **2-way bank conflict**.

**Loads per FMA**: v3 = 2 (shared); v4 = 8/16 = **0.5**.

**Cost: registers.** Each thread now holds 16 accumulators + 8 operands + indices, roughly 40–64
registers. An SM has 65,536 registers. 256 threads × 64 registers = 16K per block → up to 4
blocks per SM. More registers per thread → fewer warps fit → lower occupancy. **Register
tiling trades occupancy for reuse.** It works because each warp now has 16 independent FMAs
per step (**instruction-level parallelism**), so fewer warps are enough to keep the units busy.

**Cost: small matrices.** One block covers 64 × 64 outputs. For n = 128 that is only 4 blocks
for a GPU with 40+ SMs. Most SMs sit idle, so v4 can be *slower* than v3 at small sizes. Expect
to see this in Experiment A. It is a real trade-off, not a bug.

---

## 23. Limitations of v4 (what production libraries do further)

- **Vectorized loads** (`float4`: 16 bytes per instruction) to cut the number of load
  instructions.
- **Double buffering**: load the next tile while computing on the current one, to hide memory
  latency (with `cp.async` on Ampere and newer).
- **Warp tiling**: an extra level between the block tile and the thread tile.
- **Tensor Cores**: dedicated matrix units, many times the FP32 FMA throughput, used with FP16
  or BF16 inputs. → **v5, in Phase 7.**
- **Auto-tuning**: choosing BM/BN/BK/TM/TN per GPU and per matrix shape.

cuBLAS (behind `torch.matmul`) does all of this. In Phase 4 we measure how close v4 gets to it.

## 24. Why v5 (warp-level) is in Phase 7 and not here

Two candidates for a "warp-level" GEMM:
1. **FP32 warp tiling**: a third tiling level (block → warp → thread). It adds a lot of
   index arithmetic, and gains over v4 are often small on GPUs like the T4. It would make the
   project harder to explain without teaching a new idea.
2. **Tensor Core MMA via the WMMA API**: the 32 threads of a warp *cooperatively* compute a
   16 × 16 × 16 matrix product in one operation. This is genuinely warp-level, and it is what
   production inference GEMMs run on. It requires FP16 inputs with FP32 accumulation, which is
   exactly the topic of Phase 7.

We chose option 2. Warp-level programming with `__shfl_sync()` arrives in Phase 5 (Softmax).

---

## 25. Common mistakes (tiling)

1. **Missing the first `__syncthreads()`**: threads read tile slots that are not written yet.
   The result is wrong, sometimes, depending on timing.
2. **Missing the second `__syncthreads()`**: a fast thread overwrites the tile for step t + 1
   while slow threads are still reading step t.
3. **`__syncthreads()` inside a divergent `if`**: deadlock or undefined behavior. Put bounds
   checks on loads and stores, never around the barrier.
4. **Not zero-filling partial tiles**: garbage enters the sum when K, M or N is not a multiple
   of the tile. The tests use shapes like 65 × 63 × 9 and 255 × 257 × 129 to catch this.
5. **Returning early for out-of-range threads** (`if (row >= M) return;` at the top): those
   threads then skip loads and barriers that the others wait for.
6. **Register arrays indexed by run-time values**: they spill to slow local memory.
7. **Swapping `As[ty][k]` / `Bs[k][tx]` indices**: correct only for symmetric inputs, which is
   why the tests use random non-square matrices.

## 26. Interview explanation (~90 seconds)

> "After coalescing, my GEMM was limited by data movement, because every thread re-read A and B
> through the cache. In v3, each block stages a 32×32 tile of A and of B in shared memory. Every
> thread loads one element of each, there's a `__syncthreads()` barrier, then each thread does 32
> FMAs from shared memory, and there's a second barrier before the next tile overwrites it. That
> cuts global traffic by the tile size; arithmetic intensity goes from 0.25 to 8 FLOP per byte.
> The new bottleneck was shared memory itself: two shared loads per FMA. So in v4 each thread
> computes a 4×4 block of outputs. Per k step it loads 4 A values and 4 B values into registers
> and does 16 FMAs as an outer product, which is 0.5 loads per FMA. I assigned the 4×4 outputs
> strided by 16 rather than adjacent, so shared-memory reads are bank-conflict-free and the
> global stores are coalesced. The trade-off is register pressure, which lowers occupancy, and
> fewer, bigger blocks, which hurts small matrices. Every version is validated against a CPU
> reference on non-square shapes before timing."

## 27. What to remember

- Tiling = load once into shared memory, reuse TILE times. Global intensity ≈ TILE/4 FLOP/B.
- Two barriers per tile step: after loading (so data is ready) and after computing (so the
  tiles are free to overwrite).
- Zero-padding makes partial tiles work without special cases.
- Shared memory has 32 banks: same word → broadcast; different words in the same bank →
  conflict.
- Register tiling = an outer product per k: TM + TN loads give TM × TN FMAs.
- Register arrays need compile-time indices (`#pragma unroll`, `constexpr`).
- Every optimization has a cost: shared memory, registers, occupancy, parallelism at small sizes.

---

## 28. File guide (Part 2)

### `kernels/matmul_tiled.cu`
1. **What:** GEMM v3 kernel template plus its launcher. 2. **Why:** introduces explicit data reuse through shared memory. 3. **Inputs:** device A, B; M, N, K; tile size (8/16/32). 4. **Output:** C. 5. **Data flow:** global → shared (cooperative load) → registers (`sum`) → global. 6. **Functions:** `matmul_tiled_kernel<TILE>`, `gpu::launch_tiled<TILE>`, `gpu::matmul_tiled` (picks the template from the run-time `tile` with a `switch`). 7. **Variables:** `As`, `Bs`, `tx`, `ty`, `row`, `col`, `t`, `a_col`, `b_row`, `sum`. 8. **CUDA concepts:** `__shared__`, `__syncthreads()`, templates, `#pragma unroll`. 9. **Memory:** coalesced global loads; conflict-free shared access (§18). 10. **Thread mapping:** one thread per C element; one load per tile per thread. 11. **Sync:** two barriers per tile step (§14). 12. **Performance:** limited by shared-memory loads (2 per FMA) and barriers. 13. **Mistakes:** §25.

### `kernels/matmul_register.cu`
1. **What:** GEMM v4. 2. **Why:** cuts shared-memory traffic per FMA by 4× using register reuse. 3–4. As v3, but with no configuration parameter (fixed 64 × 64 × 8, 4 × 4 per thread). 5. **Data flow:** global → shared (block-stride cooperative loads) → registers (`a_reg`, `b_reg`) → 16 accumulators → global. 6. **Functions:** `matmul_register_kernel`, `gpu::matmul_register`. 7. **Variables:** `tid`, `thread_row`, `thread_col`, `block_row0`, `block_col0`, `acc`, `a_reg`, `b_reg`, `k0`. 8. **CUDA concepts:** 1D block reshaped into 2D, block-stride loops, register arrays, `constexpr`. 9. **Memory:** §22. 10. **Thread mapping:** strided 4 × 4 ownership (§20). 11. **Sync:** two barriers per K-step. 12. **Performance:** higher intensity, but needs more registers and gives fewer blocks. 13. **Mistakes:** §25.

### `kernels/matmul.cuh`
Launch declarations for all versions, plus `gemm_versions()`: a list of `{name, function pointer}`
used by tests and benchmarks. Each entry is a **captureless lambda**, a lambda that uses no
outside variables, which C++ can convert to a plain function pointer.

### `tests/test_matmul.cu` (extended)
Runs every version in `gemm_versions()` plus extra configurations (5 block shapes for v1/v2,
tiles 8/16 for v3) over 18 shapes. New shapes target tiling: `64×64×64` (exact tiles), `65×63×9`
(one element past a tile, and K just past the BK = 8 step), `100×200×300`.
**Verified before any GPU run:** a CPU emulation of the v3 and v4 kernels (same index
arithmetic, same load → barrier → compute phases) passed all shapes, and confirmed that v4
writes every output element exactly once.

### `src/benchmark/bench_matmul.cu` (extended)
- Experiment A: every version per size, with "vs prev" (the gain of each optimization step),
  "vs v1" and "vs CPU".
- Experiment B: Transformer shapes for every version.
- Experiment C: block shapes for v2.
- Experiment D: tile sizes for v3.
