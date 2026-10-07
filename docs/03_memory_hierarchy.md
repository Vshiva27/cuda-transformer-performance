# 03 — The GPU Memory Hierarchy

Most GPU optimization comes down to one thing: **getting data to the arithmetic units fast
enough**. This document explains where data can live on a GPU, how fast each place is, how a
warp's memory request is actually served, and how to decide whether a kernel is limited by
memory or by math. Phase 2 (GEMM v1/v2) and Phase 3 (tiling) build directly on it.

---

## 1. The levels, from fastest to slowest

```
              ┌──────────────────────────── one SM ─────────────────────────────┐
              │  Registers  (per thread)          ~1 cycle,  64K x 32-bit / SM  │
  fast,       │  Shared memory (per block)  ┐     ~20-30 cycles                 │
  tiny        │  L1 cache                   ┘ same on-chip storage, 64-256 KB/SM│
              └──────────────────────────────────────────────────────────────────┘
                                      │
              ┌──────────── shared by all SMs ───────────┐
              │  L2 cache            ~200 cycles, a few MB (T4: 4 MB, A100: 40 MB)
              └───────────────────────────────────────────┘
                                      │
  slow,       ┌────────────────────────────────────────────┐
  huge        │  Global memory (DRAM / VRAM) ~400-800 cycles, GBs (T4: 16 GB)
              └────────────────────────────────────────────┘
                                      │  PCIe (~10-25 GB/s)
                              CPU memory (host)
```

(Cycle counts are typical orders of magnitude; they vary by GPU generation.)

| Level | Who can see it | Size | Declared how | Used in this project for |
|---|---|---|---|---|
| **Registers** | one thread | up to 255 per thread | ordinary local variables (`float sum`) | accumulators, indices |
| **Local memory** | one thread | big | automatic when registers run out ("spilling") | *we try to avoid it*: it is actually in global memory, so it is slow |
| **Shared memory** | all threads of one block | ~48 KB per block by default | `__shared__ float tile[32][32];` | GEMM tiles (Phase 3), reductions (Phases 5–6) |
| **L1 cache** | one SM | shares the hardware with shared memory | automatic | caches global loads |
| **L2 cache** | whole GPU | MBs | automatic | caches everything going to DRAM |
| **Global memory** | all threads, all kernels, and the CPU via `cudaMemcpy` | GBs | `cudaMalloc` | all inputs and outputs |

**Why the speed differs so much:** registers and shared memory are *inside* the SM, physically
right next to the arithmetic units. Global memory is separate DRAM chips on the board. Every
access must travel off the chip and back.

**The key difference between cache and shared memory:**
- A **cache** (L1/L2) decides on its own what to keep. It may throw away data you are about to
  reuse.
- **Shared memory** is managed by *you*. You load data into it explicitly and it stays there
  until you overwrite it.

That control is why tiled GEMM (Phase 3) uses shared memory instead of hoping that the cache
does the right thing.

---

## 2. How a warp's memory request is served

Global memory is not fetched one float at a time. The hardware moves data in fixed-size pieces:

- **Sector** = 32 bytes = 8 floats. This is the smallest unit read from DRAM/L2.
- **Cache line** = 128 bytes = 4 sectors = 32 floats.

When a warp executes a load instruction such as `x = p[idx]`, the hardware collects the 32
addresses (one per thread) and works out **which sectors** contain them. The fewer distinct
sectors, the fewer transactions, and the less wasted bandwidth.

### Pattern 1: consecutive (coalesced) ✔

```
thread:   0    1    2    3   …   31
address: p[0] p[1] p[2] p[3] … p[31]        → 128 bytes = 4 sectors, all bytes used
```
Efficiency = bytes used ÷ bytes fetched = 128 ÷ 128 = **100%**.

### Pattern 2: same address for all threads (broadcast) ✔

```
thread:   0    1    2   …  31
address: p[7] p[7] p[7] … p[7]              → 1 sector, value sent to all 32 threads
```
Only one transaction is needed. Cheap.

### Pattern 3: strided (uncoalesced) ✘

```
thread:   0      1        2         …   31
address: p[0]  p[K]    p[2K]        …  p[31K]    (K = 1024)
         └─sector A  └─sector B  └─sector C  …   → 32 different sectors
```
32 sectors × 32 bytes = 1024 bytes fetched to deliver 128 useful bytes. Efficiency = **12.5%**.
The other 7/8 of each sector is fetched and not used by this instruction. It *may* be used later
if it stays in cache, which is why real slowdowns are often smaller than 8×.

**Rule of thumb:** consecutive threads (`threadIdx.x`, `threadIdx.x + 1`, …) should access
consecutive addresses.

---

## 3. Row-major layout: how a 2D matrix lives in 1D memory

Memory is a single long line of addresses. A matrix must be "flattened". In **row-major**
order (C/C++, PyTorch default), we store row 0 first, then row 1, and so on.

A 3 × 4 matrix (3 rows, 4 columns):

```
Logical view                      Memory (one line)
        col0 col1 col2 col3
row 0 [  a    b    c    d  ]      index:  0  1  2  3  4  5  6  7  8  9 10 11
row 1 [  e    f    g    h  ]      value:  a  b  c  d  e  f  g  h  i  j  k  l
row 2 [  i    j    k    l  ]              └─ row 0 ─┘ └─ row 1 ─┘ └─ row 2 ─┘
```

**Formula:** `M[row][col]` is at index `row * num_cols + col`.

Why: to reach row `row`, skip `row` complete rows of `num_cols` elements each. Then move `col`
steps inside the row. Example: `g` is at row 1, col 2 → `1 * 4 + 2 = 6`. ✔ (index 6 holds `g`)

Two important consequences:
- Moving **right along a row** (col + 1) means address + 1: **consecutive** memory.
- Moving **down a column** (row + 1) means address + `num_cols`: **strided** memory.

So "which direction do neighboring threads move?" decides whether access is coalesced.
This is exactly the difference between GEMM v1 and v2.

For GEMM with A: M×K, B: K×N, C: M×N:

| Element | Number of columns | Flat index |
|---|---|---|
| `A[row][k]` | K | `row * K + k` |
| `B[k][col]` | N | `k * N + col` |
| `C[row][col]` | N | `row * N + col` |

The most common GEMM bug is using the wrong number of columns (e.g. `A[row * N + k]`). Our tests
use M, N, K that are all different, so this bug cannot pass them.

---

## 4. Arithmetic intensity and the roofline model

Every kernel needs two resources: **math** (FLOP/s) and **memory traffic** (bytes/s). Which one
limits it?

**Arithmetic intensity (AI)** = FLOPs performed ÷ bytes moved to/from global memory.

The GPU has two ceilings:
- Peak compute `P` (FLOP/s), e.g. T4 ≈ 8,100 GFLOP/s FP32.
- Peak bandwidth `B` (bytes/s), e.g. T4 ≈ 320 GB/s.

A kernel with intensity AI can at best achieve:

```
attainable FLOP/s = min( P , AI × B )
```

This is the **roofline model**:

```
 FLOP/s
   │                       ___________________  ← compute roof P
   │                      /
   │                     /
   │                    /   ← memory roof: AI × B
   │                   /
   │                  /
   │________________/____________________________ AI (FLOP/byte)
                    ↑
              ridge point = P / B   (T4: 8100/320 ≈ 25 FLOP/byte)
```

- **Left of the ridge (AI < P/B): memory-bound.** Faster math would not help. Move fewer bytes
  or reuse data more.
- **Right of the ridge (AI > P/B): compute-bound.** Faster memory would not help. Do math more
  efficiently (more parallelism, FP16, Tensor Cores).

The benchmark prints your GPU's ridge point at startup (`device_info.cu`).

### Two examples

| Kernel | FLOPs | Minimum bytes | AI | On a T4 |
|---|---|---|---|---|
| Vector add, n elements | n | 12n | 0.083 | far left: memory-bound |
| GEMM n×n×n | 2n³ | 3 · 4n² = 12n² | n/6 | n = 1024 → AI ≈ 170: compute-bound *if* data is reused perfectly |

The phrase **"if data is reused perfectly"** is the whole story of GEMM optimization. GEMM *can*
be compute-bound, because each element of A is used N times and each element of B is used M
times. A naive kernel re-reads them from memory every time it uses them, so its *real* intensity
is far lower. Each optimization in Phase 3 increases real reuse:

```
v1/v2: reuse only by luck (caches)
v3:    reuse from shared memory (each element loaded once per tile, used TILE times)
v4:    reuse from registers (each value loaded once, used for several outputs)
```

---

## 5. Where Transformer inference sits on the roofline

- **Prefill** (processing the whole prompt at once): GEMMs with M = number of tokens (hundreds to
  thousands). High AI, so it can be **compute-bound**.
- **Decode** (generating one token at a time): M = 1 (or the batch size). A 1×K vector times a
  K×N weight matrix reads all K×N weights to do 2·K·N FLOPs, so AI ≈ 0.5 FLOP/byte in FP32.
  Strongly **memory-bound**.
- Softmax, LayerNorm, residual add: few FLOPs per byte. **Memory-bound**.

This is why LLM token generation speed depends mostly on **memory bandwidth**, and why weight
quantization (fewer bytes per weight) speeds it up. `bench_matmul` Experiment B measures both
prefill-shaped and decode-shaped GEMMs, so you can see this on your GPU.

---

## 6. Common mistakes

1. Mapping `threadIdx.x` to the direction that is strided in memory (GEMM v1).
2. Using the wrong row length in a flat index (`row * N + k` for A instead of `row * K + k`).
3. Assuming the cache will fix a bad access pattern. It helps a little, but not reliably.
4. Using too many registers per thread → spilling to slow local memory, or fewer warps per SM.
5. Judging a memory-bound kernel by GFLOP/s or a compute-bound kernel by GB/s. Use the roofline
   to decide which metric matters.

## 7. What to remember

- Registers ≫ shared memory ≫ L2 ≫ global memory, in speed. The size order is the reverse.
- A warp's load is served in 32-byte sectors. Consecutive threads should read consecutive
  addresses.
- Row-major: `M[r][c] = M[r * cols + c]`. Along a row = consecutive; down a column = strided.
- AI = FLOPs / bytes. Below the ridge point you are memory-bound; above it, compute-bound.
- GEMM *can* be compute-bound only if data is reused. Optimizations exist to create reuse.
- LLM decode is memory-bound; prefill can be compute-bound.
