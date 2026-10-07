# 07 — Attention (and the KV Cache)

Files covered: `kernels/attention.cu`, `kernels/attention.cuh`, `src/cpu/cpu_ops.cpp`
(`cpu::attention`), `tests/test_attention.cu`, `src/benchmark/bench_attention.cu`, and a small
fix in `kernels/softmax.cu` (v4 with masked −∞ scores).

New concepts: Query/Key/Value, scaled dot-product attention, causal masking, batched kernels
(`gridDim.z`), transposed tile loads with padding, FlashAttention-style fusion, the KV cache.

---

## 1. What problem are we solving?

In a sentence like "The animal didn't cross the street because **it** was tired", the model must
work out that "it" refers to "animal". **Attention** is the mechanism that lets each token gather
information from the other tokens. It's the defining operation of a Transformer.

### Query, Key, Value

Each token's hidden vector x is multiplied by three learned weight matrices (three GEMMs,
04 §1) to produce three vectors:

| Vector | Intuition | Shape per head |
|---|---|---|
| **Query** q = x·W_Q | "what am I looking for?" | d |
| **Key** k = x·W_K | "what do I contain?" (a label other tokens match against) | d |
| **Value** v = x·W_V | "what information do I hand over if someone attends to me?" | d |

Token i compares its query with every token's key. A larger dot product q·k means a better match.
The matches become weights (softmax), and token i's output is the weighted average of the values.

### Multi-head attention

The hidden vector (768 values in GPT-2 small) is split into **heads**: 12 heads of
**d = 64** each. Every head runs attention independently on its own 64-dimensional slice, so it
can learn a different kind of relationship. The heads are an extra parallel dimension for the GPU.

Our memory layout: `Q[heads][q_len][d]`, `K, V[heads][kv_len][d]`, `O[heads][q_len][d]`, each
head stored contiguously. (Real frameworks often store `[batch][seq][heads][d]`, which is
interleaved. Ours keeps the indexing simple; the kernels only differ in pointer offsets.)

---

## 2. The formula, step by step

```
O = softmax( Q Kᵀ / √d ) V            (for each head)

Q:  q_len  × d          K, V: kv_len × d
S = Q Kᵀ / √d     → q_len × kv_len    score of every query against every key
P = softmax(S)    → q_len × kv_len    each ROW sums to 1  (softmax over keys)
O = P V           → q_len × d         weighted average of value rows
```

1. **QKᵀ**: `S[i][j] = q_i · k_j = Σ_c Q[i][c]·K[j][c]`. This is a GEMM where the second operand
   is **transposed**: we need rows of K, not columns.
2. **Scaling by 1/√d**: a dot product of d random terms has a typical size that grows like √d.
   With d = 64, raw scores would be about 8× too large, and softmax of large scores becomes
   almost "one-hot" (one weight ≈ 1, the rest ≈ 0). The model then attends to one token only,
   and learning stalls. Dividing by √d keeps scores around size 1.
3. **Softmax** over each row (Phase 5). It needs max-subtraction for stability.
4. **PV**: a normal GEMM. Output row i = Σⱼ P[i][j] · v_j.

**FLOPs:** QKᵀ and PV are each 2·q_len·kv_len·d per head → 4·heads·q_len·kv_len·d in total.
**Memory:** S and P are heads × q_len × kv_len. That grows with the **square** of the sequence
length.

---

## 3. Hand example (first test in `tests/test_attention.cu`)

One head, 2 tokens, d = 2:

```
Q = | 1 0 |   K = | 1 0 |   V = | 1 2 |      scale = 1/√2 = 0.7071068
    | 0 1 |       | 0 1 |       | 3 4 |
```

Row 0 (query 0):
- scores = [q₀·k₀, q₀·k₁] · 0.7071 = [1·0.7071, 0] = [0.7071068, 0]
- softmax: e^0.7071068 = 2.0281150, e^0 = 1 → sum 3.0281150 → P₀ = [0.6697616, 0.3302384]
- O₀ = 0.6697616·[1, 2] + 0.3302384·[3, 4] = **[1.6604769, 2.6604769]**

Row 1 (query 1): scores = [0, 0.7071068] → P₁ = [0.3302384, 0.6697616] →
O₁ = **[2.3395231, 3.3395231]**

### Causal masking

In a decoder (GPT, LLaMA), token i must not look at tokens that come **after** it: during
generation, those tokens don't exist yet. So we set S[i][j] = −∞ for j > i. Then exp(−∞) = 0, and
the future gets zero weight.

With the mask: row 0 can only see key 0 → P₀ = [1, 0] → O₀ = V₀ = **[1, 2]**. Row 1 sees both
keys → unchanged.

When q_len < kv_len (§8, decoding), query i is the token at position `kv_len − q_len + i`, so the
rule becomes **key j is visible if j ≤ i + (kv_len − q_len)**. The code calls
`kv_len − q_len` the `causal_offset`.

---

## 4. CPU version (`cpu::attention`)

For each head and each query row: compute the visible scores (with the causal limit), take the
max, apply exp and sum, then the weighted sum of value rows. Everything in double precision.

---

## 5. Unfused GPU attention: three kernels

```
1. batched_gemm(Q, K, S, heads, q_len, kv_len, d, b_transposed = true, scale, causal)   S = Q Kᵀ · scale (+ mask)
2. softmax_online(S, S, heads·q_len, kv_len)                                            P = softmax(S), in place
3. batched_gemm(S, V, O, heads, q_len, d, kv_len, b_transposed = false, 1, no mask)     O = P V
```

All heads are handled by **one** launch per step, using a **batched** kernel.

### Batched kernels: `gridDim.z`

```cpp
dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);   // z = batch (heads)
...
const size_t batch = blockIdx.z;
A += batch * M * K;   B += batch * K * N;   C += batch * M * N;
```

The grid gets a third dimension. `blockIdx.z` says which head this block works on. Moving the
pointers to that head's matrices turns everything below into the ordinary single-GEMM code
(GEMM v3, 04 §16). One launch covers all 12 heads instead of 12 separate launches.

### Multiplying by Kᵀ without a transpose kernel

QKᵀ needs `S[i][j] = Σ_c Q[i][c] · K[j][c]`. With `b_transposed = true`, B (the K matrix) is
stored as N × K (kv_len × d), and the kernel needs the tile element `Bs[k][n] = K[n][k]`.

The **naive** load `Bs[ty][tx] = K[col][t·TILE + ty]` would have consecutive threads (tx) read
from different rows of K, which is uncoalesced (03 §2). Instead:

```cpp
const int n = blockIdx.x * TILE + ty;      // which row of K (= which key)
const int k = t * TILE + tx;               // which dimension
Bs[tx][ty] = (n < N && k < K) ? B[n * K + k] : 0.0f;
```

- Global read: consecutive `tx` → consecutive `k` within one row of K → **coalesced** ✔.
- The tile is **transposed while being stored**: element (row n, column k) goes to `Bs[k][n]`.
  The compute loop `sum += As[ty][k] * Bs[k][tx]` is then unchanged from GEMM v3.

### Why `Bs[TILE][TILE + 1]` (padding)

The store `Bs[tx][ty]`: in one warp, `ty` is fixed and `tx` = 0…31, so the warp writes a
**column** of the tile. Word address = `tx · rowlength + ty`.

- Without padding (row length 32): addresses 0·32 + ty, 1·32 + ty, 2·32 + ty, … all fall in
  **the same bank** (address mod 32 = ty) → a **32-way bank conflict** (04 §18). The store is
  serialized 32 times.
- With padding (row length 33): bank = (tx·33 + ty) mod 32 = (tx + ty) mod 32 → **32 different
  banks** ✔. One unused float per row of the tile fixes it.

The compute loop reads `Bs[k][tx]` = consecutive words → also conflict-free.

### Epilogue: scale and mask

```cpp
float v = sum * scale;
if (causal && col > row + causal_offset) v = -INFINITY;
C[row * N + col] = v;
```

Work done on the result just before it is stored is called the **epilogue** of a GEMM. Fusing
the scale and the mask here avoids two extra passes over S.

### Softmax in place

`softmax_online(S, S, …)` uses the same pointer for input and output. That is safe because each
lane reads `x[c]` and then writes `y[c]` for the **same** element, after the row's max and sum
are already known (05 §10).

**Fix made in this phase:** masked rows contain −∞. If a lane's first values were −∞ (and its
running max was still −∞), v4 computed exp(−∞ − (−∞)) = exp(NaN) = NaN. The update now skips
−∞ values (`else if (v > -INFINITY)`). They contribute exp(−∞) = 0 anyway.

### The cost: S lives in global memory

`S` is heads × q_len × kv_len floats. For 12 heads, seq = 2048: 12 × 2048 × 2048 × 4 B ≈
**201 MB**. It is written by kernel 1, read and written by kernel 2, and read by kernel 3:
about 4 × 201 MB of traffic for one attention layer, all to handle an intermediate result
nobody needs afterwards. The memory also grows with seq², which limits the maximum context
length. `bench_attention` prints this workspace size.

---

## 6. Fused attention (FlashAttention-style): never store S

### The idea

Phase 5's online softmax computes softmax in one pass with a running max m and running sum l
(05 §10). The same trick works for the **output**: keep a running, unnormalized output o, and
rescale it whenever the max changes:

```
for each key j:
    s       = q · k_j                    (scaled)
    m_new   = max(m, s)
    corr    = exp(m − m_new)             (rescales everything accumulated so far)
    p       = exp(s − m_new)
    l       = l · corr + p
    o       = o · corr + p · v_j
    m       = m_new
O = o / l
```

### Worked example (1-dimensional values)

Scores [1, 3], values v = [10, 20]:

| key | s | m_new | corr | p | l | o |
|---|---|---|---|---|---|---|
| start | | −∞ | | | 0 | 0 |
| 0 | 1 | 1 | e^(−∞) = 0 | 1 | 1 | 10 |
| 1 | 3 | 3 | e^(1−3) = 0.1353 | 1 | 1.1353 | 10·0.1353 + 20 = 21.353 |

O = 21.353 / 1.1353 = **18.808**.

Check: softmax([1, 3]) = [0.1192, 0.8808] → 0.1192·10 + 0.8808·20 = 18.808 ✔.

The scores were never stored. Each was used once and forgotten.

### Work decomposition

```
grid  = ( ceil(q_len / 8),  heads )      block = 256 threads = 8 warps
warp w of block (bx, head)  →  query row  q_row = bx·8 + w  of that head
lane l of the warp          →  dimensions l, l+32, l+64, …  of q and o  (d/32 values per lane)
```

The block walks over the keys in tiles of 32. Each tile of K and V (32 × d) is loaded into
**shared memory once** and used by all 8 warps (8 queries). That cuts K/V global reads 8×
compared with each warp reading them itself.

### Line by line (the important parts)

```cpp
const bool active = q_row < q_len;
```
In softmax, extra warps simply `return`ed (05 §8). **Here they must not**: every warp helps
load the K/V tiles, and every warp must reach every `__syncthreads()` (04 §14). Inactive warps
just skip the computation.

```cpp
float q[PER_LANE], o[PER_LANE];
q[i] = active ? Q[q_row * D + lane + 32 * i] * scale : 0.0f;
```
Each lane keeps its slice of the query in registers for the whole kernel. The query is
pre-multiplied by the scale once, instead of multiplying every score.

```cpp
const int my_keys    = causal ? min(kv_len, q_row + causal_offset + 1) : kv_len;
const int last_row   = min(q_len - 1, blockIdx.x * FUSED_WARPS + FUSED_WARPS - 1);
const int block_keys = causal ? min(kv_len, last_row + causal_offset + 1) : kv_len;
```
With causal masking, query `q_row` may see keys `0 … q_row + offset`. The block must load tiles
up to the **last** query it holds (`block_keys`). Each warp only processes keys up to its own
limit (`my_keys`). Keys beyond the limit are **skipped**, not computed and masked. This makes
causal attention about 2× cheaper than non-causal.

```cpp
for (int i = threadIdx.x; i < KEY_TILE * D; i += blockDim.x) { ... Ks[r][c] = ...; Vs[r][c] = ...; }
__syncthreads();
```
A block-stride cooperative load (04 §21): 256 threads copy 32 × d floats of K and V, coalesced.
Keys past the end are filled with 0.

```cpp
float partial = 0.0f;
for (int i = 0; i < PER_LANE; ++i) partial += q[i] * Ks[j][lane + 32 * i];
const float s = warp_reduce_sum(partial);
```
The score q · k_j is a dot product of length d, split across the 32 lanes: each lane does d/32
multiply-adds, then 5 shuffles add the 32 partial sums. Every lane receives the full score
(05 §7). Reading `Ks[j][lane + 32i]`: consecutive lanes read consecutive words → no bank
conflicts.

```cpp
const float m_new = fmaxf(m, s);
const float correction = expf(m - m_new);
const float p = expf(s - m_new);
l = l * correction + p;
for (...) o[i] = o[i] * correction + p * Vs[j][lane + 32 * i];
m = m_new;
```
The online update from the table above. Every lane does it with the same m, l, p, so the warp
stays in step; each lane updates only its own d/32 output values.

```cpp
__syncthreads();      // after the computation, before the next tile is loaded
...
O[q_row * D + lane + 32 * i] = o[i] * inv_l;    // normalize once at the end
```

### Memory comparison

| | Unfused | Fused |
|---|---|---|
| Extra global memory | heads × q_len × kv_len floats (S/P) | none |
| Global traffic for S/P | ≈ 4 × heads × q_len × kv_len × 4 B | 0 |
| Kernel launches | 3 | 1 |
| K, V reads | once per GEMM (through tiles) | once per block of 8 queries |

### Honest limitations of our fused kernel

The **memory** advantage is guaranteed. The **speed** advantage is not: our fused kernel is
compute-inefficient compared with the tiled GEMMs.
- Per key, each warp does only d/32 FMAs per lane, but then 5 shuffles and 2 `expf` calls.
  The unfused version uses tiled GEMMs with high reuse for QKᵀ and PV.
- At large seq, the unfused GEMMs may therefore win on time even though they move more bytes.
  Measure it, and be ready to explain it.

What real **FlashAttention** does differently:
- Each block takes a **tile of queries** (e.g. 64–128) and multiplies it with a **tile of keys**
  as a small GEMM (Q_tile · K_tileᵀ) on **Tensor Cores** in FP16/BF16.
- It applies the online softmax per tile row, then multiplies by V_tile, again as a GEMM.
- So it gets both GEMM efficiency and no seq² memory.

Ours is the same algorithm with "one query per warp, one key at a time", which keeps the code
readable. This is the main thing to say in an interview if asked "how does yours compare to
FlashAttention?".

---

## 7. Synchronization summary

| Kernel | Barriers | Why |
|---|---|---|
| batched_gemm | 2 per K-tile | the same as GEMM v3 |
| softmax_online | none | warp per row |
| fused | 2 per key tile | the tile must be complete before use, and fully used before overwriting |

Warps in the fused kernel never `return` early: they must reach every barrier.

---

## 8. The KV cache

### Autoregressive generation

An LLM generates text **one token at a time**. To produce token t + 1, it runs the model on the
sequence so far and samples from the output. Then it appends the token and repeats.

**Without a cache**, every step recomputes everything for all t tokens: K/V projections for every
token, and attention for every query.

**Key observation:** with causal masking, the keys and values of earlier tokens **never change**
when new tokens are appended. Token 5's k and v are the same whether the sequence is 6 or 600
tokens long.

### The KV cache

So we **store** each layer's K and V rows for all previous tokens. At each new step:
1. Compute q, k, v only for the **one** new token (a GEMM with M = 1, i.e. GEMV, 03 §5).
2. Append k, v to the cache.
3. Run attention with **q_len = 1** against **kv_len = t** cached keys/values.

That's exactly `AttentionShape{heads, 1, L, d, causal = true}`: our kernels support it directly,
because q_len and kv_len are separate parameters.

| | Without cache (per new token) | With cache (per new token) |
|---|---|---|
| Projection GEMMs | for all t tokens | for 1 token |
| Attention | t queries × t keys | 1 query × t keys |
| Total over T generated tokens | ~ T³ (attention) | ~ T² |

### The cost: memory

Per token, per layer: one K row and one V row of `heads × d = hidden` values.

```
KV cache bytes per token = 2 (K and V) × layers × hidden × bytes per value
GPT-2 small, FP16:   2 × 12 × 768  × 2 B =  36 KB per token  → 2048 tokens ≈  72 MB per sequence
LLaMA-7B,   FP16:    2 × 32 × 4096 × 2 B = 512 KB per token  → 2048 tokens ≈   1 GB per sequence
```

(`bench_attention` prints the GPT-2 numbers.) With many users served at once, the KV cache, not
the weights, often dominates GPU memory. That's why serving systems:
- manage it in pages to avoid fragmentation (**PagedAttention**, vLLM),
- share K/V between heads (**multi-query / grouped-query attention**, used by LLaMA-2-70B and
  later models: 8 K/V heads for 64 query heads),
- **quantize** it to FP8 or INT8 (08 §9).

### Decode attention is memory-bound

For q_len = 1, each cached key is used for d multiply-adds and then never again in this step:
about 2 FLOPs per 4 bytes read. Decode attention's speed is set by **how fast the KV cache can
be read**, which is another reason FP16/FP8 caches matter. Our fused kernel with q_len = 1 also
shows a parallelism problem: only `heads` blocks (12), each with one active warp. Production
kernels ("flash-decoding") **split the keys across many blocks** and merge the partial (m, l, o)
results with the same rescaling rule. It's the split-row idea from 05 §11 again.

---

## 9. Correctness evidence

- `test_attention`: the hand example (non-causal and causal, unfused), plus 11 random cases for
  both implementations against `cpu::attention`: a single token; q_len ≠ kv_len; causal with an
  offset (100 queries, 300 keys); a decode step (1 × 513); q_len not a multiple of 8 (partially
  filled blocks); kv_len not a multiple of 32 (partial key tiles); d = 32, 64, 128; and the
  GPT-2 shape 12 × 128 × 64, causal and non-causal.
- **Verified before any GPU run:** a CPU emulation of both kernels' exact logic (transposed tile
  loads, mask epilogue, warp-shuffle order, per-block and per-warp key limits, online rescaling)
  passed 9 of these cases with a worst error of 1.2e-7. The hand-example values come from the
  CPU reference.

---

## 10. Experiments in `bench_attention`, and how to read them

**A (prefill, 12 heads, d = 64, seq 128–2048, non-causal and causal):**
- The breakdown columns show where unfused time goes. As seq grows, softmax (memory-bound over
  seq² elements) and the two GEMMs (compute over seq²·d) all scale with seq².
- `workspace MB` grows with seq²: that's the memory problem.
- `speedup` = unfused / fused. Compare it with §6's limitations. Causal fused should be roughly
  2× cheaper than non-causal fused (skipped keys); causal unfused is not (it computes everything
  and masks).
- Measured on a T4 (12_results §6): our fused kernel was 0.56–0.77× the speed of unfused for
  non-causal attention, but 1.2× faster with causal masking at seq ≥ 1024, using no score
  buffer (unfused: 192 MB at seq 2048). Causal fused cost about half of non-causal fused, as
  predicted.
- Compare with PyTorch's `naive (3 kernels)` and `SDPA (fused)` rows from `python/benchmark.py`.
  Note that on the T4 with FP32 inputs, SDPA saved **no** memory (438 MB at seq 2048): PyTorch's
  memory-saving backends weren't used for FP32 there. See the FP16 SDPA rows.
  (same non-causal shapes).

**B (KV cache):** time of one decoding step with the cache (1 × L) against recomputing attention
for all L tokens (L × L). The ratio grows with L, roughly linearly. Without a cache, the
projection GEMMs would also be recomputed, so the real difference is even larger.

---

## 11. Common mistakes

1. Forgetting the 1/√d scale → softmax saturates (§2).
2. Multiplying Q by K instead of Kᵀ, or loading the transposed tile in an uncoalesced way.
3. Off-by-one in the causal mask, or ignoring the offset when q_len < kv_len (decode). The test
   covers 100 × 300 causal.
4. `return`ing from inactive warps in a kernel that uses `__syncthreads()`.
5. Online softmax with −∞ inputs producing NaN (fixed in v4, §5).
6. Allocating S for long contexts: seq² memory runs out quickly.
7. Applying softmax over the wrong axis. Softmax is over **keys** (each row of S).
8. Assuming the fused kernel is automatically faster. Fusion saves memory traffic; it doesn't
   make the compute efficient.

## 12. Interview explanation (~2 minutes)

> "Attention computes softmax(QKᵀ/√d)·V per head. Each token's query is compared with every
> key, the scores become weights, and the output is a weighted average of the values. The √d
> scaling keeps the scores around unit size so softmax doesn't saturate. My unfused version is
> three launches: a batched tiled GEMM for QKᵀ that uses `blockIdx.z` for the head, and loads K
> coalesced but stores the tile transposed into shared memory with one column of padding to
> avoid a 32-way bank conflict, with the scale and causal mask applied in the epilogue; then my
> online softmax in place; then a batched GEMM with V. Its problem is the heads × seq × seq score
> matrix: about 200 MB at seq 2048, written and read several times. My fused version is
> FlashAttention-style: each warp owns a query, the block streams K and V tiles through shared
> memory, and each warp keeps a running max, sum and output that are rescaled by exp(m_old −
> m_new) whenever the max changes, so the score matrix never exists. With causal masking, keys
> beyond the query's position are skipped entirely. It's not as fast as real FlashAttention,
> which does the Q-tile × K-tile products as Tensor Core GEMMs in FP16 instead of one key at a
> time with warp shuffles. For generation I explained and measured the KV cache: past keys and
> values don't change under causal masking, so each new token only needs q_len = 1 against the
> cache, at the cost of 2 × layers × hidden × bytes per token of memory. Decode attention is
> memory-bound on reading that cache."

## 13. What to remember

- Q = what I look for, K = what I contain, V = what I give. O = softmax(QKᵀ/√d)·V.
- Scores are q_len × kv_len: **quadratic** memory if stored.
- Causal mask: key j is visible to query i if j ≤ i + (kv_len − q_len).
- Batched kernels: `blockIdx.z` = batch/head; offset the pointers, then it's the ordinary
  kernel.
- Transposed tile store → pad shared arrays by +1 to avoid bank conflicts.
- Fusion with online softmax removes the seq² memory. Speed also needs GEMM-style tiling and
  Tensor Cores.
- KV cache: store past K/V, decode with q_len = 1; memory = 2·layers·hidden·bytes per token.
  Decode is memory-bound.

---

## 14. File guide

### `kernels/attention.cu`
1. **What:** batched GEMM (optional transposed B, scale, causal mask), unfused attention
pipeline, fused attention kernel. 2. **Why:** attention is the Transformer's core operation and
the clearest example of fusion and memory trade-offs. 3. **Inputs:** device Q, K, V, an
`AttentionShape`; the unfused path also needs a workspace. 4. **Output:** O. 5. **Data flow:**
unfused: global → (GEMM) → S in global → (softmax) → P in global → (GEMM) → O. Fused: global K/V
tiles → shared → registers (q, o, m, l) → O. 6. **Functions:** `batched_gemm_kernel<TILE, B_TRANSPOSED>`,
`attention_fused_kernel<D>`, `gpu::batched_gemm`, `gpu::attention_unfused`,
`gpu::attention_fused`, `check_shape`, `launch_fused<D>`. 7. **Variables:** `blockIdx.z`/`batch`,
`Bs[TILE][TILE+1]`, `q_row`, `active`, `my_keys`, `block_keys`, `q[]`, `o[]`, `m`, `l`,
`correction`, `p`. 8. **CUDA concepts:** 3D grids, `if constexpr`, transposed shared stores,
padding, warp-collective reductions, cooperative tile loads. 9. **Memory:** §5, §6 tables.
10. **Thread mapping:** GEMM as v3 plus z = head; fused: warp ↔ query, lane ↔ d/32 dimensions.
11. **Sync:** §7. 12. **Performance:** §6 limitations, §8 decode. 13. **Mistakes:** §11.

### `kernels/attention.cuh`
`AttentionShape {heads, q_len, kv_len, d, causal}` and the launchers. q_len ≠ kv_len enables
KV-cache decoding.

### `src/cpu/cpu_ops.cpp` — `cpu::attention`
Double-precision reference with the causal rule, per head and per query row.

### `tests/test_attention.cu`, `src/benchmark/bench_attention.cu`
§9 and §10.

### `kernels/softmax.cu` (changed)
v4 skips −∞ values in the running update (§5).
