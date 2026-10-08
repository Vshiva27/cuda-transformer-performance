# 14 — Using the Kernels in a Real LLM: GPT-2 Text Generation

Files covered: `llm/gpt2_ops.cpp`, `llm/run_gpt2.py`, and the `kv_capacity` field added to
`gpu::AttentionShape` (`kernels/attention.cuh`, `kernels/attention.cu`).

---

## 1. What this adds

Docs 02–13 measure each kernel on its own. This step runs a **real, pretrained LLM** (GPT-2 from
Hugging Face, any size from 124M to 1.5B parameters) with its decoding done by this project's
kernels, and checks the generated text against Hugging Face Transformers token by token.

It answers two questions the kernel benchmarks can't:

1. **Are the kernels correct inside a full model?** A layout or indexing bug that a unit test
   misses shows up as different logits or different generated text.
2. **What do the kernel speedups mean for tokens per second?** One decoding step runs about 15
   kernels per layer; the weights (GEMV) are only part of the time.

## 2. Which kernel runs where

One GPT-2 decoder layer for one new token (`layer_decode` in `gpt2_ops.cpp`):

```
x (residual stream, n_embd floats)
 ├─ LayerNorm                         gpu::layernorm_block                  ours
 ├─ QKV projection   (3·n_embd × n_embd) gpu::gemv_fp32 / gemv_fp16 / gemv_int8(_multirow)  ours
 │    + bias                          at::add_                               PyTorch
 ├─ write k, v into the KV cache      Tensor::copy_                          PyTorch
 ├─ attention, q_len = 1, causal      gpu::attention_fused (kv_capacity)     ours
 ├─ output projection + bias          GEMV ours, bias PyTorch
 ├─ residual add + LayerNorm          gpu::add_layernorm (fused)             ours
 ├─ MLP up (4·n_embd × n_embd) + bias GEMV ours, bias PyTorch
 ├─ GELU (tanh approximation)         at::gelu                               PyTorch
 ├─ MLP down + bias                   GEMV ours, bias PyTorch
 └─ residual add                      at::add_                               PyTorch
```

Before the first layer: token + position embedding (PyTorch indexing). After the last: the
final LayerNorm (ours) and the LM head, a GEMV with the token-embedding matrix (ours; GPT-2 ties
the two). Greedy decoding takes the argmax of the logits.

**All the bytes-heavy and math-heavy work runs on our kernels**: every weight matrix, every
LayerNorm and the attention. PyTorch only does small elementwise glue (bias, GELU, residual,
cache writes), each a few thousand floats.

## 3. What had to change to fit a real model

1. **Weight layout.** Our GEMV expects W as `[out_features][in_features]` (like `nn.Linear`).
   Hugging Face's GPT-2 stores its linear layers as `Conv1D` with weight `[in, out]`, so
   `run_gpt2.py` transposes them once at load time. The LM head (`wte`, `[vocab][n_embd]`) is
   already in the right layout.
2. **A KV cache with spare capacity.** A real cache is allocated once for `max_len` tokens,
   `[heads][max_len][d]`, and filled one row per token. The fused attention kernel used to find
   head h's keys at `h · kv_len · d`, which assumes a packed cache. `AttentionShape` now has
   `kv_capacity` (keys allocated per head; 0 = packed), and the kernel uses
   `h · kv_capacity · d`. With the default 0 the indexing is exactly the old one, so earlier
   results are unaffected. `test_attention` checks a cache whose unused rows are NaN: the result
   must equal the packed layout bit for bit.
3. **Query layout.** The QKV projection writes q, k, v back to back, each `[heads][d]`. For one
   token that is already the `[heads][q_len = 1][d]` layout attention expects, so q is passed
   without copying, and the attention output `[heads][d]` is already the concatenation of heads
   the output projection expects.
4. **Weight formats.** At load time each linear weight is converted to FP32, FP16, or INT8 with
   one scale per output row (`scale = max|w| / 127`, round half to even, clamp to ±127, the same
   rule as `cpu::quantize_int8`). The LM head uses the same format.
5. **Fewer Python calls.** A whole layer is one C++ call (`layer_decode`), so Python runs once
   per layer, not once per kernel.

## 4. How it is verified

`python3 llm/run_gpt2.py --verify` compares against Hugging Face's FP32 model (TF32 off):

- **Prompt logits:** the prompt goes through our model token by token; the logits at every
  position are compared with Hugging Face's, as max |difference| / max |logit|. Our FP32 path
  must be within **1e-3**, or the script exits with an error. Different summation orders
  (our GEMV vs cuBLAS) are the only expected difference.
- **Greedy text:** both generate 32 tokens greedily; the script reports how many leading tokens
  are identical, and prints our FP32 text.
- FP16 and INT8 weights are expected to differ more (they change the weights). The script
  reports their logit difference and token agreement; that is a measurement, not a pass/fail.

**Checked before any GPU run** (CPU, PyTorch 2.14, Transformers 5.19): a PyTorch stand-in with
the same semantics as `gpt2_ops.cpp` (same arguments, layouts and cache writes), driven by the
real `CudaGPT2` class from `run_gpt2.py`, on a random 3-layer GPT-2: FP32 logits match Hugging
Face to **2.5e-7** of the logit scale and 12/12 greedy tokens are identical (FP16: 2.1e-4,
INT8: 6.9e-3, both 12/12). This validates the weight conversion, transposes, q/k/v split, head
layout, positions and tied LM head; the CUDA kernels themselves are covered by the C++ tests.

**Measured on a GPU** (NVIDIA A100-SXM4-80GB, 8-token prompt;
[`gpt2.txt`](../benchmarks/NVIDIA_A100_SXM4_80GB/gpt2.txt),
[`gpt2_xl.txt`](../benchmarks/NVIDIA_A100_SXM4_80GB/gpt2_xl.txt)):

| Model | Weights | Prompt logits: max \|diff\| / max \|logit\| | Next-token argmax agrees (prompt positions) | Greedy tokens identical to Hugging Face |
|---|---|---|---|---|
| GPT-2 (124M) | FP32 | **1.37e-6** | 100% | **32 / 32** |
| GPT-2 (124M) | FP16 | 3.20e-4 | 100% | 32 / 32 |
| GPT-2 (124M) | INT8, per-row scale | 1.77e-2 | 100% | 4 / 32 |
| GPT-2 XL (1.5B) | FP32 | **1.05e-6** | 100% | **32 / 32** |
| GPT-2 XL (1.5B) | FP16 | 1.59e-4 | 100% | 32 / 32 |
| GPT-2 XL (1.5B) | INT8, per-row scale | 7.63e-3 | 100% | **32 / 32** |

- **The kernels are correct inside the full model,** for both sizes. FP32 agrees with Hugging
  Face to about 1e-6 of the logit scale (the limit was 1e-3), and the generated text is identical.
- **FP16 weights** change the logits by 0.02–0.03% and the text not at all over 32 tokens.
- **INT8 weights** change GPT-2 small's logits by 1.8% of the logit scale. Every next-token
  choice on the prompt still matched, but greedy generation picked a different token at position
  5, and from there the two texts differ (each later token is conditioned on the earlier ones).
  On GPT-2 XL the INT8 error is 0.8% and all 32 tokens match. This is per-row INT8 with no
  calibration, applied to every layer including the LM head; why the larger model is less
  affected, and which layers contribute most, was not measured. 32 tokens from one prompt is a
  spot check, not a quality evaluation (no perplexity was measured).

## 5. How the speed is measured

`--bench` measures wall-clock time per generated token at batch size 1, after a warm-up, from
the same prompt position each time:

| Row | Weights | Kernels |
|---|---|---|
| Hugging Face, FP32 | FP32 | PyTorch eager (cuBLAS GEMMs, SDPA attention) |
| ours, FP32 weights | FP32 | this project |
| ours, FP16 weights | FP16 | this project (`gemv_fp16`) |
| ours, INT8 weights, 1 row/warp | INT8 | this project (`gemv_int8`) |
| ours, INT8 weights, 2 rows/warp | INT8 | this project (`gemv_int8_multirow`, R = 2) |
| Hugging Face, FP16 | FP16 | PyTorch eager, FP16 activations too |

**How to read it (predictions written before the runs):**

- Every row includes Python and launch overhead: about 15 kernel launches per layer per token.
  For **GPT-2 small** (124M parameters, 12 layers, 0.5 GB of FP32 weights) that overhead is
  likely the largest part of the time, so the weight format will matter little, and a
  difference to Hugging Face mostly measures framework overhead, not kernels.
- For **GPT-2 XL** (1.5B parameters, 48 layers, 6.2 GB FP32), reading the weights once per token
  takes about 4.5 ms in FP32 at the 40 GB A100's measured ~1.37 TB/s (about 3 ms on the 80 GB
  model), 1.1 ms in INT8. There the weight format should show up in tokens per second, but not as the 3.2× of the isolated GEMV, because
  the overhead and the non-GEMV kernels don't shrink.
- Hugging Face FP16 also halves activation bytes and uses Tensor Cores; our rows keep FP32
  activations. Compare our INT8 row with it as "a different design", not like for like.

### Measured: GPT-2 small (124M) on an A100-SXM4-80GB

[`benchmarks/NVIDIA_A100_SXM4_80GB/gpt2.csv`](../benchmarks/NVIDIA_A100_SXM4_80GB/gpt2.csv); batch 1,
128 generated tokens after an 8-token prompt, wall clock. Note: this is the **80 GB** A100
(about 2.0 TB/s peak), not the 40 GB model of 12_results §9.

| Implementation | Weights read per token | ms / token | tokens / s | vs Hugging Face FP32 |
|---|---|---|---|---|
| Hugging Face, FP32 | 497.8 MB | 9.028 | 110.8 | 1.00× |
| ours, FP32 weights | 494.1 MB | 1.636 | 611.1 | 5.52× |
| ours, FP16 weights | 247.1 MB | 1.618 | 618.0 | 5.58× |
| ours, INT8, 1 row/warp | 124.1 MB | 1.587 | 630.0 | 5.69× |
| ours, INT8, 2 rows/warp | 124.1 MB | 1.577 | 634.3 | 5.73× |
| Hugging Face, FP16 | 248.9 MB | 9.019 | 110.9 | 1.00× |

**What this shows, and what it doesn't:**

1. **GPT-2 small at batch 1 is launch-bound, as predicted.** Cutting the weights 4× (494 →
   124 MB) changed our time by only 3.6% (1.636 → 1.577 ms). Reading 494 MB at ~2 TB/s takes
   about 0.25 ms, and about 185 kernel launches per token fill the rest: ~1.6 ms / 185 ≈ 8.5–8.8 µs
   per launch including dispatch. While the CPU is the bottleneck the GPU waits between
   kernels, so faster kernels barely change the wall time.
2. **The 5.5× over Hugging Face is overhead, not faster kernels.** Hugging Face FP32 and FP16
   take the same 9.0 ms although FP16 reads half the bytes and uses Tensor Cores, so Hugging
   Face is overhead-bound too. Ours is faster because one decoder layer is one C++ call instead
   of a stack of Python modules. Do not quote it as a kernel speedup.
3. **To see the weight format matter, the model must be larger** (GPT-2 XL, below), or the
   launches must go away (CUDA Graphs, fused glue kernels).

### Measured: GPT-2 XL (1.5B) on the same A100-SXM4-80GB

[`benchmarks/NVIDIA_A100_SXM4_80GB/gpt2_xl.csv`](../benchmarks/NVIDIA_A100_SXM4_80GB/gpt2_xl.csv);
same settings. 48 layers, n_embd = 1600; one layer's FP32 weights (123 MB) are larger than the
40 MB L2, so every token reads all weights from DRAM.

| Implementation | Weights read per token | ms / token | tokens / s | vs Hugging Face FP32 | vs our FP32 |
|---|---|---|---|---|---|
| Hugging Face, FP32 | 6,230 MB | 34.143 | 29.3 | 1.00× | – |
| ours, FP32 weights | 6,220 MB | 7.537 | 132.7 | 4.53× | 1.00× |
| ours, FP16 weights | 3,110 MB | 6.524 | 153.3 | 5.23× | 1.16× |
| ours, INT8, 1 row/warp | 1,558 MB | 6.711 | 149.0 | 5.09× | 1.12× |
| ours, INT8, 2 rows/warp | 1,558 MB | 6.509 | 153.6 | 5.25× | 1.16× |
| Hugging Face, FP16 | 3,115 MB | 36.017 | 27.8 | 0.95× | – |

1. **The weight format now shows up, but only once.** FP32 → FP16 saves 1.0 ms per token
   (1.16×). FP16 → INT8 saves nothing (6.524 vs 6.509 ms), although INT8 reads half the bytes.
2. **FP16 and INT8 sit on the launch floor.** About 725 launches per token (48 layers × 15 + 5)
   at the ~8.5 µs per launch measured on GPT-2 small give **≈ 6.2 ms**; FP16 and INT8 measure
   6.5 ms. Reading the weights takes far less: about 1.8 ms (FP16) and 0.9 ms (INT8) at 85% of
   the ~2.0 TB/s peak, so the GPU finishes each kernel before the CPU has launched the next.
   INT8's smaller weights can't make a CPU-bound loop faster.
3. **FP32 is above the floor** (7.5 ms; reading 6.2 GB takes about 3.6 ms at 85% of peak). The
   likely reason: its largest GEMVs take longer than the launch interval (an XL MLP layer is
   41 MB in FP32, ~24 µs), so in those parts of the step the GPU, not the CPU, is the
   bottleneck. Not verified with a timeline.
4. **The ~5× over Hugging Face is again overhead.** Hugging Face FP16 (36.0 ms) is no faster
   than FP32 (34.1 ms): it is overhead-bound as well.
5. **Conclusion for INT8 end to end:** at batch 1 with ~15 launches per layer, the GEMV's 3.2×
   (12_results §9.2) cannot reach tokens per second, because the kernel time is hidden behind
   the launch time. The next step would be removing launches (CUDA Graphs over the decode step,
   fusing bias/GELU/residual into the GEMV and LayerNorm kernels), not faster GEMVs.

## 6. How to run it

On Colab (A100 or T4), after cloning the repository:

```bash
pip install transformers        # if not already installed
python3 llm/run_gpt2.py --verify --bench --model gpt2    --csv benchmarks/<GPU>/gpt2.csv
python3 llm/run_gpt2.py --verify --bench --model gpt2-xl --csv benchmarks/<GPU>/gpt2_xl.csv
```

The first run compiles the extension (about a minute) and downloads the checkpoint. `<GPU>` is
the folder name used by `run_all.sh`, e.g. `NVIDIA_A100_SXM4_40GB`.

## 7. Limits

- **Batch size 1, greedy decoding.** No batching, no sampling strategies.
- **FP32 activations.** Real deployments use FP16/BF16 activations; that needs FP16 versions of
  LayerNorm, attention and the GEMV inputs.
- **No prefill kernel.** The prompt is processed one token at a time through the decode path.
  A real engine runs the prompt as GEMMs (where cuBLAS beats our GEMMs; 12_results §9).
- **Glue on PyTorch.** Bias, GELU, residual add and cache writes are separate small PyTorch
  kernels; fusing them into our kernels would cut launches.
- **GPT-2 only.** LLaMA-style models also need RMSNorm, RoPE, SiLU-gated MLPs and grouped-query
  attention.
- **INT8 is weight-only and per-row**, with no calibration; the error is measured as logit
  difference and token agreement, not as model quality (perplexity).

## 8. Interview explanation

> "To check the kernels inside a real model, I ran GPT-2 decoding on them through a small
> PyTorch extension: every linear layer, LayerNorm and the attention use my kernels, PyTorch only
> does the elementwise glue. I had to change one thing in the kernels: real KV caches are
> preallocated, so the attention kernel got a capacity parameter for the per-head stride, tested
> with NaN in the unused rows. I verify against Hugging Face by comparing logits at every prompt
> position and the greedy text token by token: in FP32 the logits agree to 1.4e-6 and all 32
> generated tokens match, for GPT-2 small and XL. For speed, decoding at batch 1 turned out
> launch-bound: about 15 launches per layer at ~8.5 µs each. On GPT-2 XL that floor is about
> 6.2 ms per token, FP16 and INT8 both measure 6.5 ms, and only FP32 (7.5 ms) is above it, so
> INT8's 3.2× GEMV speedup doesn't reach tokens per second. The fix would be fewer launches, CUDA
> Graphs and fused glue, not faster GEMVs. My version was ~5× faster than Hugging Face eager, but
> Hugging Face FP32 and FP16 took the same time, so that gap is framework overhead, not my
> kernels, and I don't present it as a kernel result."
