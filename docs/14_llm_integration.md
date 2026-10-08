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

**How to read it (predictions, not measurements yet):**

- Every row includes Python and launch overhead: about 15 kernel launches per layer per token.
  For **GPT-2 small** (124M parameters, 12 layers, 0.5 GB of FP32 weights) that overhead is
  likely the largest part of the time, so the weight format will matter little, and a
  difference to Hugging Face mostly measures framework overhead, not kernels.
- For **GPT-2 XL** (1.5B parameters, 48 layers, 6.2 GB FP32), reading the weights once per token
  takes about 4.5 ms in FP32 at the A100's measured ~1.37 TB/s, 1.1 ms in INT8. There the weight
  format should show up in tokens per second, but not as the 3.2× of the isolated GEMV, because
  the overhead and the non-GEMV kernels don't shrink.
- Hugging Face FP16 also halves activation bytes and uses Tensor Cores; our rows keep FP32
  activations. Compare our INT8 row with it as "a different design", not like for like.

Results go to `benchmarks/<GPU>/gpt2.csv`. Not measured yet.

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
> position and the greedy text token by token. The benchmark is in tokens per second, and I
> expect it to show that at batch size 1 a small model is launch-bound, so the INT8 kernel only
> pays off for a model large enough that reading the weights dominates."
