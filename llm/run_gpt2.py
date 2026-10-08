"""
run_gpt2.py — GPT-2 text generation on this project's CUDA kernels, checked against
Hugging Face Transformers and benchmarked in tokens per second.

Usage (needs an NVIDIA GPU, PyTorch with CUDA, and `transformers`):
    python3 llm/run_gpt2.py --verify --bench --model gpt2 \
        --csv benchmarks/<GPU>/gpt2.csv

What it does:
  1. Builds llm/gpt2_ops.cpp + the kernels it needs as a PyTorch extension.
  2. Loads a GPT-2 checkpoint from Hugging Face and copies its weights into the layout
     our kernels expect, as FP32, FP16 or INT8 (symmetric, one scale per output row,
     the same rule as cpu::quantize_int8).
  3. --verify: feeds the prompt token by token and compares our logits with Hugging Face's
     FP32 logits at every position, then compares greedy generations token by token.
     Exits with an error if our FP32 path differs by more than 1e-3 of the logit scale.
  4. --bench: decode speed (ms per generated token, tokens/s) for Hugging Face FP32 and FP16
     and for our FP32, FP16, INT8 (one row per warp) and INT8 (2 rows per warp) weights.

Scope: batch size 1, greedy decoding, FP32 activations. The prompt is also fed one token
at a time (no separate prefill kernel). Explained in docs/14_llm_integration.md.
"""

import argparse
import csv
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT = "The GPU kernels behind Transformer inference are"


# -----------------------------------------------------------------------------
# Building the extension
# -----------------------------------------------------------------------------
def build_ops(verbose: bool):
    from torch.utils.cpp_extension import load

    sources = [os.path.join(ROOT, "llm", "gpt2_ops.cpp")] + [
        os.path.join(ROOT, "kernels", name) for name in ("gemv.cu", "layernorm.cu", "attention.cu", "softmax.cu")
    ]
    return load(name="ctp_gpt2_ops", sources=sources,
                extra_include_paths=[os.path.join(ROOT, "kernels"), os.path.join(ROOT, "src")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=verbose)


# -----------------------------------------------------------------------------
# The model on our kernels
# -----------------------------------------------------------------------------
EMPTY = None  # set to an empty CUDA tensor in main()


def quantize_rows(w: torch.Tensor):
    """Symmetric INT8, one scale per row: scale = max|w| / 127, q = round(w / scale)."""
    scale = w.abs().amax(dim=1) / 127.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))  # an all-zero row stays 0
    q = torch.round(w / safe[:, None]).clamp_(-127, 127).to(torch.int8)  # round half to even
    return q.contiguous(), scale.float().contiguous()


def convert_linear(w_out_in: torch.Tensor, weights: str):
    """[out, in] float32 weight -> (w, scale) in the requested format."""
    w = w_out_in.float().contiguous()
    if weights == "fp32":
        return w, EMPTY
    if weights == "fp16":
        return w.half().contiguous(), EMPTY
    if weights == "int8":
        return quantize_rows(w)
    raise ValueError(weights)


class CudaGPT2:
    """GPT-2 decoding (batch 1) with every LayerNorm, linear layer and attention on our kernels."""

    def __init__(self, hf_model, ops, weights: str, rows_per_warp: int, max_len: int):
        cfg = hf_model.config
        self.ops, self.rows_per_warp = ops, rows_per_warp
        self.n_layer, self.n_head, self.n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.max_len = min(max_len, cfg.n_positions)
        sd = {k: v.detach().float() for k, v in hf_model.state_dict().items()}
        f = lambda name: sd[name].contiguous()  # noqa: E731

        self.wte, self.wpe = f("transformer.wte.weight"), f("transformer.wpe.weight")
        self.weight_bytes = 0
        self.layers = []
        for i in range(self.n_layer):
            pre = f"transformer.h.{i}."
            params = [f(pre + "ln_1.weight"), f(pre + "ln_1.bias")]
            # HF GPT-2 uses Conv1D with weight [in, out]: transpose to [out, in] = N x K.
            for name in ("attn.c_attn", "attn.c_proj"):
                w, s = convert_linear(sd[pre + name + ".weight"].t(), weights)
                params += [w, s, f(pre + name + ".bias")]
                self.weight_bytes += w.numel() * w.element_size() + (s.numel() * 4 if s is not EMPTY else 0)
            params += [f(pre + "ln_2.weight"), f(pre + "ln_2.bias")]
            for name in ("mlp.c_fc", "mlp.c_proj"):
                w, s = convert_linear(sd[pre + name + ".weight"].t(), weights)
                params += [w, s, f(pre + name + ".bias")]
                self.weight_bytes += w.numel() * w.element_size() + (s.numel() * 4 if s is not EMPTY else 0)
            self.layers.append(params)
        self.lnf_w, self.lnf_b = f("transformer.ln_f.weight"), f("transformer.ln_f.bias")
        # LM head = the token embedding (tied weights), already [vocab, n_embd] = N x K.
        self.head_w, self.head_s = convert_linear(self.wte, weights)
        self.weight_bytes += self.head_w.numel() * self.head_w.element_size() + (
            self.head_s.numel() * 4 if self.head_s is not EMPTY else 0)

        d = self.n_embd // self.n_head
        shape = (self.n_layer, self.n_head, self.max_len, d)
        self.k_cache = torch.zeros(shape, device=self.wte.device)
        self.v_cache = torch.zeros(shape, device=self.wte.device)

    def step(self, token, pos: int) -> torch.Tensor:
        """Logits [vocab] for `token` (int or 0-d CUDA tensor) at position `pos`."""
        if pos >= self.max_len:
            raise ValueError(f"position {pos} exceeds the KV cache capacity {self.max_len}")
        x = (self.wte[token] + self.wpe[pos]).contiguous()
        for i, params in enumerate(self.layers):
            x = self.ops.layer_decode(x, params, self.k_cache[i], self.v_cache[i], pos, self.n_head,
                                      self.rows_per_warp)
        x = self.ops.layernorm(x, self.lnf_w, self.lnf_b)
        return self.ops.linear(self.head_w, self.head_s, EMPTY, x, self.rows_per_warp)


# -----------------------------------------------------------------------------
# Reference (Hugging Face) decoding with its KV cache
# -----------------------------------------------------------------------------
def hf_decode_steps(model, ids: torch.Tensor, new_tokens: int):
    """Greedy: prompt in one forward pass, then one token per step. Returns (generated, per-step logits)."""
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        generated, all_logits = [], []
        for _ in range(new_tokens):
            tok = logits.argmax()
            generated.append(tok)
            all_logits.append(logits)
            out = model(tok.view(1, 1), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
    return generated, all_logits


def ours_decode_steps(m: CudaGPT2, ids: torch.Tensor, new_tokens: int):
    with torch.no_grad():
        prompt = ids[0].tolist()
        for pos, tok in enumerate(prompt):
            logits = m.step(tok, pos)
        generated, all_logits = [], []
        pos = len(prompt)
        for _ in range(new_tokens):
            tok = logits.argmax()
            generated.append(tok)
            all_logits.append(logits)
            logits = m.step(tok, pos)
            pos += 1
    return generated, all_logits


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
def verify(hf_model, ops, ids, tok, args) -> bool:
    print("\n== Verification against Hugging Face (FP32) ==")
    with torch.no_grad():
        ref_prompt_logits = hf_model(ids).logits[0]  # [T, vocab], all prompt positions at once
    ref_tokens, _ = hf_decode_steps(hf_model, ids, args.verify_tokens)
    ref_tokens = [int(t) for t in ref_tokens]
    scale = ref_prompt_logits.abs().max().item()
    ok = True
    for weights, rows in (("fp32", 2), ("fp16", 2), ("int8", 2)):
        m = CudaGPT2(hf_model, ops, weights, rows, args.max_len)
        with torch.no_grad():
            ours = torch.stack([m.step(t, p) for p, t in enumerate(ids[0].tolist())])
        diff = (ours - ref_prompt_logits).abs().max().item() / scale
        same_argmax = (ours.argmax(dim=1) == ref_prompt_logits.argmax(dim=1)).float().mean().item()
        gen, _ = ours_decode_steps(m, ids, args.verify_tokens)
        gen = [int(t) for t in gen]
        match = next((i for i, (a, b) in enumerate(zip(gen, ref_tokens)) if a != b), len(gen))
        print(f"  {weights:5s} weights: prompt logits max |diff| = {diff:.2e} of max |logit|, "
              f"next-token argmax agrees at {100 * same_argmax:.0f}% of prompt positions; "
              f"greedy tokens identical for {match}/{len(gen)}")
        if weights == "fp32":
            print(f"        text: {tok.decode(ids[0].tolist() + gen)!r}")
            if diff > 1e-3:
                ok = False
                print("  FAIL: FP32 logits differ from Hugging Face by more than 1e-3 of the logit scale")
        del m
        torch.cuda.empty_cache()
    return ok


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
def time_decode(step_fn, n_tokens: int) -> float:
    """Average wall-clock ms per call of step_fn() over n_tokens calls."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n_tokens


def bench(hf_model, ops, ids, args, gpu_name):
    print(f"\n== Decode speed: batch 1, {args.bench_tokens} generated tokens after a "
          f"{ids.shape[1]}-token prompt (wall clock, includes Python and launch overhead) ==")
    rows = []
    T = ids.shape[1]

    def bench_hf(model, label, weight_bytes):
        with torch.no_grad():
            state = {}

            def reset():
                out = model(ids, use_cache=True)
                state["past"], state["tok"] = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)

            def step():
                out = model(state["tok"], past_key_values=state["past"], use_cache=True)
                state["past"], state["tok"] = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)

            reset()
            for _ in range(args.warmup):  # warm-up: clocks, caches, first-call costs
                step()
            reset()  # time from the same position every time
            ms = time_decode(step, args.bench_tokens)
        rows.append((label, ms, weight_bytes))

    def bench_ours(weights, rows_per_warp, label):
        m = CudaGPT2(hf_model, ops, weights, rows_per_warp, args.max_len)
        with torch.no_grad():
            state = {}

            def reset():
                for p, t in enumerate(ids[0].tolist()):
                    logits = m.step(t, p)
                state["pos"], state["tok"] = T, logits.argmax()

            def step():
                logits = m.step(state["tok"], state["pos"])
                state["pos"] += 1
                state["tok"] = logits.argmax()

            reset()
            for _ in range(args.warmup):  # warm-up: clocks, caches, first-call costs
                step()
            reset()  # time from the same position every time
            ms = time_decode(step, args.bench_tokens)
        rows.append((label, ms, m.weight_bytes))
        m = None  # release this format's weights before building the next
        torch.cuda.empty_cache()

    n_params = sum(p.numel() for p in hf_model.parameters())
    bench_hf(hf_model, "Hugging Face, FP32", 4 * n_params)
    bench_ours("fp32", 2, "ours, FP32 weights")
    bench_ours("fp16", 2, "ours, FP16 weights")
    bench_ours("int8", 0, "ours, INT8 weights, 1 row/warp")
    bench_ours("int8", 2, "ours, INT8 weights, 2 rows/warp")
    hf_model.half()  # last: converting back to FP32 would not restore the original weights
    bench_hf(hf_model, "Hugging Face, FP16", 2 * n_params)

    base = rows[0][1]
    print(f"  {'implementation':34s} | {'weights MB':>10s} | {'ms/token':>9s} | {'tokens/s':>9s} | {'vs HF FP32':>10s}")
    for label, ms, wb in rows:
        print(f"  {label:34s} | {wb / 1e6:10.1f} | {ms:9.3f} | {1e3 / ms:9.1f} | {base / ms:9.2f}x")
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gpu", "model", "impl", "weight_mb", "ms_per_token", "tokens_per_s", "prompt_tokens",
                        "generated_tokens"])
            for label, ms, wb in rows:
                w.writerow([gpu_name, args.model, label, f"{wb / 1e6:.1f}", f"{ms:.4f}", f"{1e3 / ms:.2f}", T,
                            args.bench_tokens])
        print(f"Wrote {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 on this project's CUDA kernels")
    parser.add_argument("--model", default="gpt2", help="gpt2, gpt2-medium, gpt2-large or gpt2-xl")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--verify-tokens", type=int, default=32)
    parser.add_argument("--bench-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--csv", default="")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if not (args.verify or args.bench):
        parser.error("choose --verify and/or --bench")
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU available.")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.backends.cuda.matmul.allow_tf32 = False  # true FP32 reference, like the rest of the project
    torch.backends.cudnn.allow_tf32 = False
    global EMPTY
    EMPTY = torch.empty(0, device="cuda")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}   model: {args.model}   torch {torch.__version__}")
    print("Building the extension (first time: about a minute)...")
    ops = build_ops(args.verbose_build)

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    hf_model = GPT2LMHeadModel.from_pretrained(args.model).cuda().eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    needed = ids.shape[1] + max(args.verify_tokens, args.bench_tokens + args.warmup) + 1
    if needed > min(args.max_len, hf_model.config.n_positions):
        sys.exit(f"prompt + generated tokens ({needed}) exceed the context length")

    ok = True
    if args.verify:
        ok = verify(hf_model, ops, ids, tok, args)
    if args.bench:
        bench(hf_model, ops, ids, args, gpu_name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
