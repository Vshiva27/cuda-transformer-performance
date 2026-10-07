"""
pytorch_baseline.py — PyTorch versions of every operation in this project,
plus the timing and measurement helpers used by benchmark.py.

Why this file exists
--------------------
PyTorch is what most people use for inference. Behind each call below sits a
professionally tuned GPU kernel (cuBLAS for matmul, PyTorch's own CUDA kernels
for softmax / layer_norm, fused attention kernels for SDPA). Comparing our
hand-written kernels against these tells us how far from "production quality"
we are, and why.

Nothing here is used INSIDE our CUDA implementations. It is only a reference.
Explained in docs/09_benchmarking.md.
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# =============================================================================
# 1. Device information (never assume a specific GPU)
# =============================================================================
def gpu_info() -> dict:
    """Return a dict describing the GPU PyTorch will use."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible to PyTorch. "
            "On Colab: Runtime -> Change runtime type -> GPU."
        )
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "cc_major": props.major,
        "sm_count": props.multi_processor_count,
        "total_mem_gib": total_bytes / 2**30,
        "free_mem_bytes": free_bytes,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }


def configure_for_fair_comparison() -> None:
    """Make PyTorch compute exactly what our kernels compute.

    TF32 ("TensorFloat-32") is a reduced-precision mode on Ampere and newer
    GPUs: FP32 inputs are rounded to ~10 mantissa bits inside Tensor Cores.
    It is much faster but less precise. Our FP32 kernels do true FP32 math, so
    we switch TF32 off; otherwise we would compare different arithmetic.
    (benchmark.py measures TF32 separately, as a precision experiment.)

    For FP16 matmul we also forbid "reduced precision reduction", so cuBLAS
    accumulates FP16 products in FP32 — the same rule our FP16 kernel will use
    in Phase 7.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


# =============================================================================
# 2. Timing
# =============================================================================
@dataclass
class Timing:
    gpu_ms: float   # time between two CUDA events around the loop, per call
    wall_ms: float  # wall-clock time of the loop (Python + launch + GPU), per call


def time_cuda(fn, warmup: int = 5, iters: int = 50) -> Timing:
    """Average time of one call to fn(), which must launch GPU work.

    Same method as time_gpu_ms() in src/benchmark/bench_utils.cuh:
      1. warm-up calls (first calls pay one-time costs), then synchronize;
      2. record a start event, call fn() `iters` times, record a stop event;
      3. wait for the stop event, read the GPU time between the events.
    We also measure wall-clock time of the same loop. When the GPU work is
    tiny, both numbers are dominated by CPU-side overhead (Python + PyTorch
    dispatch + kernel launch), not by the kernel itself — see docs/09.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    stop.synchronize()  # CPU waits until the GPU has reached the stop event
    t1 = time.perf_counter()

    return Timing(gpu_ms=start.elapsed_time(stop) / iters, wall_ms=(t1 - t0) * 1e3 / iters)


def time_cpu(fn, warmup: int = 1, iters: int = 5) -> float:
    """Average wall-clock time (ms) of fn() running on the CPU."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def peak_extra_memory_mb(fn) -> float:
    """GPU memory (MB) that one call to fn() needs on top of what already exists.

    PyTorch keeps freed GPU memory in a "caching allocator" instead of
    returning it to CUDA, so nvidia-smi is useless for this. We ask the
    allocator directly: peak allocated bytes during the call, minus the bytes
    allocated before it. This includes the output and all temporary tensors
    (for naive attention: the full seq x seq score matrix).
    """
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - before) / 2**20


def max_abs_error(result: torch.Tensor, reference: torch.Tensor) -> float:
    """Largest |result - reference|, computed in float64 on the CPU."""
    return (result.double().cpu() - reference.double().cpu()).abs().max().item()


# =============================================================================
# 3. The operations (one line each — that is the point of a framework)
# =============================================================================
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c = a + b (the residual connection)."""
    return a + b


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. On a GPU this calls cuBLAS."""
    return a @ b


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension (each row sums to 1)."""
    return torch.softmax(x, dim=-1)


def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta."""
    return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)


def add_layernorm(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                  eps: float = 1e-5):
    """h = x + residual; y = LayerNorm(h). Returns (h, y). Eager PyTorch runs
    this as two kernels; our CUDA version fuses them into one."""
    h = x + residual
    return h, F.layer_norm(h, (h.shape[-1],), gamma, beta, eps)


def attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d)) V, written as three separate steps.

    Shapes: q, k, v = (heads, seq, d). The intermediate `scores` is
    (heads, seq, seq) — it grows with seq^2 and is written to and read from
    global memory. This is what our CUDA attention (Phase 8) will do first.
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch's fused attention (scaled_dot_product_attention).

    Depending on dtype and GPU it uses a fused kernel (FlashAttention or
    "memory-efficient" attention) that never stores the full seq x seq score
    matrix in global memory — compare its memory column with attention_naive.
    """
    return F.scaled_dot_product_attention(q, k, v)
