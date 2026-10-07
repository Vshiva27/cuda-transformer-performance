"""
benchmark.py — benchmark every PyTorch baseline on the current GPU.

For each operation and shape it:
  1. checks correctness against a float64 CPU reference (prints max abs error),
  2. times the GPU (CUDA events) and the full Python call (wall clock),
  3. times PyTorch on the CPU, for shapes where that is affordable,
  4. measures the extra GPU memory one call needs,
  5. derives GFLOP/s and/or GB/s,
and finally writes everything to benchmarks/pytorch_<gpu>.csv.

The shapes are the SAME as in the C++ benchmarks (src/benchmark/*.cu), so
rows can be compared one to one in docs/12_results.md.

Usage (from the project root):
    python python/benchmark.py                 # all operations
    python python/benchmark.py --ops matmul softmax
"""

import argparse
import csv
import os
import re
import time

import torch

import pytorch_baseline as pb

# -----------------------------------------------------------------------------
# Shapes — keep in sync with the C++ benchmarks.
# -----------------------------------------------------------------------------
VECTOR_SIZES = [2**10, 2**14, 2**18, 2**20, 2**22, 2**24, 2**26]
MATMUL_SQUARE = [32, 64, 128, 256, 512, 1024, 2048, 4096]  # 4096 fills large GPUs (A100/H100)
MATMUL_TRANSFORMER = [  # GPT-2 small: hidden 768, MLP 3072; M = tokens
    ("QKV proj, prefill 512", 512, 3 * 768, 768),
    ("MLP up, prefill 512", 512, 3072, 768),
    ("MLP down, prefill 512", 512, 768, 3072),
    ("QKV proj, decode 1", 1, 3 * 768, 768),
    ("MLP up, decode 1", 1, 3072, 768),
]
SOFTMAX_SHAPES = [(1024, 128), (4096, 512), (4096, 1024), (12 * 1024, 1024)]  # (rows, cols)
LAYERNORM_SHAPES = [(512, 768), (2048, 768), (4096, 1024), (8192, 4096)]      # (tokens, hidden)
ATTENTION_SHAPES = [(12, 128, 64), (12, 512, 64), (12, 1024, 64), (12, 2048, 64)]  # (heads, seq, d)

ALL_OPS = ["vector_add", "matmul", "softmax", "layernorm", "attention"]
FP32_BYTES = 4


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def choose_iters(fn) -> int:
    """Pick an iteration count so one measurement takes roughly 0.2 s."""
    fn()  # the first call may include one-time setup; do not time it
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one_call_s = max(time.perf_counter() - t0, 1e-6)
    return int(min(200, max(5, 0.2 / one_call_s)))


def fits_in_gpu(bytes_needed: int, info: dict) -> bool:
    """Use at most half of the free GPU memory (same rule as the C++ benchmarks)."""
    return bytes_needed <= info["free_mem_bytes"] // 2


def make_row(op, impl, shape, dtype, timing, cpu_ms=None, flops=None, bytes_moved=None,
             mem_mb=None, err=None) -> dict:
    """One result row. GFLOP/s and GB/s are derived from the GPU (event) time."""
    gpu_ms = timing.gpu_ms
    return {
        "op": op,
        "impl": impl,
        "shape": shape,
        "dtype": dtype,
        "gpu_ms": gpu_ms,
        "wall_ms": timing.wall_ms,
        "cpu_ms": cpu_ms,
        "gflops": flops / (gpu_ms * 1e-3) / 1e9 if flops else None,
        "gbs": bytes_moved / (gpu_ms * 1e-3) / 1e9 if bytes_moved else None,
        "extra_mem_mb": mem_mb,
        "max_abs_err": err,
    }


def fmt(value, spec: str) -> str:
    """Format a number, or print '-' (with the same width) when it was not measured."""
    width = int(spec.split(".")[0])
    return "-".rjust(width) if value is None else format(value, spec)


def print_rows(title: str, rows: list) -> None:
    print(f"\n{title}")
    print(f"  {'impl':<22} {'shape':<34} {'dtype':<6} {'GPU ms':>9} {'wall ms':>9} {'CPU ms':>10} "
          f"{'GFLOP/s':>9} {'GB/s':>8} {'mem MB':>8} {'max err':>9}")
    for r in rows:
        print(f"  {r['impl']:<22} {r['shape']:<34} {r['dtype']:<6} {fmt(r['gpu_ms'], '9.4f')} "
              f"{fmt(r['wall_ms'], '9.4f')} {fmt(r['cpu_ms'], '10.3f')} {fmt(r['gflops'], '9.1f')} "
              f"{fmt(r['gbs'], '8.1f')} {fmt(r['extra_mem_mb'], '8.1f')} {fmt(r['max_abs_err'], '9.2e')}")


# -----------------------------------------------------------------------------
# One function per operation
# -----------------------------------------------------------------------------
def bench_vector_add(info: dict) -> list:
    rows = []
    for n in VECTOR_SIZES:
        if not fits_in_gpu(3 * n * FP32_BYTES, info):
            print(f"  vector_add n={n}: skipped (GPU memory)")
            continue
        a_cpu, b_cpu = torch.randn(n), torch.randn(n)
        a, b = a_cpu.cuda(), b_cpu.cuda()

        err = pb.max_abs_error(pb.vector_add(a, b), a_cpu.double() + b_cpu.double())
        fn = lambda: pb.vector_add(a, b)
        timing = pb.time_cuda(fn, iters=choose_iters(fn))
        cpu_ms = pb.time_cpu(lambda: pb.vector_add(a_cpu, b_cpu))
        rows.append(make_row("vector_add", "torch add", f"n={n}", "fp32", timing, cpu_ms,
                             flops=n, bytes_moved=3 * n * FP32_BYTES,
                             mem_mb=pb.peak_extra_memory_mb(fn), err=err))
    print_rows("vector_add  (GB/s = 12 bytes per element / GPU time)", rows)
    return rows


def bench_one_matmul(rows, info, label, M, N, K, with_cpu: bool) -> None:
    if not fits_in_gpu(FP32_BYTES * (M * K + K * N + M * N), info):
        print(f"  matmul {label}: skipped (GPU memory)")
        return
    a_cpu, b_cpu = torch.randn(M, K), torch.randn(K, N)
    a, b = a_cpu.cuda(), b_cpu.cuda()
    # float64 CPU reference, skipped only for the largest problem (2048^3) where it is slow.
    reference = a_cpu.double() @ b_cpu.double() if M * N * K <= 2**31 else None
    flops = 2 * M * N * K
    min_bytes = FP32_BYTES * (M * K + K * N + M * N)
    shape = f"{label} {M}x{N}x{K}"

    # FP32 (TF32 disabled): the apples-to-apples comparison with our kernels.
    fn = lambda: pb.matmul(a, b)
    err = pb.max_abs_error(fn(), reference) if reference is not None else None
    timing = pb.time_cuda(fn, iters=choose_iters(fn))
    cpu_ms = pb.time_cpu(lambda: pb.matmul(a_cpu, b_cpu), iters=3) if with_cpu else None
    rows.append(make_row("matmul", "cuBLAS fp32", shape, "fp32", timing, cpu_ms, flops=flops,
                         bytes_moved=min_bytes, mem_mb=pb.peak_extra_memory_mb(fn), err=err))

    # FP16 inputs, FP32 accumulation (Tensor Cores) — preview of Phase 7.
    # The error includes rounding the INPUTS to FP16, measured against the
    # exact product of the original FP32 inputs.
    a16, b16 = a.half(), b.half()
    fn16 = lambda: pb.matmul(a16, b16)
    err16 = pb.max_abs_error(fn16(), reference) if reference is not None else None
    timing16 = pb.time_cuda(fn16, iters=choose_iters(fn16))
    rows.append(make_row("matmul", "cuBLAS fp16 (fp32 acc)", shape, "fp16", timing16, None, flops=flops,
                         bytes_moved=min_bytes // 2, mem_mb=pb.peak_extra_memory_mb(fn16), err=err16))


def bench_matmul(info: dict) -> list:
    rows = []
    for n in MATMUL_SQUARE:
        bench_one_matmul(rows, info, "square", n, n, n, with_cpu=n <= 1024)
    for label, M, N, K in MATMUL_TRANSFORMER:
        bench_one_matmul(rows, info, label, M, N, K, with_cpu=True)
    print_rows("matmul  (GFLOP/s = 2*M*N*K / GPU time; GB/s = minimum bytes / GPU time)", rows)

    # Precision experiment: TF32 exists on compute capability 8.0+ (Ampere and newer).
    if info["cc_major"] >= 8:
        n = 1024
        a_cpu, b_cpu = torch.randn(n, n), torch.randn(n, n)
        a, b = a_cpu.cuda(), b_cpu.cuda()
        reference = a_cpu.double() @ b_cpu.double()
        tf32_rows = []
        for allow in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = allow
            fn = lambda: pb.matmul(a, b)
            err = pb.max_abs_error(fn(), reference)
            timing = pb.time_cuda(fn, iters=choose_iters(fn))
            tf32_rows.append(make_row("matmul", "cuBLAS tf32" if allow else "cuBLAS fp32", f"square {n}x{n}x{n}",
                                      "tf32" if allow else "fp32", timing, flops=2 * n**3, err=err))
        torch.backends.cuda.matmul.allow_tf32 = False
        print_rows("TF32 experiment (same FP32 inputs; TF32 rounds them to 10 mantissa bits inside Tensor Cores)",
                   tf32_rows)
        rows.extend(tf32_rows)
    return rows


def bench_softmax(info: dict) -> list:
    rows = []
    for r, c in SOFTMAX_SHAPES:
        if not fits_in_gpu(2 * r * c * FP32_BYTES, info):
            continue
        x_cpu = torch.randn(r, c) * 3.0
        x = x_cpu.cuda()
        err = pb.max_abs_error(pb.softmax(x), torch.softmax(x_cpu.double(), dim=-1))
        fn = lambda: pb.softmax(x)
        timing = pb.time_cuda(fn, iters=choose_iters(fn))
        cpu_ms = pb.time_cpu(lambda: pb.softmax(x_cpu))
        rows.append(make_row("softmax", "torch softmax", f"{r}x{c}", "fp32", timing, cpu_ms,
                             bytes_moved=2 * r * c * FP32_BYTES, mem_mb=pb.peak_extra_memory_mb(fn), err=err))
    print_rows("softmax over rows  (GB/s = read x + write y, minimum bytes)", rows)
    return rows


def bench_layernorm(info: dict) -> list:
    rows = []
    for t, h in LAYERNORM_SHAPES:
        if not fits_in_gpu(2 * t * h * FP32_BYTES, info):
            continue
        x_cpu, g_cpu, b_cpu = torch.randn(t, h), torch.randn(h), torch.randn(h)
        x, g, b = x_cpu.cuda(), g_cpu.cuda(), b_cpu.cuda()
        err = pb.max_abs_error(pb.layernorm(x, g, b), pb.layernorm(x_cpu.double(), g_cpu.double(), b_cpu.double()))
        fn = lambda: pb.layernorm(x, g, b)
        timing = pb.time_cuda(fn, iters=choose_iters(fn))
        cpu_ms = pb.time_cpu(lambda: pb.layernorm(x_cpu, g_cpu, b_cpu))
        rows.append(make_row("layernorm", "torch layer_norm", f"{t}x{h}", "fp32", timing, cpu_ms,
                             bytes_moved=2 * t * h * FP32_BYTES, mem_mb=pb.peak_extra_memory_mb(fn), err=err))

        # Residual add + LayerNorm as PyTorch runs it in eager mode: two kernels,
        # h is written to memory and read back. Compare with our fused kernel
        # (bench_layernorm Experiment B). GB/s uses the task minimum: 16 bytes/element.
        r = torch.randn(t, h, device="cuda")
        fn2 = lambda: pb.add_layernorm(x, r, g, b)
        timing2 = pb.time_cuda(fn2, iters=choose_iters(fn2))
        rows.append(make_row("add_layernorm", "torch add + layer_norm", f"{t}x{h}", "fp32", timing2,
                             bytes_moved=4 * t * h * FP32_BYTES, mem_mb=pb.peak_extra_memory_mb(fn2)))
    print_rows("layernorm over hidden dim  (GB/s = minimum bytes: 8/element for LN, 16/element for add+LN)", rows)
    return rows


def bench_attention(info: dict) -> list:
    rows = []
    for heads, seq, d in ATTENTION_SHAPES:
        # The naive version materializes a heads x seq x seq score matrix (twice: scores and probs).
        if not fits_in_gpu(FP32_BYTES * (4 * heads * seq * d + 2 * heads * seq * seq), info):
            continue
        q_cpu, k_cpu, v_cpu = (torch.randn(heads, seq, d) for _ in range(3))
        q, k, v = q_cpu.cuda(), k_cpu.cuda(), v_cpu.cuda()
        reference = (pb.attention_naive(q_cpu.double(), k_cpu.double(), v_cpu.double())
                     if seq <= 1024 else None)
        flops = 4 * heads * seq * seq * d  # QK^T (2*s*s*d) + PV (2*s*s*d) per head
        shape = f"h={heads} seq={seq} d={d}"

        for impl, func in (("naive (3 kernels)", pb.attention_naive), ("SDPA (fused)", pb.attention_sdpa)):
            fn = lambda: func(q, k, v)
            err = pb.max_abs_error(fn(), reference) if reference is not None else None
            timing = pb.time_cuda(fn, iters=choose_iters(fn))
            cpu_ms = pb.time_cpu(lambda: func(q_cpu, k_cpu, v_cpu), iters=2) if seq <= 1024 else None
            rows.append(make_row("attention", impl, shape, "fp32", timing, cpu_ms, flops=flops,
                                 mem_mb=pb.peak_extra_memory_mb(fn), err=err))

        # FP16 SDPA: PyTorch's memory-saving fused attention kernels need FP16/BF16 inputs
        # (in the measured T4 run, FP32 SDPA saved no memory). The error includes rounding
        # Q, K, V to FP16.
        q16, k16, v16 = q.half(), k.half(), v.half()
        fn16 = lambda: pb.attention_sdpa(q16, k16, v16)
        err16 = pb.max_abs_error(fn16(), reference) if reference is not None else None
        timing16 = pb.time_cuda(fn16, iters=choose_iters(fn16))
        rows.append(make_row("attention", "SDPA (fused) fp16", shape, "fp16", timing16, flops=flops,
                             mem_mb=pb.peak_extra_memory_mb(fn16), err=err16))
    print_rows("attention softmax(QK^T/sqrt(d))V  (GFLOP/s counts the two matmuls; mem = extra GPU memory per call)",
               rows)
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch baseline benchmarks")
    parser.add_argument("--ops", nargs="+", choices=ALL_OPS, default=ALL_OPS,
                        help="operations to benchmark (default: all)")
    parser.add_argument("--csv", default=None, help="output CSV path (default: benchmarks/pytorch_<gpu>.csv)")
    args = parser.parse_args()

    info = pb.gpu_info()
    pb.configure_for_fair_comparison()
    torch.manual_seed(0)

    print("=================== GPU (PyTorch view) ===================")
    print(f"Device              : {info['name']}")
    print(f"Compute capability  : {info['compute_capability']}")
    print(f"SMs                 : {info['sm_count']}")
    print(f"Memory              : {info['total_mem_gib']:.2f} GiB total, {info['free_mem_bytes'] / 2**30:.2f} GiB free")
    print(f"PyTorch / CUDA      : {info['torch_version']} / {info['torch_cuda_version']}")
    print(f"CPU threads (torch) : {info['cpu_threads']}  (PyTorch CPU ops are multi-threaded;"
          f" our C++ CPU reference is single-threaded)")
    print("TF32                : disabled (true FP32, like our kernels)")
    print("===========================================================")

    benches = {
        "vector_add": bench_vector_add,
        "matmul": bench_matmul,
        "softmax": bench_softmax,
        "layernorm": bench_layernorm,
        "attention": bench_attention,
    }
    all_rows = []
    # inference_mode: no autograd bookkeeping, exactly like real inference.
    with torch.inference_mode():
        for op in args.ops:
            all_rows.extend(benches[op](info))

    if not all_rows:
        print("\nNo results (all shapes skipped).")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gpu_tag = re.sub(r"[^A-Za-z0-9]+", "_", info["name"]).strip("_")
    csv_path = args.csv or os.path.join(project_root, "benchmarks", f"pytorch_{gpu_tag}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gpu"] + list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({"gpu": info["name"], **row})
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
