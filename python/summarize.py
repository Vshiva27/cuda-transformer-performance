"""
summarize.py — turn the CSV files of one benchmark run into a Markdown report.

Usage:
    python python/summarize.py benchmarks/<GPU_TAG>

Reads every C++ benchmark CSV (written with --csv, see src/benchmark/csv_log.h)
and the PyTorch CSV (python/benchmark.py), and writes <dir>/summary.md with:
  1. headline numbers (the ones docs/12_results.md and the resume bullets need),
     each computed from measured rows — if a row is missing it says "n/a",
     it never guesses;
  2. every measurement, grouped by benchmark and experiment.
Explained in docs/09_benchmarking.md, Part 2.
"""

import csv
import os
import sys
from collections import OrderedDict

CPP_BENCHMARKS = ["vector_add", "matmul", "softmax", "layernorm", "precision", "attention"]


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------
def read_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    """CSV cell -> float, or None if empty / not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find(rows: list, **criteria):
    """First row whose columns equal all the given values, or None."""
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def metric(row, column):
    return number(row.get(column)) if row else None


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def fmt(value, digits=2) -> str:
    v = number(value)
    if v is None:
        return "-"
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
        return f"{v:.3g}"
    return f"{v:.{digits}f}"


def ratio(a, b) -> str:
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{a / b:.2f}x"


def table(header: list, rows: list) -> list:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


# -----------------------------------------------------------------------------
# Headline numbers
# -----------------------------------------------------------------------------
def headlines(data: dict, torch_rows: list) -> list:
    out = []

    def line(text):
        out.append(f"- {text}")

    # GEMM, n = 1024
    mm = data.get("matmul", [])
    shape = "square 1024x1024x1024"
    g = {name: metric(find(mm, experiment="A square", impl=name, shape=shape), "gflops")
         for name in ["cpu (1 thread)", "v1 naive", "v2 coalesced", "v3 tiled-32", "v4 register 4x4"]}
    cublas = metric(find(torch_rows, op="matmul", impl="cuBLAS fp32", shape=shape), "gflops")
    cublas16 = metric(find(torch_rows, op="matmul", impl="cuBLAS fp16 (fp32 acc)", shape=shape), "gflops")
    line(f"**GEMM 1024³ FP32 (GFLOP/s):** CPU {fmt(g['cpu (1 thread)'], 1)} → v1 {fmt(g['v1 naive'], 1)} → "
         f"v2 {fmt(g['v2 coalesced'], 1)} → v3 {fmt(g['v3 tiled-32'], 1)} → v4 {fmt(g['v4 register 4x4'], 1)}; "
         f"cuBLAS {fmt(cublas, 1)}")
    line(f"GEMM v4 vs v1: {ratio(g['v4 register 4x4'], g['v1 naive'])}; v4 vs single-thread CPU: "
         f"{ratio(g['v4 register 4x4'], g['cpu (1 thread)'])}; v4 as a fraction of cuBLAS FP32: "
         f"{ratio(g['v4 register 4x4'], cublas)}")

    pr = data.get("precision", [])
    v4p = metric(find(pr, experiment="A speed", impl="fp32 v4 register", shape=shape), "gflops")
    wmma = metric(find(pr, experiment="A speed", impl="v5 WMMA, fp32 acc", shape=shape), "gflops")
    line(f"**Tensor Cores (1024³):** v5 WMMA {fmt(wmma, 1)} GFLOP/s vs FP32 v4 {fmt(v4p, 1)} → {ratio(wmma, v4p)}; "
         f"cuBLAS FP16 {fmt(cublas16, 1)} GFLOP/s")
    acc16 = find(pr, experiment="B accuracy vs K", impl="fp16 tiled, fp16 acc", shape="64x64x16384")
    acc32 = find(pr, experiment="B accuracy vs K", impl="fp16 tiled, fp32 acc", shape="64x64x16384")
    line(f"**Accumulation precision (K = 16384):** FP16 accumulator {acc16['note'] if acc16 else 'n/a'}; "
         f"FP32 accumulator {acc32['note'] if acc32 else 'n/a'}")
    ovf = find(pr, experiment="C overflow", impl="fp16 tiled, fp16 acc")
    line(f"**FP16 accumulator overflow (inputs in [0,8], K = 8192):** {ovf['note'] if ovf else 'n/a'}")

    # Vector add
    va = [r for r in data.get("vector_add", []) if r["experiment"] == "A sizes" and r["impl"] == "naive"]
    if va:
        last = va[-1]
        line(f"**Vector add {last['shape']}:** {fmt(last['gbs'], 1)} GB/s achieved (memory-bound; compare with peak "
             f"bandwidth in environment/bench output)")

    # Softmax
    sm = data.get("softmax", [])
    s_shape = "4096x1024"
    s = {name: metric(find(sm, experiment="A shapes", impl=name, shape=s_shape), "gbs")
         for name in ["v1 thread/row", "v2 block/row", "v3 warp/row", "v4 warp online"]}
    torch_sm = metric(find(torch_rows, op="softmax", shape=s_shape), "gbs")
    line(f"**Softmax {s_shape} (GB/s):** v1 {fmt(s['v1 thread/row'], 1)}, v2 {fmt(s['v2 block/row'], 1)}, "
         f"v3 {fmt(s['v3 warp/row'], 1)}, v4 {fmt(s['v4 warp online'], 1)}; PyTorch {fmt(torch_sm, 1)} "
         f"→ v4 vs v1 {ratio(s['v4 warp online'], s['v1 thread/row'])}")

    # LayerNorm and fusion
    ln = data.get("layernorm", [])
    l_shape = "4096x1024"
    l1 = metric(find(ln, experiment="A shapes", impl="v1 thread/row", shape=l_shape), "gbs")
    l3 = metric(find(ln, experiment="A shapes", impl="v3 block/row regs", shape=l_shape), "gbs")
    torch_ln = metric(find(torch_rows, op="layernorm", shape=l_shape), "gbs")
    line(f"**LayerNorm {l_shape} (GB/s):** v1 {fmt(l1, 1)}, v3 {fmt(l3, 1)}, PyTorch {fmt(torch_ln, 1)} "
         f"→ v3 vs v1 {ratio(l3, l1)}")
    unf = metric(find(ln, experiment="B fusion", impl="unfused (vector_add + layernorm v3)", shape=l_shape), "ms")
    fus = metric(find(ln, experiment="B fusion", impl="fused add_layernorm", shape=l_shape), "ms")
    torch_fuse = metric(find(torch_rows, op="add_layernorm", shape=l_shape), "gpu_ms")
    line(f"**Fused residual add + LayerNorm {l_shape}:** unfused {fmt(unf, 4)} ms, fused {fmt(fus, 4)} ms "
         f"→ {ratio(unf, fus)}; PyTorch add + layer_norm {fmt(torch_fuse, 4)} ms")

    # Attention
    at = data.get("attention", [])
    for seq in (512, 2048):
        a_shape = f"h=12 seq={seq} d=64"
        unfused = find(at, experiment="A prefill", impl="unfused (3 kernels)", shape=a_shape)
        fused = find(at, experiment="A prefill", impl="fused (1 kernel)", shape=a_shape)
        t_naive = find(torch_rows, op="attention", impl="naive (3 kernels)", shape=a_shape)
        t_sdpa = find(torch_rows, op="attention", impl="SDPA (fused)", shape=a_shape)
        line(f"**Attention {a_shape} (ms):** ours unfused {fmt(metric(unfused, 'ms'), 3)} "
             f"({unfused['note'] if unfused else 'n/a'}), ours fused {fmt(metric(fused, 'ms'), 3)}; "
             f"PyTorch naive {fmt(metric(t_naive, 'gpu_ms'), 3)} ({fmt(metric(t_naive, 'extra_mem_mb'), 1)} MB extra), "
             f"SDPA {fmt(metric(t_sdpa, 'gpu_ms'), 3)} ({fmt(metric(t_sdpa, 'extra_mem_mb'), 1)} MB extra)")
    for ctx in (512, 2048):
        cached = metric(find(at, experiment="B kv cache", impl="decode with KV cache (q_len=1)",
                             shape=f"context={ctx}"), "ms")
        full = metric(find(at, experiment="B kv cache", impl="recompute all (q_len=L)", shape=f"context={ctx}"), "ms")
        line(f"**KV cache, context {ctx}:** one decode step {fmt(cached, 4)} ms with cache vs {fmt(full, 4)} ms "
             f"recomputing attention → {ratio(full, cached)}")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python python/summarize.py benchmarks/<GPU_TAG>")
    run_dir = sys.argv[1]

    data = {name: read_rows(os.path.join(run_dir, f"{name}.csv")) for name in CPP_BENCHMARKS}
    torch_rows = read_rows(os.path.join(run_dir, "pytorch.csv"))
    any_rows = next((rows for rows in list(data.values()) + [torch_rows] if rows), None)
    if any_rows is None:
        raise SystemExit(f"no CSV files found in {run_dir}")
    gpu = any_rows[0].get("gpu", "unknown GPU")

    lines = [f"# Benchmark summary — {gpu}", "",
             "Generated by `python/summarize.py` from the CSV files in this folder. "
             "Every number below was measured; missing measurements are shown as n/a.", "",
             "## Headline numbers", ""]
    lines += headlines(data, torch_rows)

    for name in CPP_BENCHMARKS:
        rows = data[name]
        if not rows:
            continue
        lines += ["", f"## {name} (our CUDA kernels)"]
        groups = OrderedDict()
        for r in rows:
            groups.setdefault(r["experiment"], []).append(r)
        for experiment, group in groups.items():
            lines += ["", f"### {experiment}", ""]
            lines += table(["impl", "shape", "dtype", "ms", "GFLOP/s", "GB/s", "note"],
                           [[r["impl"], r["shape"], r["dtype"], fmt(r["ms"], 4), fmt(r["gflops"], 1),
                             fmt(r["gbs"], 1), r["note"]] for r in group])

    if torch_rows:
        lines += ["", "## PyTorch baseline"]
        groups = OrderedDict()
        for r in torch_rows:
            groups.setdefault(r["op"], []).append(r)
        for op, group in groups.items():
            lines += ["", f"### {op}", ""]
            lines += table(["impl", "shape", "dtype", "GPU ms", "wall ms", "CPU ms", "GFLOP/s", "GB/s", "extra MB",
                            "max err"],
                           [[r["impl"], r["shape"], r["dtype"], fmt(r["gpu_ms"], 4), fmt(r["wall_ms"], 4),
                             fmt(r["cpu_ms"], 3), fmt(r["gflops"], 1), fmt(r["gbs"], 1), fmt(r["extra_mem_mb"], 1),
                             fmt(r["max_abs_err"], 3)] for r in group])

    path = os.path.join(run_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
