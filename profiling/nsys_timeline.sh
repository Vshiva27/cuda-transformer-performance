#!/usr/bin/env bash
# =============================================================================
# nsys_timeline.sh — Nsight Systems: a timeline of every CUDA API call and
# kernel, plus summary tables. docs/10_nsight_profiling.md section 4.
#
#   bash profiling/nsys_timeline.sh [case]      (default: all)
#
# Output (profiling/reports/<GPU>/):
#   timeline_<case>.nsys-rep        open in the Nsight Systems GUI on your PC
#   timeline_<case>_stats.txt       text tables: kernel times, API times, NVTX ranges
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# One folder per GPU, like benchmarks/ (e.g. profiling/reports/Tesla_T4), so runs on
# different GPUs never overwrite each other.
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed "s/[^A-Za-z0-9]/_/g")
OUT="profiling/reports/${GPU_TAG:-unknown_GPU}"
mkdir -p "$OUT"
CASE="${1:-all}"

if ! command -v nsys > /dev/null; then
    echo "nsys (Nsight Systems) not found on PATH. See docs/10_nsight_profiling.md section 2." >&2
    exit 1
fi

# --trace=cuda,nvtx : record CUDA runtime calls, kernels, memcpys, and our NVTX ranges
# --reps 5          : each target 5 times, so the first-launch cost is visible separately
nsys profile --trace=cuda,nvtx --force-overwrite=true \
     -o "$OUT/timeline_${CASE}" \
     ./build/profile_targets "$CASE" --reps 5

# Summary tables from the report:
#   cuda_gpu_kern_sum : per kernel name -- count, total, average, min, max GPU time
#   cuda_api_sum      : per CUDA API call -- how much CPU time launches/syncs/mallocs take
#   nvtx_sum          : per NVTX range (our labels) -- wall time
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum \
     "$OUT/timeline_${CASE}.nsys-rep" | tee "$OUT/timeline_${CASE}_stats.txt"
