#!/usr/bin/env bash
# =============================================================================
# resource_usage.sh — registers, shared memory, local memory (spills) and
# stack for EVERY kernel, read from the compiled binary. No profiler and no
# GPU-counter permissions needed. docs/10_nsight_profiling.md section 3.
#
#   bash profiling/resource_usage.sh        (after building)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# One folder per GPU, like benchmarks/ (e.g. profiling/reports/Tesla_T4), so runs on
# different GPUs never overwrite each other.
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed "s/[^A-Za-z0-9]/_/g")
OUT="profiling/reports/${GPU_TAG:-unknown_GPU}"
mkdir -p "$OUT"

LIB=build/libctp_core.a
if [ ! -f "$LIB" ]; then
    echo "Build first: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

# cuobjdump prints one line per kernel:  Function <mangled name>:  REG:.. STACK:.. SHARED:.. LOCAL:..
# c++filt turns mangled C++ names (_Z...) back into readable ones.
cuobjdump --dump-resource-usage "$LIB" | c++filt | tee "$OUT/resource_usage.txt"
echo
echo "Saved to $OUT/resource_usage.txt"
