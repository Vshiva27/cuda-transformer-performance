#!/usr/bin/env bash
# =============================================================================
# run_all.sh — build, test, benchmark, and summarize, in one command.
#
#   bash scripts/run_all.sh
#
# Results go to benchmarks/<GPU name>/:
#   environment.txt          GPU, driver, CUDA and compiler versions
#   tests.txt                correctness test output
#   <benchmark>.txt / .csv   printed tables and machine-readable rows
#   pytorch.txt / .csv       PyTorch baseline
#   summary.md               all tables + headline numbers (python/summarize.py)
#
# The script STOPS if the build or any correctness test fails: benchmark
# numbers of an incorrect kernel are worthless. Explained in
# docs/09_benchmarking.md, Part 2.
# =============================================================================

# -e: stop at the first failing command; -u: error on unset variables;
# -o pipefail: a pipeline fails if ANY command in it fails (not only the last),
# so "ctest | tee" still stops the script when ctest fails.
set -euo pipefail

cd "$(dirname "$0")/.."   # run from the project root, wherever the script is called from

if ! command -v nvidia-smi > /dev/null; then
    echo "nvidia-smi not found: this machine has no NVIDIA GPU driver." >&2
    echo "On Colab: Runtime -> Change runtime type -> GPU." >&2
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_TAG=$(echo "$GPU_NAME" | sed 's/[^A-Za-z0-9]/_/g')
OUT="benchmarks/$GPU_TAG"
mkdir -p "$OUT"
echo "GPU: $GPU_NAME  ->  results in $OUT"

# ---- 1. Record the environment ------------------------------------------------
{
    echo "date: $(date -u '+%Y-%m-%d %H:%M UTC')"
    nvidia-smi
    nvcc --version || true
    cmake --version | head -n 1
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || true
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw --format=csv || true
    echo "CPU (for the CPU baselines):"
    lscpu | grep -E "Model name|^CPU\(s\)|Thread" || true
} > "$OUT/environment.txt" 2>&1

# ---- 2. Build -----------------------------------------------------------------------
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

# ---- 3. Correctness first -----------------------------------------------------------
ctest --test-dir build --output-on-failure 2>&1 | tee "$OUT/tests.txt"

# ---- 4. Benchmarks -------------------------------------------------------------------
for bench in vector_add matmul softmax layernorm precision attention; do
    echo
    echo "================ bench_$bench ================"
    "./build/bench_$bench" --csv "$OUT/$bench.csv" 2>&1 | tee "$OUT/$bench.txt"
done

# ---- 5. PyTorch baseline --------------------------------------------------------------
echo
echo "================ PyTorch baseline ================"
python3 python/benchmark.py --csv "$OUT/pytorch.csv" 2>&1 | tee "$OUT/pytorch.txt"

# ---- 6. Summary ----------------------------------------------------------------------------
python3 python/summarize.py "$OUT"
echo
echo "Done. Open $OUT/summary.md"
