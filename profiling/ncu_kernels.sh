#!/usr/bin/env bash
# =============================================================================
# ncu_kernels.sh — Nsight Compute: detailed per-kernel analysis.
# docs/10_nsight_profiling.md sections 5-7.
#
#   bash profiling/ncu_kernels.sh <case>
#   case: vector_add | gemm | precision | softmax | layernorm | attention
#
# Output (profiling/reports/<GPU>/):
#   ncu_<case>.ncu-rep        full report (open in the Nsight Compute GUI on your PC)
#   ncu_<case>_details.txt    the same report as text: every section, every kernel
#   ncu_<case>_metrics.csv    a fixed list of key metrics, one row per kernel x metric
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
# One folder per GPU, like benchmarks/ (e.g. profiling/reports/Tesla_T4), so runs on
# different GPUs never overwrite each other.
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed "s/[^A-Za-z0-9]/_/g")
OUT="profiling/reports/${GPU_TAG:-unknown_GPU}"
mkdir -p "$OUT"
CASE="${1:?usage: bash profiling/ncu_kernels.sh <case>}"

if ! command -v ncu > /dev/null; then
    echo "ncu (Nsight Compute) not found on PATH. See docs/10_nsight_profiling.md section 2." >&2
    exit 1
fi

# ---- Run 1: report sections (each section = a group of related metrics + ncu's own analysis)
#   SpeedOfLight                 how close to peak compute / memory throughput
#   LaunchStats                  grid, block, registers per thread, shared memory per block, waves
#   Occupancy                    theoretical vs achieved occupancy, and what limits it
#   MemoryWorkloadAnalysis(_Tables) L1/L2/DRAM traffic, hit rates, sectors per request, bank conflicts
#   ComputeWorkloadAnalysis      utilization of each pipeline (FMA, ALU, LSU, Tensor, ...)
#   WarpStateStats               WHY warps are stalled (memory, barrier, shared memory, ...)
#   SchedulerStats               how often each scheduler had a warp ready to issue
#   SourceCounters               per-source-line hot spots, uncoalesced accesses (needs -lineinfo)
# Clocks: ncu locks the GPU to its BASE clock by default (--clock-control base) so that
# results are repeatable. Durations are therefore longer than in the benchmarks;
# percentages of peak are still meaningful.
ncu --force-overwrite \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    --section ComputeWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
    --section SourceCounters \
    -o "$OUT/ncu_${CASE}" \
    ./build/profile_targets "$CASE"

ncu --import "$OUT/ncu_${CASE}.ncu-rep" --page details \
    > "$OUT/ncu_${CASE}_details.txt"

# ---- Run 2: a fixed list of metrics we quote in docs/10, as CSV.
# Metric names follow   <unit>__<counter>.<rollup>  e.g. dram__bytes_read.sum = total bytes read from DRAM.
METRICS=(
    gpu__time_duration.sum                                   # kernel duration
    sm__cycles_elapsed.avg.per_second                        # actual SM clock during the kernel
    sm__throughput.avg.pct_of_peak_sustained_elapsed         # compute "speed of light" %
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed  # memory "speed of light" %
    dram__throughput.avg.pct_of_peak_sustained_elapsed       # DRAM bandwidth used, % of peak
    dram__bytes_read.sum
    dram__bytes_write.sum
    lts__t_sector_hit_rate.pct                               # L2 hit rate
    l1tex__t_sector_hit_rate.pct                             # L1 hit rate
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum          # global load instructions (warp-level)
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum           # 32-byte sectors they touched
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum # shared-memory load bank conflicts
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum # shared-memory store bank conflicts
    sm__warps_active.avg.pct_of_peak_sustained_active        # ACHIEVED occupancy
    launch__registers_per_thread
    launch__occupancy_limit_registers
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_blocks
    launch__waves_per_multiprocessor
)
# Strip the comments: keep only the metric name of each line above.
METRIC_LIST=$(printf "%s\n" "${METRICS[@]}" | awk '{print $1}' | paste -sd, -)
# --log-file sends ncu's CSV to the file, while the program's own launch list
# stays on the terminal (otherwise both would be mixed into one stream).
ncu --csv --log-file "$OUT/ncu_${CASE}_metrics.csv" --metrics "$METRIC_LIST" \
    ./build/profile_targets "$CASE" \
    || echo "WARNING: ncu reported an error (e.g. a metric not available on this GPU); see the CSV file." >&2

echo
echo "Saved: $OUT/ncu_${CASE}.ncu-rep, ncu_${CASE}_details.txt, ncu_${CASE}_metrics.csv"
