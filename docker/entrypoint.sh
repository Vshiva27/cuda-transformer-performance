#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — container entry point (see Dockerfile and docs/docker.md).
#
# Warns early if the container cannot see a GPU, then runs the given command
# (default: bash scripts/run_all.sh). Any command works, e.g.:
#   docker run --rm --gpus all cuda-transformer-performance ctest --test-dir build --output-on-failure
#   docker run --rm -it --gpus all cuda-transformer-performance bash
#   docker run --rm --gpus all -p 127.0.0.1:8888:8888 cuda-transformer-performance jupyter
# =============================================================================
set -euo pipefail

if ! nvidia-smi > /dev/null 2>&1; then
    echo "WARNING: no NVIDIA GPU is visible in this container." >&2
    echo "  Start it with --gpus all (needs the NVIDIA Container Toolkit on the host)." >&2
    echo "  Tests, benchmarks and profiling need a GPU; building and chart generation do not." >&2
fi

# Shortcut: Jupyter server for Google Colab's "Connect to a local runtime" (docs/docker.md, part B).
if [ "${1:-}" = "jupyter" ]; then
    shift
    exec jupyter notebook \
        --ip=0.0.0.0 --port="${JUPYTER_PORT:-8888}" --no-browser --allow-root \
        --ServerApp.allow_origin='https://colab.research.google.com' \
        --ServerApp.port_retries=0 \
        --ServerApp.root_dir=/workspace/cuda-transformer-performance \
        "$@"
fi

exec "$@"
