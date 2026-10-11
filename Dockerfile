# syntax=docker/dockerfile:1
# =============================================================================
# Container image for the whole project: CUDA kernels, tests, benchmarks,
# Nsight profiling, the PyTorch baseline, GPT-2 on the kernels, chart generation,
# and a Jupyter server (for Colab "local runtime"). Instructions: docs/docker.md.
#
#   docker build -t cuda-transformer-performance .
#   docker run --rm --gpus all cuda-transformer-performance          # = bash scripts/run_all.sh
#
# The GPU is only needed at RUN time. The build compiles for a fixed list of GPU
# architectures (CUDA_ARCHS) because there is no GPU to detect during `docker build`.
# =============================================================================

# CUDA 12.6 runs on any driver >= 525 (CUDA minor-version compatibility) and on Colab.
# Every CUDA 12.x / 13.x devel tag works; keep TORCH_INDEX_URL on the same CUDA major.
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04

# Node.js for assets/make_charts.js and profiling/pivot_metrics.js. They use `??`, which needs
# Node >= 14; Ubuntu 22.04's apt package is Node 12, so the binary is taken from the official image.
FROM node:20-bookworm-slim AS node

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# 75 = T4, 80 = A100, 86 = RTX 30xx / A10, 89 = L4 / RTX 40xx, 90 = H100. Each entry also
# embeds PTX, so newer GPUs JIT-compile the sm_90 code. Narrow it (e.g. "75") to build faster.
ARG CUDA_ARCHS="75;80;86;89;90"
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# Nsight Systems (nsys) adds ~1 GB; set to 0 to leave it out. Nsight Compute (ncu) ships with the devel image.
ARG INSTALL_NSIGHT=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions

# Build tools and helpers used by the scripts: g++ (C++17), python3, ninja (PyTorch extension
# builds), binutils' c++filt (resource_usage.sh), lscpu (run_all.sh), zip (packing results).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential git python3 python3-dev python3-pip ninja-build \
        ca-certificates util-linux zip unzip less \
 && CUDA_PKG="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -)" \
 && apt-get install -y --no-install-recommends "cuda-nsight-compute-${CUDA_PKG}" \
 && if [ "${INSTALL_NSIGHT}" = "1" ]; then \
        apt-get install -y --no-install-recommends "cuda-nsight-systems-${CUDA_PKG}"; \
    fi \
 && rm -rf /var/lib/apt/lists/*

# ncu / nsys are installed outside /usr/local/cuda/bin in some CUDA versions: put them on PATH.
RUN for tool in ncu nsys; do \
        if ! command -v "$tool" > /dev/null; then \
            found="$(find /opt/nvidia /usr/local/cuda* -name "$tool" -type f -perm -u+x 2>/dev/null | head -n 1)"; \
            if [ -n "$found" ]; then ln -s "$found" "/usr/local/bin/$tool"; fi; \
        fi; \
    done

COPY --from=node /usr/local/bin/node /usr/local/bin/node

# Python packages. CMake comes from pip: >= 3.25 is needed for the CUDA::nvtx3 target
# (NVTX labels in profile_targets); Ubuntu 22.04's apt CMake is 3.22.
WORKDIR /workspace/cuda-transformer-performance
COPY requirements.txt ./
RUN python3 -m pip install "cmake>=3.25,<4" \
 && python3 -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
 && python3 -m pip install -r requirements.txt "notebook>=7,<8"

# The project, built once in the image (Release, same as scripts/run_all.sh).
# BUILD_JOBS: parallel compile jobs (default: all cores); lower it if the build runs out of memory.
ARG BUILD_JOBS=""
COPY . .
RUN sed -i 's/\r$//' scripts/*.sh profiling/*.sh docker/*.sh \
 && chmod +x scripts/*.sh profiling/*.sh docker/*.sh \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
 && cmake --build build -j "${BUILD_JOBS:-$(nproc)}"

# Jupyter (Colab local runtime, docs/docker.md part B).
EXPOSE 8888

ENTRYPOINT ["/workspace/cuda-transformer-performance/docker/entrypoint.sh"]
CMD ["bash", "scripts/run_all.sh"]
