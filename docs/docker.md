# Running the project in Docker

The repository ships a container image that holds everything the project needs: the CUDA
toolkit, the built kernels, tests and benchmarks, Nsight Compute and Nsight Systems, PyTorch and
Hugging Face Transformers, Node.js for the chart scripts, and a Jupyter server. The source code is
unchanged. The container runs the same scripts as a native install (`scripts/run_all.sh`,
`profiling/*.sh`, `python/*.py`, `llm/run_gpt2.py`, `assets/make_charts.js`).

- **Part A** covers a Linux server with an NVIDIA GPU: setup, build, run, volumes, ports,
  debugging and testing.
- **Part B** covers Google Colab, which cannot run Docker itself. It describes three ways
  to work with the image from Colab.
- **Part C** is troubleshooting.

## Files

| File | Purpose |
|---|---|
| [`Dockerfile`](../Dockerfile) | The image: `nvidia/cuda:12.6.3-devel-ubuntu22.04`, PyTorch 2.7.1 (cu126), CMake ≥ 3.25, Node 20, Nsight tools, project built in `build/` |
| [`.dockerignore`](../.dockerignore) | Keeps `.git`, `build/`, caches and binary profiler reports out of the build context |
| [`docker/entrypoint.sh`](../docker/entrypoint.sh) | Warns if no GPU is visible, provides a `jupyter` shortcut, then runs your command |
| [`docker-compose.yml`](../docker-compose.yml) | The same setup as `docker run`, with GPU, volumes and port defined once |

Inside the container the project is at `/workspace/cuda-transformer-performance`, which is also
the working directory. The default command is `bash scripts/run_all.sh`.

### What needs a GPU

| Works without a GPU (e.g. while building, or on a laptop) | Needs an NVIDIA GPU at run time |
|---|---|
| `docker build`, compiling the kernels, `cuobjdump` resource usage, `python/summarize.py`, `assets/make_charts.js`, `profiling/pivot_metrics.js`, compiling the GPT-2 PyTorch extension | `ctest`, every `bench_*`, `profile_targets`, `ncu`, `nsys`, `python/benchmark.py`, `llm/run_gpt2.py`, `scripts/run_all.sh` |

---

# Part A: Linux server with an NVIDIA GPU

## A1. Host setup (one time)

The host needs only the **NVIDIA driver**, **Docker Engine** and the **NVIDIA Container
Toolkit**. The CUDA toolkit, compilers and Python packages all come with the image.

**1. NVIDIA driver.** The image uses CUDA 12.6, which needs driver **≥ 525** (CUDA
minor-version compatibility). Check with:

```bash
nvidia-smi        # the "Driver Version" field must be >= 525
```

If the driver is missing, install it on Ubuntu with `sudo ubuntu-drivers install` and reboot.
For older drivers, see [C3](#c3-cuda-driver-version-is-insufficient).

**2. Docker Engine** (skip if `docker --version` already works):

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"     # use docker without sudo; log out and back in afterwards
```

**3. NVIDIA Container Toolkit**, which gives containers access to the GPU (Ubuntu/Debian; for other
distributions see NVIDIA's install guide for the toolkit):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**4. Check that containers can see the GPU:**

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

If this prints the GPU table, the host is ready. If it doesn't, see [C1](#c1-could-not-select-device-driver-with-capabilities-gpu).

## A2. Build the image

```bash
git clone <your-repository-url> cuda-transformer-performance
cd cuda-transformer-performance
docker build -t cuda-transformer-performance .
```

The first build downloads about 7 GB (CUDA devel image, Nsight tools and PyTorch). The resulting
image is 7.7 GB; keep about 20 GB of disk free. On an 8-core machine, compiling all kernels for the
five architectures took under a minute, so build time is mostly download time. Later builds reuse
the cached layers. If you only change project source files, only the last step is recompiled.

**Build arguments** (all optional, passed as `--build-arg NAME=value`):

| Argument | Default | When to change it |
|---|---|---|
| `CUDA_ARCHS` | `75;80;86;89;90` | Build only for your GPU, which is faster: T4 `75`, V100 `70`, A100 `80`, A10/RTX 30xx `86`, L4/RTX 40xx `89`, H100 `90`. Newer GPUs run the embedded sm_90 PTX. |
| `CUDA_VERSION` | `12.6.3` | Older host driver ([C3](#c3-cuda-driver-version-is-insufficient)) or a newer toolkit. Any `nvidia/cuda:<ver>-devel-ubuntu22.04` tag. |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | Keep its CUDA version close to `CUDA_VERSION` (`cu118` for 11.8, `cu128` for 12.8). |
| `TORCH_VERSION` | `2.7.1` | Pin another PyTorch release available on that index. |
| `INSTALL_NSIGHT` | `1` | `0` leaves out Nsight Systems (`nsys`, about 1 GB). Nsight Compute (`ncu`) is always included. |
| `BUILD_JOBS` | all cores | Lower it, e.g. `4`, if the build runs out of memory. |

Examples:

```bash
# T4 only (e.g. a cloud T4 server), faster build
docker build --build-arg CUDA_ARCHS=75 -t cuda-transformer-performance .

# Host driver 450-524: CUDA 11.8 toolkit and the matching PyTorch build
docker build --build-arg CUDA_VERSION=11.8.0 \
             --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
             --build-arg CUDA_ARCHS="75;80;86;89;90" \
             -t cuda-transformer-performance:cu118 .
```

With Docker Compose, `docker compose build` does the same, using the arguments listed in
`docker-compose.yml`.

## A3. Run

Every command below runs from the repository root on the host. The volumes (`-v`) write results
into this checkout's `benchmarks/` and `profiling/reports/`, which is exactly where a native
run writes them. They are explained in [A4](#a4-volumes).

To save typing, define the common options once:

```bash
CTP_RUN=(docker run --rm --gpus all --shm-size=2g
  -v "$PWD/benchmarks:/workspace/cuda-transformer-performance/benchmarks"
  -v "$PWD/profiling/reports:/workspace/cuda-transformer-performance/profiling/reports"
  -v ctp-cache:/root/.cache
  cuda-transformer-performance)
```

**Full pipeline**: environment record, correctness tests (stops on failure), all benchmarks,
PyTorch baseline and summary. This is the same as `bash scripts/run_all.sh`:

```bash
"${CTP_RUN[@]}"
# results: benchmarks/<GPU>/summary.md   (e.g. benchmarks/NVIDIA_A100_SXM4_40GB/summary.md)
```

**Individual steps** (the README's step-by-step commands, inside the container):

```bash
"${CTP_RUN[@]}" ctest --test-dir build --output-on-failure        # correctness tests
"${CTP_RUN[@]}" ./build/bench_matmul                               # any bench_* (also: --csv file.csv)
"${CTP_RUN[@]}" python3 python/benchmark.py --ops matmul softmax   # PyTorch baseline
"${CTP_RUN[@]}" python3 python/summarize.py benchmarks/<GPU>       # rebuild summary.md
"${CTP_RUN[@]}" bash profiling/resource_usage.sh                   # registers / smem / spills
```

**GPT-2 on the project's kernels** ([docs/14](14_llm_integration.md)). The first run downloads
the model and compiles the PyTorch extension (a few minutes). Both are kept in the `ctp-cache`
volume, so later runs start immediately:

```bash
"${CTP_RUN[@]}" python3 llm/run_gpt2.py --verify --bench --model gpt2 --csv benchmarks/<GPU>/gpt2.csv
```

**Profiling with Nsight** ([docs/10](10_nsight_profiling.md)). `ncu` reads GPU performance
counters, which needs extra privileges inside a container, so add `--cap-add=SYS_ADMIN` (see
[C5](#c5-ncu-err_nvgpuctrperm)):

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN \
  -v "$PWD/profiling/reports:/workspace/cuda-transformer-performance/profiling/reports" \
  cuda-transformer-performance bash profiling/ncu_kernels.sh gemm

"${CTP_RUN[@]}" bash profiling/nsys_timeline.sh all
```

The `.ncu-rep` / `.nsys-rep` files land in `profiling/reports/<GPU>/` on the host. Open them
with the Nsight GUIs on your PC.

**Charts** (no GPU needed). Regenerate the README charts from a results folder:

```bash
docker run --rm -v "$PWD/assets:/workspace/cuda-transformer-performance/assets" \
  cuda-transformer-performance node assets/make_charts.js
# for another GPU: ... node assets/make_charts.js benchmarks/<GPU> profiling/reports/<GPU> assets/<GPU>
```

**Choosing a GPU** on multi-GPU servers:

```bash
docker run --rm --gpus '"device=1"' cuda-transformer-performance nvidia-smi -L   # only GPU 1
```

Benchmarks use device 0 *inside* the container, which is the one you pass in.

## A4. Volumes

| Host path (bind mount) | Container path | What it holds |
|---|---|---|
| `$PWD/benchmarks` | `/workspace/cuda-transformer-performance/benchmarks` | `benchmarks/<GPU>/` results of `run_all.sh`, `python/benchmark.py`, `run_gpt2.py --csv` |
| `$PWD/profiling/reports` | `/workspace/cuda-transformer-performance/profiling/reports` | Nsight reports and their text/CSV exports |
| `$PWD/assets` (charts only) | `/workspace/cuda-transformer-performance/assets` | regenerated SVG charts |
| named volume `ctp-cache` | `/root/.cache` | Hugging Face models, the compiled GPT-2 extension, pip cache |

Notes:

- A bind mount **replaces** the folder in the image with the host folder. Mounting this checkout's
  own `benchmarks/` keeps the existing T4/A100 results visible (they come from the host). If you
  mount an empty directory instead (e.g. `-v /data/ctp-results:/workspace/.../benchmarks`), only
  the new results appear there.
- The container runs as root, so new files on the host belong to root. Take them back with
  `sudo chown -R "$USER:$USER" benchmarks profiling/reports`.
- Remove the cache with `docker volume rm ctp-cache`.

**Development mode: mount the whole source tree.** Edit kernels on the host and rebuild in the
container, with no image rebuild. The build goes to a separate directory, `build-docker/`, so it
never mixes with a native `build/` (both directory names are in `.dockerignore`):

```bash
docker run --rm -it --gpus all --shm-size=2g \
  -v "$PWD:/workspace/cuda-transformer-performance" -v ctp-cache:/root/.cache \
  cuda-transformer-performance bash

# inside the container:
cmake -S . -B build-docker -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-docker -j
ctest --test-dir build-docker --output-on-failure
```

`native` works here because the GPU is visible at run time. `scripts/run_all.sh` and the profiling
scripts always use `build/`. To run them in development mode, configure `build/` instead of
`build-docker/`.

## A5. Ports: Jupyter

The image includes Jupyter Notebook, started with the `jupyter` shortcut on port 8888. It allows
the Google Colab origin, so it can serve as a Colab local runtime ([Part B](#b1-recommended-colab-ui--this-container-on-your-gpu-server)).

```bash
docker run --rm -it --gpus all --shm-size=2g -p 127.0.0.1:8888:8888 \
  -v "$PWD/benchmarks:/workspace/cuda-transformer-performance/benchmarks" \
  -v ctp-cache:/root/.cache \
  cuda-transformer-performance jupyter
```

- `-p 127.0.0.1:8888:8888` publishes container port 8888 on the host's **loopback only**. This
  server can run arbitrary code, so don't publish it on `0.0.0.0` on a server with a public IP.
  Reach it through an SSH tunnel instead (below).
- The log prints a URL with a token: `http://127.0.0.1:8888/tree?token=...`. To get it again:
  `docker exec <container> jupyter server list`.
- Use a different host port if 8888 is taken: `-p 127.0.0.1:8899:8888`. To change the port
  inside the container as well: `-e JUPYTER_PORT=9000 -p 127.0.0.1:9000:9000`.
- Extra Jupyter options go after `jupyter`, e.g. `... cuda-transformer-performance jupyter --IdentityProvider.token=mysecret`.

From your laptop, forward the server's port and open the URL locally:

```bash
ssh -N -L 8888:127.0.0.1:8888 user@your-gpu-server
# then open http://127.0.0.1:8888/tree?token=... in the laptop's browser
```

## A6. Docker Compose

`docker-compose.yml` has the GPU reservation, volumes, shared memory and port built in. You need
Docker Compose v2 (`docker compose version`).

```bash
docker compose build
docker compose run --rm ctp                                           # full pipeline
docker compose run --rm ctp ctest --test-dir build --output-on-failure
docker compose run --rm ctp python3 llm/run_gpt2.py --verify --model gpt2
docker compose up jupyter                                             # Jupyter on 127.0.0.1:8888
docker compose down
```

## A7. Debugging

**Get a shell** in a fresh container, or in a running one:

```bash
docker run --rm -it --gpus all cuda-transformer-performance bash
docker ps                                   # find the container name / ID
docker exec -it <container> bash            # shell in a running container (e.g. the Jupyter one)
docker logs -f <container>                  # its output
```

Bypass the entrypoint if needed: `docker run --rm -it --entrypoint bash cuda-transformer-performance`.

**Check the environment** inside the container:

```bash
nvidia-smi
nvcc --version && cmake --version | head -n 1 && node --version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
which ncu nsys compute-sanitizer cuda-gdb
```

**Debug CUDA errors.** The devel image includes NVIDIA's debugging tools:

```bash
# Synchronize after every launch, so errors are reported at the kernel that caused them
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DCTP_DEBUG_SYNC=ON
cmake --build build-debug -j && ./build-debug/test_softmax

# Out-of-bounds / misaligned memory accesses, race conditions in shared memory
compute-sanitizer ./build/test_softmax
compute-sanitizer --tool racecheck ./build/test_matmul

# Step through a kernel (Debug build with -G for device debug info)
cmake -S . -B build-g -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_FLAGS=-G
cmake --build build-g -j --target test_vector_add && cuda-gdb ./build-g/test_vector_add
```

`cuda-gdb` may need `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` on `docker run`.

**GPT-2 extension build output:** `python3 llm/run_gpt2.py --verbose-build ...`.

## A8. Testing the image

Run these checks after building. The first four don't need a GPU:

```bash
IMG=cuda-transformer-performance

# 1. Toolchain and Python stack
docker run --rm $IMG bash -c 'nvcc --version | tail -n 2; cmake --version | head -n 1; node --version;
  ncu --version | tail -n 1; nsys --version;
  python3 -c "import torch, transformers, numpy; print(torch.__version__, torch.version.cuda, transformers.__version__)"'

# 2. Every executable was built
docker run --rm $IMG bash -c 'ls build/test_* build/bench_* build/profile_targets'

# 3. Data tools reproduce the committed outputs (no output from diff = identical)
docker run --rm $IMG bash -c 'node assets/make_charts.js benchmarks/Tesla_T4 profiling/reports/Tesla_T4 /tmp/charts \
  && for f in /tmp/charts/*.svg; do diff -q "$f" "assets/$(basename "$f")"; done && echo charts OK'

# 4. Kernel resource usage from the binary
docker run --rm $IMG bash -c 'cuobjdump --dump-resource-usage build/libctp_core.a | c++filt | grep -c Function'

# 5. With a GPU: correctness tests (must say "100% tests passed, 0 tests failed out of 7")
docker run --rm --gpus all $IMG ctest --test-dir build --output-on-failure

# 6. With a GPU: one benchmark, and PyTorch sees the GPU
docker run --rm --gpus all $IMG ./build/bench_vector_add
docker run --rm --gpus all $IMG python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

# Part B: Google Colab

**Colab cannot run Docker.** A Colab VM is itself a container: it has no Docker daemon, and you
cannot get the privileges to start one, so `docker build` and `docker run` don't work in a Colab
notebook. There are three ways to use this project from Colab:

| Option | GPU comes from | Uses this image | Effort |
|---|---|---|---|
| **B1.** Colab UI connected to this container on your server (local runtime) | your server | yes | low, needs a GPU machine |
| **B2.** Run the image inside a Colab VM with `udocker` | Colab | yes, without root | experimental |
| **B3.** No container: the existing notebooks | Colab | no, same scripts | none |

## B1. (Recommended) Colab UI + this container on your GPU server

Colab can send code to a Jupyter server you run yourself ("Connect to a local runtime"). The
container from Part A becomes the Colab runtime: you use the notebook interface, and every cell
runs in the container on your server's GPU.

**1. On the GPU server**, build the image ([A2](#a2-build-the-image)) and start Jupyter:

```bash
cd cuda-transformer-performance
docker run --rm -it --gpus all --shm-size=2g -p 127.0.0.1:8888:8888 \
  -v "$PWD/benchmarks:/workspace/cuda-transformer-performance/benchmarks" \
  -v "$PWD/profiling/reports:/workspace/cuda-transformer-performance/profiling/reports" \
  -v ctp-cache:/root/.cache \
  cuda-transformer-performance jupyter
```

Copy the URL with the token from the log: `http://127.0.0.1:8888/tree?token=<TOKEN>`.

**2. On your laptop**, forward the port (the browser running Colab must reach `127.0.0.1:8888`):

```bash
ssh -N -L 8888:127.0.0.1:8888 user@your-gpu-server
```

If Docker runs on the laptop itself (Linux, or Windows with WSL2 and an NVIDIA GPU), skip the tunnel.

**3. In Colab**, open the notebook (e.g. `notebooks/colab_run.ipynb`, or a new one) → **Connect**
(top right) arrow → **Connect to a local runtime** → paste
`http://127.0.0.1:8888/?token=<TOKEN>` → **Connect**.

**4. Run cells.** The project is already the working directory, so skip the clone/upload cells and
run the commands directly:

```python
!nvidia-smi
!ctest --test-dir build --output-on-failure
!bash scripts/run_all.sh
```

```python
import glob
from IPython.display import Markdown, display
display(Markdown(open(sorted(glob.glob("benchmarks/*/summary.md"))[-1]).read()))
```

Results are written to the server's `benchmarks/` through the volume. `files.download` from
`google.colab` doesn't work on a local runtime, so copy the files with `scp` instead.

## B2. (Experimental) Run the image inside a Colab GPU VM with udocker

[udocker](https://github.com/indigo-dc/udocker) runs Docker images without a daemon or root
privileges, and can pass the NVIDIA driver into the container. This path is experimental: it was
not verified for this project, and the Nsight profilers are unlikely to work under it. Use B1 or
B3 for profiling.

**1. On a machine with Docker**, build the image and push it to a registry Colab can pull from.
For example, with the GitHub Container Registry (needs a personal access token with
`write:packages`):

```bash
docker build --build-arg CUDA_ARCHS=75 --build-arg INSTALL_NSIGHT=0 -t ghcr.io/<user>/cuda-transformer-performance:t4 .
echo <TOKEN> | docker login ghcr.io -u <user> --password-stdin
docker push ghcr.io/<user>/cuda-transformer-performance:t4
```

`CUDA_ARCHS=75` matches Colab's free T4. Use `75;80` if you also use A100 runtimes. Make the
package public on GitHub, or run `udocker login` in Colab first.

**2. In Colab** (Runtime → Change runtime type → **T4 GPU**), run:

```python
!pip install -q udocker
!udocker --allow-root install
!udocker --allow-root pull ghcr.io/<user>/cuda-transformer-performance:t4
!udocker --allow-root create --name=ctp ghcr.io/<user>/cuda-transformer-performance:t4
!udocker --allow-root setup --execmode=F3 --nvidia ctp
```

```python
!udocker --allow-root run ctp nvidia-smi
!udocker --allow-root run --workdir=/workspace/cuda-transformer-performance ctp ctest --test-dir build --output-on-failure
!udocker --allow-root run --workdir=/workspace/cuda-transformer-performance \
    --volume=/content/results:/workspace/cuda-transformer-performance/benchmarks \
    ctp bash scripts/run_all.sh
```

Results land in `/content/results`. Download them with
`!zip -q -r results.zip /content/results` and `files.download("results.zip")`.

If `nvidia-smi` fails inside udocker, run `setup` again with `--execmode=F1` or `--execmode=P1`.
If it still fails, use B3.

## B3. No container: the existing Colab notebooks

Colab's GPU runtime already has the CUDA toolkit, CMake, PyTorch and Transformers, which is
everything this image adds. The project's notebooks run the same scripts natively:

1. Open [`notebooks/colab_run.ipynb`](../notebooks/colab_run.ipynb) in Colab →
   Runtime → Change runtime type → **GPU** → run all cells (build → tests → benchmarks → summary).
2. Profiling: [`notebooks/colab_profile.ipynb`](../notebooks/colab_profile.ipynb).

The results have the same format as the container's, so you can mix them.

---

# Part C: Troubleshooting

### C1. `could not select device driver "" with capabilities: [[gpu]]`

Docker doesn't know about the NVIDIA runtime. Install the NVIDIA Container Toolkit and restart
Docker ([A1 step 3](#a1-host-setup-one-time)), then check with
`docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.

### C2. `WARNING: no NVIDIA GPU is visible in this container`

The container was started without `--gpus all` (Compose: the `deploy.resources` block is
missing, or `docker-compose` v1 is in use; upgrade to `docker compose` v2). Also check that
`nvidia-smi` works on the host. With rootless Docker, set `no-cgroups = true` in
`/etc/nvidia-container-runtime/config.toml`.

### C3. `CUDA driver version is insufficient for CUDA runtime version`

The host driver is older than the image's CUDA needs (CUDA 12.x: ≥ 525). Either update the driver,
or build with CUDA 11.8, which needs a driver ≥ 450:
`docker build --build-arg CUDA_VERSION=11.8.0 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 -t cuda-transformer-performance:cu118 .`

### C4. `no kernel image is available for execution on the device`

The image wasn't compiled for this GPU's architecture. Check it with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`, then rebuild with that value included,
e.g. `--build-arg CUDA_ARCHS="70;75;80"` for a V100 (7.0). Tensor Core tests need compute
capability ≥ 7.0.

### C5. `ncu`: `ERR_NVGPUCTRPERM` (no permission to access GPU performance counters)

Add `--cap-add=SYS_ADMIN` to `docker run`. If that's not allowed, enable counters for all users
on the **host** and reboot:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

Cloud VMs and MIG slices may not allow counters at all ([docs/09 §21](09_benchmarking.md)).
`nsys` and `profiling/resource_usage.sh` don't need counters.

### C6. Permission denied on `benchmarks/` or `profiling/reports/` on the host

Files written by the container belong to root: `sudo chown -R "$USER:$USER" benchmarks profiling/reports`.
On hosts with SELinux (Fedora, RHEL), add `:z` to bind mounts: `-v "$PWD/benchmarks:/workspace/cuda-transformer-performance/benchmarks:z"`.

### C7. Build fails with `Killed` or `nvcc ... terminated by signal 9`

The compile ran out of memory. Use fewer parallel jobs and/or fewer architectures:
`docker build --build-arg BUILD_JOBS=2 --build-arg CUDA_ARCHS=75 -t cuda-transformer-performance .`
On Docker Desktop, raise the memory limit under Settings → Resources.

### C8. `port is already allocated` / Colab can't connect to the local runtime

- Choose another host port: `-p 127.0.0.1:8899:8888`, then use `http://127.0.0.1:8899/?token=...`.
- The Colab browser must reach that address. Check the SSH tunnel is running, and open the URL in a
  normal tab first.
- Copy the token exactly: `docker exec <container> jupyter server list`.
- Browsers that block third-party access to localhost may need it allowed for colab.research.google.com.

### C9. GPT-2: extension build errors, or the model download fails

- Delete a stale extension build: `docker run --rm -v ctp-cache:/root/.cache cuda-transformer-performance rm -rf /root/.cache/torch_extensions`.
- Show the compiler output: `--verbose-build`.
- With no internet access, download the model on another machine into a Hugging Face cache,
  mount it at `/root/.cache/huggingface`, and run with `-e HF_HUB_OFFLINE=1`.

### C10. `CMake Error ... CMAKE_CUDA_ARCHITECTURES "native" ... no GPU`

`native` can only be used when a GPU is visible, so it fails in `docker build` or in a
container started without `--gpus`. Pass an explicit list instead, e.g.
`-DCMAKE_CUDA_ARCHITECTURES=75`.

### C11. Bash scripts fail with `$'\r': command not found`

The scripts were saved with Windows line endings. The image converts the scripts it copies. For a
mounted source tree (development mode), run `sed -i 's/\r$//' scripts/*.sh profiling/*.sh docker/*.sh`,
or rely on `.gitattributes`, which checks them out with LF.

### C12. Benchmark numbers differ from a native run

The container adds no measurable overhead to GPU kernels. Different numbers usually come from a
different GPU, clocks, other jobs on a shared GPU, or MIG slices. Read [docs/09 §21](09_benchmarking.md)
before comparing, and check `benchmarks/<GPU>/environment.txt` from both runs.
