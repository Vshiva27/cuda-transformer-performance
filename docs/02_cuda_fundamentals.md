# 02 — CUDA Fundamentals (Phase 1: Vector Addition)

This document explains every idea used in Phase 1. Read it top to bottom; each section only
uses words that were explained in an earlier section.

Files covered: `kernels/vector_add.cu`, `kernels/vector_add.cuh`, `src/cpu/cpu_ops.*`,
`src/utils/*`, `src/benchmark/bench_utils.cuh`, `src/benchmark/bench_vector_add.cu`,
`tests/test_vector_add.cu`, `CMakeLists.txt`.

---

## 0. Vocabulary (read this first)

| Word | Meaning in plain English |
|---|---|
| **Host** | The CPU and its memory (normal RAM). Code in `main()` runs on the host. |
| **Device** | The GPU and its own memory (VRAM). The CPU cannot directly read device memory, and the GPU cannot directly read normal host memory. |
| **Kernel** | A function that runs **on the GPU**, executed by many threads at once. Marked `__global__`. |
| **Thread** | One worker that executes the kernel's code once, with its own variables. A GPU runs thousands at the same time. |
| **Block** (thread block) | A group of threads (up to 1024). Threads in the same block can cooperate (share fast memory, wait for each other). |
| **Grid** | All the blocks launched by one kernel call. |
| **Warp** | A group of **32 threads** inside a block that the hardware executes together, as one unit, running the same instruction at the same time. |
| **SM** (Streaming Multiprocessor) | One "core cluster" of the GPU. A T4 has 40 SMs, an A100 has 108. Each block runs entirely on one SM. One SM can hold several blocks at once. |
| **Global memory** | The GPU's large main memory (e.g. 16 GB on a T4). Big but slow: hundreds of clock cycles per access. Every thread can read it. |
| **Register** | The tiniest, fastest storage: a variable private to one thread, inside the SM. Local variables like `int i` normally live in registers. |
| **Latency** | How long one thing takes (e.g. one inference request: 20 ms). |
| **Throughput** | How many things finish per second (e.g. 500 requests/s, or 300 GB/s of memory traffic). |
| **Bandwidth** | Maximum data per second a memory can deliver. T4 ≈ 320 GB/s, A100 ≈ 1555 GB/s (theoretical). |

---

## 1. Why a GPU at all?

A CPU has a few (4–64) very powerful cores, each designed to finish *one* sequence of
instructions as fast as possible. A GPU has thousands of simple lanes designed to run *the
same instruction on lots of different data* at once.

Transformer inference is mostly "do the same arithmetic on millions of numbers" (multiply
matrices, add vectors, normalize rows). That is exactly the GPU's strength.

The GPU's weakness: it has a startup cost (launching a kernel takes a few microseconds) and its
memory is separate from the CPU's (copying data over the PCIe bus is slow). So a GPU only wins
when there is **enough work** and the data **stays on the GPU**. Phase 1's benchmark is designed
to show both effects.

---

## 2. The problem: vector addition

```
A = [1, 2, 3, 4]
B = [10, 20, 30, 40]
C = A + B = [11, 22, 33, 44]      C[i] = A[i] + B[i]
```

**Where this appears in a Transformer:** the residual connection, `x = x + sublayer(x)`, is
exactly a vector addition over all elements of the hidden state.

### CPU version (`src/cpu/cpu_ops.cpp`)

```cpp
void vector_add(const float* a, const float* b, float* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

- `const float* a` — a **pointer** (a memory address) to the first `float` of array A.
  `const` means "this function promises not to modify A".
- `float* c` — pointer to the output; not `const`, because we write to it.
- `std::size_t` — an unsigned integer type big enough to count any array size.
- The loop does element 0, then 1, then 2 … one after another. For n = 16 million, that is 16
  million sequential steps (the compiler may use SIMD instructions to do 4–8 at a time, but
  still on one core).

**Key observation:** each `c[i]` depends only on `a[i]` and `b[i]`. No element needs any other
element. So all n additions *could* happen at the same time. This is called an
**embarrassingly parallel** problem.

---

## 3. The GPU idea: one thread per element

Instead of one worker looping over n elements, launch n workers (threads), and let thread
number `i` compute only `c[i]`.

The only question each thread must answer is: **"What is my i?"**

### The thread hierarchy

You don't launch n separate threads. You launch a **grid** of **blocks**, each with the same
number of **threads**:

```
Launch: grid of 4 blocks, 8 threads per block (blockDim.x = 8, gridDim.x = 4)

            block 0                 block 1                 block 2                 block 3
       ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
thread │ 0 1 2 3 4 5 6 7 │     │ 0 1 2 3 4 5 6 7 │     │ 0 1 2 3 4 5 6 7 │     │ 0 1 2 3 4 5 6 7 │  ← threadIdx.x
       └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
global i:  0 … 7                  8 … 15                  16 … 23                 24 … 31
```

Each thread can read four built-in variables (CUDA fills them in automatically):

| Variable | Meaning | Same for whom? |
|---|---|---|
| `threadIdx.x` | my position **inside my block** (0 … blockDim.x−1) | different for each thread in a block |
| `blockIdx.x` | which block I am in (0 … gridDim.x−1) | same for all threads in a block |
| `blockDim.x` | how many threads per block | same for everyone |
| `gridDim.x` | how many blocks in the grid | same for everyone |

They have `.x`, `.y`, `.z` parts because grids and blocks can be 1D, 2D or 3D. Vector add is
1D, so we only use `.x`. Matrix multiply (Phase 2) will use `.x` for columns and `.y` for rows.

---

## 4. The kernel, line by line (`kernels/vector_add.cu`)

```cpp
__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Line 1: `__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c, int n)`

- `__global__` — a CUDA keyword meaning "this function runs on the GPU and is called from the
  CPU". That makes it a kernel.
- `void` — kernels never return a value. Results are written to memory (here, to `c`).
- `const float* a, const float* b` — pointers to the inputs **in GPU memory**. If you pass a
  CPU pointer here, the kernel crashes with an "illegal memory access" error.
- `float* c` — pointer to the output in GPU memory.
- `int n` — the number of elements. It is passed by value: every thread gets its own copy.

Important: **this code is executed by every thread**. If you launch 1024 threads, these lines
run 1024 times in parallel, each time with different `threadIdx`/`blockIdx` values.

### Line 2: `int i = blockIdx.x * blockDim.x + threadIdx.x;`

Piece by piece:

- `int` — a 32-bit signed integer. It lives in a **register** (private to this thread).
- `i` — the name we chose for "the global index of this thread".
- `blockIdx` — built-in: the index of the block this thread belongs to.
- `.x` — the x-dimension of it (the only one we use here).
- `*` — multiplication.
- `blockDim` — built-in: the size of each block.
- `.x` — again the x-dimension: the number of threads per block in x.
- `+` — addition.
- `threadIdx` — built-in: this thread's position inside its block.
- `.x` — x-dimension.

**Why does this give the global index?** Think of seats in a cinema: each row has `blockDim.x`
seats. If you are in row `blockIdx.x`, all earlier rows contain `blockIdx.x * blockDim.x`
seats. Your seat number in the whole cinema is "seats in earlier rows + your seat in this row":

```
i = (number of threads in all earlier blocks) + (my position in my block)
  =        blockIdx.x * blockDim.x            +        threadIdx.x
```

**Concrete example** with `blockDim.x = 256`:

| blockIdx.x | threadIdx.x | calculation | i |
|---|---|---|---|
| 0 | 0 | 0·256 + 0 | 0 |
| 0 | 255 | 0·256 + 255 | 255 |
| 1 | 0 | 1·256 + 0 | 256 |
| 2 | 5 | 2·256 + 5 | 517 |
| 3 | 231 | 3·256 + 231 | 999 |

Every i from 0 upward is produced exactly once. No gaps, no duplicates.

### Line 3: `if (i < n) {`

The **bounds check**. Blocks all have the same size, but n may not be a multiple of it.

Example: n = 1000, block size 256 → we need ⌈1000/256⌉ = 4 blocks = 1024 threads. Threads
with i = 1000 … 1023 have no element to work on. Without this check they would read past
the end of `a` and `b` and **write past the end of `c`**, corrupting other memory or crashing.

### Line 4: `c[i] = a[i] + b[i];`

- `a[i]` — read the i-th float from global memory.
- `b[i]` — read the i-th float of B.
- `+` — one floating-point addition (in a register).
- `c[i] = …` — write the result to global memory.

So each thread does **2 reads, 1 addition, 1 write**.

---

## 5. Launching the kernel (host side)

```cpp
void vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
    if (n <= 0) return;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_naive_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();
}
```

- The `d_` prefix is a naming convention: "this pointer points to **device** memory".
- `if (n <= 0) return;` — a launch with 0 blocks is an error in CUDA, so empty input is handled
  on the CPU side.
- `(n + block_size - 1) / block_size` — **ceiling division** using integers. Integer division in
  C++ rounds down, so adding `block_size - 1` first makes it round up:
  - n = 1000, block = 256: (1000 + 255) / 256 = 1255 / 256 = **4** (4.9 rounded down). ✔
  - n = 1024, block = 256: (1024 + 255) / 256 = 1279 / 256 = **4** (exactly enough). ✔
  - n = 1025, block = 256: (1025 + 255) / 256 = 1280 / 256 = **5**. ✔
- `kernel<<<grid_size, block_size>>>(args)` — the CUDA launch syntax. The values between `<<<`
  and `>>>` are the **launch configuration**: number of blocks, then threads per block. This line
  does **not** wait for the kernel to finish (see §10).
- `CUDA_CHECK_KERNEL()` — checks whether the launch was accepted (see §9).

### Why block size 256?

- Must be ≤ 1024 (hardware limit).
- Should be a **multiple of 32** (warp size), otherwise the last warp of each block is partly
  empty and wastes hardware lanes.
- 128–512 usually works well: big enough to give the SM plenty of warps, small enough that
  several blocks can share an SM. The benchmark's block-size sweep lets you check this yourself.

---

## 6. Warps: how threads really execute

The SM does not schedule individual threads. It schedules **warps of 32 threads**. All 32
threads in a warp execute the **same instruction at the same time** on different data. This
model is called **SIMT** (Single Instruction, Multiple Threads).

```
block of 256 threads = 8 warps
warp 0: threads   0 –  31
warp 1: threads  32 –  63
…
warp 7: threads 224 – 255
```

**Warp divergence:** if threads in one warp take different sides of an `if`, the warp must run
both sides one after another, with some threads switched off each time. In vector add, only one
warp in the whole grid (the one containing i = n) can have threads on both sides of `if (i < n)`.
So divergence costs nothing here. It matters later, in reductions (Softmax/LayerNorm).

**Why warps hide memory latency:** a global memory read takes ~400–800 clock cycles. When a
warp waits for memory, the SM instantly switches to another warp that is ready to compute.
With many warps on each SM, there is almost always one that is ready. That is why GPUs want
*many more threads than cores*.

---

## 7. Memory access pattern: coalescing (first look)

When the 32 threads of a warp execute `a[i]`, they request `a[32w+0], a[32w+1], …, a[32w+31]`.
These are 32 **consecutive** floats = 128 consecutive bytes. The memory system can serve this
with very few large memory transactions. This is called a **coalesced** access, and it is the
best possible pattern.

```
warp threads:   t0   t1   t2   t3  …  t31
reads:         a[0] a[1] a[2] a[3] … a[31]     ← one contiguous 128-byte chunk ✔ coalesced
```

If instead thread t read `a[t * 32]`, each thread would touch a different 128-byte chunk, and
the warp would need 32 separate transactions for the same amount of useful data. Phase 2 (GEMM)
will show this effect directly by comparing two kernels that differ only in their access pattern.

---

## 8. Version 2: the grid-stride loop

```cpp
__global__ void vector_add_grid_stride_kernel(const float* a, const float* b, float* c, int n) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = start; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

- `start` — the same global index formula as before: this thread's first element.
- `stride = gridDim.x * blockDim.x` — the **total number of threads in the grid**.
- The loop: handle element `start`, then jump forward by the whole grid's width, and repeat
  until past the end.

**Example:** n = 10, grid = 2 blocks × 2 threads = 4 threads, so stride = 4.

| thread (block, thread) | start | elements handled |
|---|---|---|
| (0,0) | 0 | 0, 4, 8 |
| (0,1) | 1 | 1, 5, 9 |
| (1,0) | 2 | 2, 6 |
| (1,1) | 3 | 3, 7 |

Every element 0–9 is covered exactly once. In each loop step, neighboring threads still touch
neighboring elements (0,1,2,3 then 4,5,6,7), so access stays **coalesced**.

**Why bother?**
1. The kernel works for **any n with any grid size**. The test proves this using grids of 1 and 3
   blocks.
2. We can launch exactly enough blocks to fill the GPU once, instead of millions of blocks.
   Each thread does more work, and per-block setup costs are paid fewer times.
3. The bounds check is built into the loop condition `i < n`.

For vector add the speed difference is usually small, because the operation is limited by
memory bandwidth either way. The pattern matters more later (Softmax, LayerNorm), and it is a
standard interview topic.

### How many blocks for the grid-stride version? Occupancy (first look)

**Occupancy** = (warps active on an SM) ÷ (maximum warps an SM can hold). The number of blocks
that can live on one SM at the same time is limited by:
- threads per SM (e.g. 1024 on T4, 2048 on A100),
- a maximum number of blocks per SM (16 on T4, 32 on A100),
- registers used by each thread,
- shared memory used by each block.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_size, 0)` asks CUDA
to do that calculation for our kernel. We launch `sm_count × blocks_per_sm` blocks: enough to fill
every SM completely, and no more.

Example: block size 32 on a T4 → at most 16 blocks per SM × 32 threads = 512 threads, which is
only half of the SM's 1024. Occupancy is 50%, *even though the code is identical*. You may see
this in the block-size sweep. Use your own run's numbers, not this prediction.

---

## 9. Error handling (`src/utils/cuda_check.cuh`)

### `cudaError_t`

Almost every CUDA runtime function returns a `cudaError_t`, an enum value. `cudaSuccess` (0)
means OK; anything else is an error code such as `cudaErrorMemoryAllocation`.
`cudaGetErrorName(err)` and `cudaGetErrorString(err)` turn it into readable text.

### The `CUDA_CHECK` macro

```cpp
#define CUDA_CHECK(call) cuda_check_impl((call), #call, __FILE__, __LINE__)
```

- A **macro** is text substitution done before compilation.
- `(call)` — runs the CUDA call and passes its return value.
- `#call` — the "stringify" operator: turns the code itself into a string, e.g.
  `"cudaMalloc(&ptr_, count_ * sizeof(T))"`, so the error message shows the failing code.
- `__FILE__`, `__LINE__` — filled in by the compiler with the current file name and line.
- `cuda_check_impl` prints all that and calls `exit(EXIT_FAILURE)` if the call failed.

### Why kernel errors are tricky

A kernel launch has no return value. Errors can appear at two different moments:

1. **Launch errors**: CUDA refuses to start the kernel, for example because the block size is
   2048 (> 1024) or the grid size is 0. These are detected at launch time and stored.
   `cudaGetLastError()` returns the stored error and resets it. That is what `CUDA_CHECK_KERNEL()`
   calls after every launch.
2. **Execution errors**: the kernel started but then did something illegal, for example reading
   outside an array. The CPU has already moved on (the launch is asynchronous, §10), so the error
   is only reported by the **next call that waits for the GPU**, such as `cudaMemcpy`,
   `cudaDeviceSynchronize()` or `cudaEventSynchronize()`. The error message then points to *that*
   line, not to the kernel that caused it. This confuses many beginners.

`cudaDeviceSynchronize()` makes the CPU wait until **all** previously queued GPU work has finished,
and returns any error that happened during it.

To find which kernel caused an execution error, build with `-DCTP_DEBUG_SYNC=ON`. Then
`CUDA_CHECK_KERNEL()` also calls `cudaDeviceSynchronize()` after every launch, and the error is
reported at the guilty launch. This is slow, so it is only for debugging.
(`compute-sanitizer ./build/test_vector_add` is NVIDIA's tool that finds out-of-bounds accesses
precisely.)

Most execution errors are **sticky**: after an illegal memory access, the CUDA context is broken
and every later CUDA call fails too. That is another reason to stop immediately with `exit`.

---

## 10. Asynchronous execution and timing

### The work queue

When the CPU executes `kernel<<<…>>>(…)`, it does **not** run the kernel. It puts a "run this
kernel" command into a queue, called a **stream**, and returns within a few microseconds. The GPU
takes commands from the queue in order. This is **asynchronous execution**: the CPU and GPU work
at the same time.

```
CPU:  [launch K1][launch K2][launch K3][…CPU continues with other work…][sync: wait]
GPU:         [------ K1 ------][------ K2 ------][------ K3 ------]
```

### Why CPU wall-clock timing of a kernel is wrong

```cpp
timer.start();
kernel<<<…>>>(…);        // returns immediately!
double ms = timer.stop_ms();   // measured only the time to queue the launch
```

The benchmark prints this on purpose in the **`no-sync`** column. For large n it will be far
smaller than the real kernel time. It measures the launch, not the work.

### Option 1: wall-clock + synchronize

Call `cudaDeviceSynchronize()` before stopping the CPU timer. This is correct, but it also counts
CPU-side launch overhead and OS scheduling noise. We use this for **end-to-end** time, because
that is exactly what a caller waits for.

### Option 2: CUDA events (`src/utils/timer.cuh`)

A **CUDA event** is a marker placed in the GPU's queue. When the GPU reaches the marker, it
records a GPU timestamp.

```cpp
cudaEventRecord(start_);          // marker 1 goes into the queue
kernel<<<…>>>(…);                 // the work
cudaEventRecord(stop_);           // marker 2 goes into the queue
cudaEventSynchronize(stop_);      // CPU waits until the GPU reaches marker 2
cudaEventElapsedTime(&ms, start_, stop_);  // GPU time between the markers
```

This measures time **on the GPU's clock**, between the two points in the queue. It excludes CPU
overhead, so it is the right tool for **kernel time**. Resolution is about 0.5 µs.

### Warm-up and averaging (`src/benchmark/bench_utils.cuh`)

`time_gpu_ms(fn, warmup, iters)`:
1. Runs `fn` `warmup` times and synchronizes. The first launches pay one-time costs: loading the
   kernel code onto the GPU, the GPU raising its clock from idle, and caches being cold.
2. Records a start event, launches `fn` `iters` times back-to-back, records a stop event.
3. Returns total ÷ iters.

A single event pair around all iterations means the GPU runs the kernels back-to-back, which is
how inference really runs (many kernels queued one after another).

### Kernel time vs end-to-end time

```
end-to-end = copy A to GPU + copy B to GPU + kernel + copy C back to CPU
```

Copies go over **PCIe**, the connection between the CPU and the GPU card: roughly 6–25 GB/s,
compared with roughly 300–2000 GB/s inside the GPU. For vector add, which does very little math
per byte, the copies take **much longer than the kernel**. End-to-end GPU time can even be
*slower than the CPU*.

**Lesson for inference:** real inference engines load the model weights onto the GPU **once** and
keep activations on the GPU between layers. Only the small input (tokens) and output (logits or
token IDs) cross PCIe.

---

## 11. Performance: why vector add is memory-bound

**Arithmetic intensity** = useful math operations ÷ bytes moved from/to memory.

Per element: 1 addition (1 FLOP); bytes = read 4 + read 4 + write 4 = 12 bytes.
Arithmetic intensity = 1/12 ≈ **0.083 FLOP/byte**.

A T4 can do ~8,000 GFLOP/s in FP32 but only move ~320 GB/s. To keep the math units busy, a kernel
would need ~8000/320 = **25 FLOPs per byte**. Vector add has 300× less. So the math units sit
idle, and the speed is set entirely by **memory bandwidth**. We call this **memory-bound**.

That is why the benchmark reports **GB/s** and **% of peak bandwidth**, not FLOP/s:

```
bandwidth = (3 × n × 4 bytes) / kernel_time
```

What to expect (predictions, check against your own run):
- **Small n** (1K–16K): the kernel lasts only a few µs, dominated by fixed launch overhead.
  Bandwidth is far below peak, and the CPU may even be faster.
- **Large n** (4M+): bandwidth approaches 70–90% of theoretical peak. That is the ceiling for a
  memory-bound kernel. No code change can make vector add faster than the memory allows. The only
  way to win further is to **move fewer bytes**, for example by fusing it with the next operation
  (Phase 6: residual add fused into LayerNorm) or by using FP16 (Phase 7: half the bytes).

The **CPU/naive** column is the speedup of GPU kernel time over CPU time. It is not the
end-to-end speedup; compare the CPU column with the end-to-end column for that.

---

### Measured on a Tesla T4 (12_results §1)

- ✔ Large vectors: **262.7 GB/s = 82% of the 320 GB/s theoretical peak**.
- ✔ n = 1,024: 2.7 µs (the launch floor). The single-thread CPU is 10× faster at this size.
- ✔ End-to-end with PCIe copies at n = 2²⁶: 173.7 ms, **2.6× slower than the CPU** (66.9 ms),
  for a kernel that takes 3.1 ms.
- A surprise worth understanding: n = 262,144 showed **728 GB/s, 227% of DRAM peak**. The three
  arrays total 3 MB and fit in the T4's 4 MB **L2 cache**, so the repeated benchmark iterations
  read from L2, not DRAM. Small benchmarks can measure the cache instead of memory. Always
  check whether your working set fits in L2.
- The grid-stride version with 160 blocks was slower than one-thread-per-element at large n
  (3.88 vs 3.07 ms): too few memory requests in flight to saturate DRAM.

## 12. DeviceBuffer: RAII for GPU memory (`src/utils/device_buffer.cuh`)

Manual CUDA memory management looks like this:

```cpp
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));   // allocate on GPU
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
...
cudaFree(d_a);                         // easy to forget, or skip by an early return
```

- `cudaMalloc(&d_a, bytes)` — reserves `bytes` of GPU global memory and stores the address in
  `d_a`. We pass `&d_a` (the address of our pointer variable) so the function can write the new
  address into it.
- `cudaMemcpy(dst, src, bytes, direction)` — copies bytes. The direction tells CUDA which side is
  which. This call **waits** for previously queued GPU work and for the copy itself before
  returning (for normal "pageable" CPU memory).
- `cudaFree(d_a)` — gives the memory back.

`DeviceBuffer<T>` wraps this:

- **Template** (`template <typename T>`): the class is written once and works for any element type:
  `DeviceBuffer<float>`, and later `DeviceBuffer<__half>` for FP16.
- **Constructor** calls `cudaMalloc`; **destructor** `~DeviceBuffer()` calls `cudaFree`. C++
  guarantees the destructor runs when the object goes out of scope, so memory cannot leak.
- **Copy is deleted** (`= delete`): copying would give two objects the same pointer, and both
  destructors would free it (double free → crash).
- **Move is allowed**: ownership is transferred and the old object is set to `nullptr`.
- `explicit` on the constructor prevents accidental conversion, such as `DeviceBuffer<float> b = 5;`.
- `copy_from_host` / `copy_to_host` take a `std::vector` and check that sizes match.

---

## 13. Device query (`src/utils/device_info.*`)

At startup every program calls `query_gpu_info()`, which uses:
- `cudaGetDeviceCount` — are there any GPUs? (clear message if not, e.g. Colab without GPU runtime)
- `cudaGetDeviceProperties` — name, compute capability, SM count, limits.
- `cudaDeviceGetAttribute` — memory clock and bus width (removed from the properties struct in
  CUDA 13, so we use the attribute API, which works on all versions).
- `cudaMemGetInfo` — free/total memory; benchmarks skip sizes that don't fit.
- `cudaRuntimeGetVersion` / `cudaDriverGetVersion` — e.g. `12040` means 12.4.

**Compute capability** (e.g. 7.5) is the GPU architecture's version number. It decides which
features exist; for example FP16 Tensor Cores exist from 7.0 onward.

**Theoretical bandwidth** = 2 × memory clock × bus width in bytes. The 2 is because data moves twice
per clock ("double data rate"). Real kernels reach at most ~90% of it.

---

## 14. How the project is built (`CMakeLists.txt`)

- **CMake** generates the real build commands from a short description. `nvcc` (NVIDIA's compiler)
  compiles `.cu` files; `g++` compiles `.cpp` files.
- `CMAKE_CUDA_ARCHITECTURES native` — generate machine code for the GPU in this machine. GPU code
  is compiled for a specific architecture (e.g. `sm_75` for T4); the wrong architecture means the
  kernel cannot run ("no kernel image is available").
- `-lineinfo` — keeps source-line information in the GPU binary so Nsight Compute can show which
  line is slow. It does not change performance.
- `Release` — turns on optimizations (`-O3`). Never benchmark a Debug build.
- `ctp_core` — one static library with all kernels and helpers. **Compiling** turns each source
  file into an object file; **linking** combines object files (and the CUDA runtime library) into
  an executable.
- `add_test` registers tests so `ctest` runs them all.

---

## 15. File guide

### `kernels/vector_add.cu`
1. **What:** two vector-add kernels plus their launch functions. 2. **Why:** the simplest kernel, used to learn indexing and to measure pure memory bandwidth. 3. **Inputs:** device pointers `a`, `b`, length `n`, block size (and grid size for v2). 4. **Outputs:** `c` in device memory. 5. **Data flow:** global memory → registers (add) → global memory. 6. **Functions:** `vector_add_naive_kernel`, `vector_add_grid_stride_kernel`, `gpu::vector_add_naive`, `gpu::vector_add_grid_stride`, `gpu::vector_add_grid_stride_default_grid`. 7. **Variables:** `i`/`start` (global index), `stride` (threads in grid), `grid_size`. 8. **CUDA concepts:** `__global__`, `<<<>>>`, built-in index variables, occupancy API. 9. **Memory:** fully coalesced reads and writes, no reuse, so no shared memory needed. 10. **Thread mapping:** thread i ↔ element i (v1); thread t ↔ elements t, t+stride, … (v2). 11. **Sync:** none needed; threads never share data. 12. **Performance:** memory-bound; limit = bandwidth. 13. **Mistakes:** forgetting `if (i < n)`; integer division without rounding up; passing host pointers; block size not a multiple of 32 or > 1024.

### `kernels/vector_add.cuh`
Declares the launch functions so `.cu` programs (benchmarks, tests) can call them without seeing the kernels. Uses `namespace gpu` to separate them from `namespace cpu` versions with the same name.

### `src/cpu/cpu_ops.h / .cpp`
CPU reference implementations. Inputs/outputs: host pointers. Used as ground truth in tests and as the baseline in benchmarks. Written plainly (single thread). Compiled with `-O3`, so the compiler may vectorize the loop. That still makes it a fair single-core baseline.

### `src/utils/cuda_check.cuh`
`CUDA_CHECK(call)` and `CUDA_CHECK_KERNEL()`. Input: a `cudaError_t`. Output: nothing, or a message and exit. Common mistake: not checking at all, or only checking at the end, so the message points to the wrong place.

### `src/utils/device_buffer.cuh`
RAII owner of GPU memory, `DeviceBuffer<T>`. Covered in §12. Common mistake: dereferencing `data()` on the CPU. It is a GPU address.

### `src/utils/timer.cuh`
`CpuTimer` (steady_clock wall time) and `GpuTimer` (CUDA events, RAII). Covered in §10.

### `src/utils/host_utils.h`
`fill_random` (fixed-seed `std::mt19937`, reproducible inputs) and `compare_arrays` (the same rule as `torch.allclose`: `|out−ref| ≤ atol + rtol·|ref|`; NaN always fails).

### `src/utils/device_info.h / .cu`
Runtime GPU detection, covered in §13.

### `src/benchmark/bench_utils.cuh`
`time_gpu_ms`, `time_cpu_ms`, `bandwidth_gbs`. These are **function templates** taking a lambda `fn`. A **lambda** (`[&] { … }`) is an unnamed function written inline; `[&]` means it can use the surrounding variables by reference.

### `src/benchmark/bench_vector_add.cu`
Experiment program. For each size: checks correctness, then measures CPU, naive, grid-stride, end-to-end and the deliberately wrong no-sync time, plus achieved bandwidth. Then sweeps block sizes. Skips sizes that need more than half the free GPU memory.

### `tests/test_vector_add.cu`
Tests sizes around warp (32) and block (256) boundaries, n = 0, and a large odd size, for 4 block sizes and 3 grid sizes. Output is pre-filled with −999 ("poison") so a skipped element is detected. Tolerance is **exactly 0**: IEEE-754 floating-point addition is *correctly rounded*, so CPU and GPU must give bit-identical results for a single `a + b`. (Later operations like sums of many numbers will need a tolerance, because the order of additions changes the rounding.)

---

## 16. Common mistakes (Phase 1)

1. Missing bounds check `if (i < n)` → out-of-bounds writes, which may silently corrupt data.
2. `n / block_size` instead of ceiling division → the last partial block is never launched.
3. Passing host (CPU) pointers to a kernel → "illegal memory access".
4. Timing a kernel with a CPU timer without synchronizing → measures only the launch.
5. No warm-up → the first-run costs inflate the result.
6. Benchmarking a Debug build.
7. Not checking errors → the program prints wrong results with no error message.
8. Thinking "more threads = faster" → past the bandwidth limit, more threads change nothing.
9. Comparing kernel time with CPU time and calling it end-to-end speedup.

---

## 17. Interview explanation (say it in ~60 seconds)

> "I started with vector addition to set up the methodology. Each CUDA thread computes one
> element, using the global index `blockIdx.x * blockDim.x + threadIdx.x`, with a bounds check
> because n is not always a multiple of the block size. I also wrote a grid-stride version that
> launches just enough blocks to fill the SMs, using the occupancy API, and loops over the data.
> Vector add has an arithmetic intensity of 1/12 FLOP per byte, so it is memory-bound. I measured
> achieved bandwidth with CUDA events and compared it with the GPU's theoretical peak, which I read
> from the device at runtime. I also measured end-to-end time including PCIe copies, which showed
> that transfers dominate. That is why inference engines keep weights and activations resident on
> the GPU. Every result is first checked against a CPU reference."

---

## 18. What to remember

- **Index formula:** `i = blockIdx.x * blockDim.x + threadIdx.x` = threads in earlier blocks + my position.
- **Grid size:** `(n + block − 1) / block`, plus a bounds check.
- **Warp** = 32 threads executing in lockstep; the scheduler swaps warps to hide memory latency.
- **Coalesced** = neighboring threads access neighboring addresses.
- **Kernel launches are asynchronous.** Time kernels with CUDA events, after a warm-up.
- **Errors from a kernel's execution show up at the next synchronizing call.**
- **Vector add is memory-bound.** Judge it by GB/s against peak, not by FLOP/s.
- **PCIe copies are expensive.** Keep data on the GPU.
