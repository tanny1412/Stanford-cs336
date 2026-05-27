# Lecture 6 — CS336: Kernels, Triton

## Summary
GPU kernel execution model — wave quantization, and how hardware and execution interact.

---

## Key Concepts

### 1. Waves and Wave Quantization

**Wave** = one round of blocks distributed across all SMs simultaneously.

```
132 SMs, 200 blocks launched:
Wave 0: 132 blocks → all 132 SMs busy → runs to completion
Wave 1: 68 blocks  → only 68 SMs busy, 64 SMs sit idle (the "tail")
```

**Wave Quantization** = performance drop caused by the last partial wave. Last wave is only partially full → half the GPU idle for the entire duration of that wave. GPU doesn't finish faster just because fewer blocks run — same time, wasted SMs.

**Fix:** make number of blocks divisible by number of SMs — every wave completely full:
```
132 SMs → use 132, 264, 396 blocks (multiples) → no partial tail wave
```

Rule of thumb: number of blocks ≥ 4× number of SMs. More waves = tail wave matters less as fraction of total work.

This is why matrix dimensions are multiples of 64/128/256 — controls how many blocks launch and whether they divide evenly into full waves.

---

### 2. Benchmarking vs Profiling

**Benchmarking** — measures how long something takes end to end:
```
start timer → run operation → stop timer → report time
```
Tells you: "this forward pass takes 12ms." Doesn't tell you why.

**Profiling** — measures where time is being spent inside the operation:
```
LayerNorm:  2ms  (17%)
Attention:  7ms  (58%)
FFN:        3ms  (25%)
```
Tells you exactly which part is the bottleneck → where to optimize.

**Why not trust spec sheets:**
Advertised GPU specs assume perfect conditions. Your actual workload has specific batch sizes, sequence lengths, library versions — performance varies wildly from the spec. Only measuring your actual code tells the truth.

Rule: every time you make a change, benchmark and profile. Otherwise you're guessing whether your optimization actually helped.

---

### 3. Warmup Iterations and torch.cuda.synchronize

**Warmup iterations** — run operation a few times before timing:

First run is artificially slow due to:
- JIT compilation — kernel compiled on first run
- Memory allocation — GPU allocates buffers
- Cache cold start — caches empty, data fetches go all the way to HBM

After warmup: compilation cached, memory allocated, caches warm → timing reflects real steady state performance.

**torch.cuda.synchronize()** — GPU runs asynchronously. CPU launches kernel and returns immediately while GPU still computes in background. Without synchronize, timer measures "time to launch kernel" not "time to finish kernel." synchronize() blocks until GPU actually finishes → accurate timing.

**Why synchronize twice:**
```
run()
synchronize()        ← ensure warmup fully done before clock starts

start_time = time.time()
run()
synchronize()        ← ensure actual run fully done before clock stops
stop_time = time.time()
```

First synchronize: prevents warmup work overlapping with timed run.
Second synchronize: ensures you measure full execution time, not just launch time.

---

### 4. PyTorch Profiler

PyTorch built-in profiler wraps your code and tracks every operation:
```
Name          → which operation (LayerNorm, matmul etc.)
Self CPU %    → % of time in this op itself (excluding children)
Self CPU      → actual time in this op
CPU total %   → % including all sub-operations it calls
CPU total     → total time including sub-ops
CPU time avg  → average per call
# of Calls    → how many times called
```

**Self CPU vs CPU total:**
If LayerNorm calls mean and std internally:
- Self CPU = time in LayerNorm's own code only
- CPU total = LayerNorm + mean + std combined

---

### 5. Profiling a Simple Add — What Actually Happens

Profiling `a + b`:
```
aten::add                  98%   1.392ms  ← PyTorch dispatch overhead
vectorized_elementwise_kernel  0%         ← actual CUDA kernel (runs on GPU, not CPU)
cudaLaunchKernel           1.37% 19us     ← CPU time to tell GPU to run the kernel
cudaDeviceSynchronize      0.62%          ← waiting for GPU to finish
```

**Key insight:** `aten::add` shows 98% CPU time but that's just PyTorch overhead (Python → C++ → dispatch). The actual GPU kernel shows 0% CPU time because it runs on the GPU — CPU just launches it and moves on.

**cudaLaunchKernel takes 19 microseconds** just to launch — fixed cost regardless of operation size. For tiny operations, launch overhead dominates. This is exactly why operator fusion matters — fuse 10 small ops into 1 kernel = pay launch cost once instead of 10 times.

---

### 6. CPU vs CUDA Time — The Full Picture

Profiling `a + b` with both CPU and CUDA columns:
```
aten::add:
  Self CPU  = 1.392ms   (98% — PyTorch dispatch overhead)
  Self CUDA = 17.119us  (actual GPU execution)

vectorized_elementwise_kernel:
  Self CPU  = 0us       (runs on GPU, no CPU time)
  Self CUDA = 17.119us  (where GPU work actually happens)

cudaLaunchKernel:
  Self CPU  = 19.392us  (CPU overhead to launch)
  Self CUDA = 0us       (just a launch call, no GPU work)
```

**The gap:**
```
CPU time total:  1.420ms
CUDA time total: 17.119us
```

CPU spends 1.4ms on overhead (Python → C++ → PyTorch dispatch). GPU does actual work in 17 microseconds. That's 80× more time on CPU dispatch than actual GPU compute.

For large matrix multiplies, GPU time dominates and overhead doesn't matter. For small ops, you're paying mostly overhead — why batching small operations and operator fusion are so important.

---

### 7. Relationship Between aten::add and cudaLaunchKernel

`aten::add` and `vectorized_elementwise_kernel` show same CUDA time (17us) — they're the same GPU execution seen at two levels. Parent op and the child kernel it launched.

`aten::add` Self CPU (1.392ms) ≠ `cudaLaunchKernel` Self CPU (19us) — these are different things:

```
aten::add (1.392ms total CPU)
├── Python → C++ dispatch
├── find right kernel
├── allocate outputs
├── cudaLaunchKernel (19us)  ← just this tiny piece
└── other overhead
```

1.392ms = full cost of the whole operation. 19us = just the specific moment of telling GPU to run the kernel. cudaLaunchKernel is one small sub-step inside aten::add's total CPU time.

---

### 8. How PyTorch matmul Actually Works — Full Stack

```
Python (PyTorch) — torch.matmul(A, B)
      ↓
C++ (PyTorch internals — aten dispatch)
      ↓
cuBLAS / cuDNN / CUTLASS  ← pre-written optimized CUDA libraries
      ↓
CUDA kernel runs on GPU SMs
```

**For matmul:** `torch.matmul` dispatches through C++ to **cuBLAS** — NVIDIA's highly optimized matrix multiply library. Contains hand-tuned CUDA kernels using tiling, memory coalescing, tensor cores — written by NVIDIA engineers over years.

**What profiler exposes:**
```
torch.matmul (Python)
      ↓
aten::mm (C++ dispatch)
      ↓
cublas_gemm_kernel (actual GPU kernel from cuBLAS)
```

All layers visible with separate CPU and CUDA times.

**Where custom kernels fit:**
If no library exists for your operation (e.g. FlashAttention didn't exist yet), you write your own CUDA/Triton kernel at the same bottom layer — replacing or supplementing what NVIDIA provides.

---

### 9. Why Two aten Entries for matmul

```
aten::matmul  ← high level dispatcher, handles any shape/dimension
      ↓
aten::mm      ← lower level 2D matrix multiply (actual implementation)
      ↓
cublas_sgemm_kernel  ← actual GPU kernel (4.992us CUDA time)
```

`aten::matmul` barely does any work (1.17% Self CPU) — just looks at input shapes and dispatches to the right op. For 2D matrix → calls `aten::mm`. For batched → calls something else.

`aten::mm` does the real setup work (42% Self CPU).

Three levels for what feels like one operation in Python.

---

### 10. The Full Stack — C++, CUDA C++, and Triton

CUDA kernels are written in **CUDA C++** (not plain C) — C++ with extra GPU syntax (`__global__`, `threadIdx`, `blockIdx`). Compiled by NVIDIA's nvcc compiler to GPU machine code.

```
Python (PyTorch)          → .py files
      ↓
C++ (aten dispatch)       → .cpp files
      ↓
CUDA C++ (cuBLAS kernels) → .cu files (closed source, NVIDIA)
      ↓
PTX (GPU assembly)
      ↓
GPU machine code
```

**With Triton (corrected compilation chain):**
```
Python (PyTorch)
      ↓
C++ (aten dispatch)
      ↓
Triton kernel (.py)       ← you write this
      ↓
LLVM IR
      ↓
PTX (GPU assembly — via LLVM NVPTX backend)
      ↓
GPU machine code (via ptxas)
```

Triton compiles directly to PTX via LLVM — skips CUDA C++ entirely. PTX = NVIDIA's assembly language for GPUs. LLVM is a powerful compiler so this often produces better optimized code than going through CUDA C++.

Write Triton kernel, register it with PyTorch → C++ dispatcher calls your kernel instead of default cuBLAS.

FlashAttention works exactly this way — Triton kernel registered into PyTorch so `F.scaled_dot_product_attention` dispatches to FlashAttention instead of naive PyTorch implementation.

---

### 11. Profiling Composite Operations — cdist Example

`cdist` computes pairwise distances between vectors. Profiler reveals what's actually happening:

```
aten::cdist                  ← top level dispatcher
      ↓
aten::_euclidean_dist        ← euclidean distance computation
      ↓
aten::matmul                 ← cdist uses matrix multiply internally!
      ↓
aten::mm                     ← 2D matmul
      ↓
cublas_sgemm_kernel          ← actual GPU work (343us CUDA time)
```

**Why cdist uses matmul:**
Euclidean distance reformulated as:
```
||a - b||² = ||a||² + ||b||² - 2a·b
```
The `a·b` term is a matrix multiply — turns distance computation into a GPU-friendly matmul instead of computing each pair separately. Classic math trick.

Extra ops at bottom (`CopyArrayBatchedD2D`, `vectorized_elementwise_kernel`) handle the `||a||²` and `||b||²` terms.

**Key lesson:** what looks like one simple function in Python is actually several operations under the hood — matmul, copies, element-wise ops all chained. Profiler is the only way to see this.

---

### 12. Autograd — What It Actually Is

**Autograd** = PyTorch's automatic differentiation engine. Automatically computes gradients during backward pass.

During forward pass: autograd silently records every operation into a **computation graph**.
During `loss.backward()`: walks graph in reverse, applies chain rule, calls appropriate CUDA kernels.

```
Forward pass:
x → matmul → ReLU → LayerNorm → loss
(autograd records every op into graph)

Backward pass (loss.backward()):
autograd walks graph in reverse
      ↓
needs gradient of LayerNorm → calls LayerNorm backward CUDA kernel
      ↓
needs gradient of ReLU → calls ReLU backward CUDA kernel
      ↓
needs gradient of matmul → calls cublas backward kernel
      ↓
dL/dW ready for optimizer
```

Autograd = traffic controller/orchestrator. Knows the graph, knows the math rule for each op's backward, dispatches right CUDA kernels in right order. Actual compute still happens in CUDA kernels on GPU.

Without autograd: manually write backward pass for every operation yourself.

---

### 13. NVIDIA Nsight Systems vs PyTorch Profiler

**PyTorch profiler:**
- Shows Python/C++ level operations
- Good for seeing which PyTorch ops are slow
- Easy to use, built into PyTorch

**NVIDIA Nsight Systems:**
- Goes all the way down to GPU hardware level
- Shows what's happening on every SM, every warp, every memory transfer
- GPU utilization, memory bandwidth, kernel occupancy, PCIe transfers
- Timeline view — see exactly when each kernel runs, when GPU is idle, where bottlenecks are

```
PyTorch profiler → what is slow (operation level)
Nsight Systems   → why it is slow (hardware level)
```

Nsight tells you: was it memory bandwidth limited? compute limited? SMs underutilized? wave quantization issue?

Used by people writing custom CUDA/Triton kernels squeezing out every last bit of performance. For most ML work, PyTorch profiler is enough — Nsight is for deep kernel optimization.

---

### 14. Why Print Statements Kill GPU Performance

Normally CPU queues many GPU kernels ahead of time without waiting:
```
CPU: queue kernel 1 → queue kernel 2 → queue kernel 3 → ...
GPU:                  run kernel 1   →  run kernel 2  → ...
```
GPU always has work ready — no idle time.

**With a print statement:**
```python
output = matmul(A, B)
print(output)       ← CPU must wait for GPU to finish and copy data back
next_op(output)
```
To print, CPU needs the actual value → must wait for GPU to finish → transfer data back to CPU memory → entire pipeline stalls:
```
CPU: queue kernel 1 → WAIT for GPU → print → queue kernel 2
GPU: run kernel 1   → idle         →       → run kernel 2
```

GPU sits completely idle while CPU prints. Nsight shows this as a big gap in the GPU timeline — no kernels running.

Never put print statements inside training loops — even one print per step can cut GPU utilization significantly.

---

### 15. CDF of the Gaussian

**CDF (Cumulative Distribution Function)** = probability that a value falls below a certain point.

For standard normal (mean=0, std=1):
```
CDF(-∞) = 0    (nothing below -infinity)
CDF(0)  = 0.5  (50% of values below the mean)
CDF(+∞) = 1    (everything below +infinity)
```

PDF = bell curve. CDF = running area under that bell curve from left to right (S-shaped curve).

**Why it comes up in ML — GeLU:**
```
GeLU(x) = x × Φ(x)    where Φ(x) = Gaussian CDF
```
- Very negative x → Φ(x) ≈ 0 → output ≈ 0
- Very positive x → Φ(x) ≈ 1 → output ≈ x
Smooth transition between the two — the smooth gating behavior of GeLU.

---

### 16. CUDA Grid → Block → Thread Hierarchy

**Why we need Grid:**
Block has a size limit (max 1024 threads). Large matrix multiply needs millions of threads — too many for one block. Grid = collection of all blocks for one kernel launch.

```
Grid (entire job)
└── many Blocks (chunks of the job, each runs on one SM)
    └── many Threads (individual workers, max 1024 per block)
```

Grid size scales with job size:
```
small matrix (128×128)   → few threads needed   → grid has few blocks
large matrix (4096×4096) → millions of threads  → grid has many blocks
```

**Launching a kernel:**
```cuda
dim3 numBlocks(...)      ← calculated based on problem size
dim3 blockSize(256)      ← threads per block (max 1024)
myKernel<<<numBlocks, blockSize>>>(...)
```

GPU automatically distributes blocks across all SMs. One thread per output element typically.

**How each thread knows what to compute:**
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x
```
Each thread computes its unique index from block position + thread position within block → knows which output element to compute.

---

### 17. CUDA Kernel — GeLU Example

```cuda
__global__ void gelu_kernel(...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // unique global index per thread
    if (i < num_elements) {                           // guard for extra threads
        out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}
```

- `__global__` — runs on GPU, called from CPU
- Each thread computes its unique index i → computes GeLU for one element → writes to out[i]
- Guard needed because ceiling division launches more threads than elements

**Kernel launch:**
```cpp
gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements)
```
- `<<<num_blocks, block_size>>>` — grid config (CUDA syntax)
- `.data_ptr<float>()` — raw GPU memory pointer (kernel can't take PyTorch tensor directly)
- `torch::Tensor x` — C++ type declaration, same as Python type hint `x: torch.Tensor`

---

### 18. Triton Kernel — GeLU Example (Full Breakdown)

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
```
- `@triton.jit` — compile this Python function into a GPU kernel
- `x_ptr, y_ptr` — raw pointers to input/output in GPU memory
- `BLOCK_SIZE: tl.constexpr` — compile-time constant, how many elements per block

```python
pid = tl.program_id(axis=0)
```
- `pid` = program ID = block ID (equivalent to `blockIdx.x` in CUDA)
- Each block gets unique pid: 0, 1, 2, 3...

```python
block_start = pid * BLOCK_SIZE
```
- Where this block's data starts in the array
- Block 0 → element 0, Block 1 → element BLOCK_SIZE, Block 2 → element 2×BLOCK_SIZE

```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```
- `tl.arange(0, BLOCK_SIZE)` = `[0, 1, 2, ..., BLOCK_SIZE-1]` — a vector
- `offsets` = all element indices this block will process simultaneously
- **Key Triton difference:** block handles a whole VECTOR of elements, not one at a time

```python
mask = offsets < num_elements
```
- Guard for elements beyond actual data size (same as `if i < num_elements` in CUDA)

```python
x = tl.load(x_ptr + offsets, mask=mask)
```
- Load ALL elements at offsets in one vectorized memory read
- masked positions ignored

```python
a = 0.79788456 * (x + 0.044715 * x * x * x)
```
- GeLU formula applied to entire vector `x` at once — not a single number, a vector of BLOCK_SIZE numbers

**CUDA vs Triton mental model:**
```
CUDA:   each thread = one element
        thread: compute index i → load in[i] → compute → store out[i]

Triton: each block = vector of BLOCK_SIZE elements
        block: compute offsets vector → load vector → compute on whole vector → store vector
```

Triton thinks in vectors, not individual threads. Thread-level parallelism handled automatically by Triton — you just write logic for one block handling many elements. That's why Triton is easier than CUDA.

---

### 19. Triton vs CUDA — Which to Learn

Triton gives 90% of CUDA's power with 20% of the complexity.

**What Triton handles automatically:**
- Thread-level parallelism inside blocks
- Most shared memory management
- Low-level indexing

**What you'd miss skipping CUDA:**
- Fine-grained shared memory control
- Very exotic hardware optimizations
- Working at NVIDIA or on CUDA compiler itself

**For everything else** — custom attention kernels, fused operations, inference optimization — Triton is what people actually use. Even researchers at top labs prefer Triton over raw CUDA for new kernels — faster to write and debug.

**Practical approach:** understand CUDA conceptually (done) → write in Triton when you need custom kernels.

---

### 20. Triton Learning Roadmap (After This Course)

Prerequisites already covered: threads/blocks/warps/SMs, memory hierarchy, tiling, coalescing, operator fusion.

**Step-by-step Triton course:**
1. **Vector operations** — write GeLU, ReLU, elementwise add (already read GeLU kernel)
2. **Reduction operations** — softmax, LayerNorm (introduces tiling across a row)
3. **Matrix multiply** — write your own matmul (tiling really clicks here)
4. **Fused operations** — fuse two ops into one kernel (e.g. linear + ReLU)
5. **Attention** — read FlashAttention Triton implementation line by line

Each step = one project. By step 5: understand FlashAttention at implementation level, not just conceptually.

---

### 21. torch.compile and Tiling — Where to Put Them

**torch.compile** — just wrap your model/function, handles operator fusion automatically:
```python
model = torch.compile(model)
compiled_fn = torch.compile(my_function)
```
No other changes needed. Analyzes computation graph, finds fusable ops, generates optimized kernels.

**Tiling in normal PyTorch** — handled automatically inside cuBLAS/Triton kernels when you call `torch.matmul`. Tile sizes chosen by library based on matrix dimensions and GPU architecture. You don't control this.

**Tiling in custom Triton kernel** — you control `BLOCK_SIZE` yourself:
```python
BLOCK_SIZE = 1024  # your tile size
```
Too small = not enough memory reuse. Too large = doesn't fit in shared memory.

**Practical summary:**
```
Normal PyTorch     → cuBLAS handles tiling automatically
torch.compile      → handles operator fusion automatically
Custom Triton      → you control BLOCK_SIZE yourself
```

For most use cases: just use `torch.compile` and let it handle everything.

---

### 22. Softmax in Triton — From Scratch

**What softmax does:**
```
input:  [1, 2, 3]
step 1: exp each → [2.71, 7.39, 20.09]
step 2: sum → 30.19
step 3: divide each by sum → [0.09, 0.24, 0.67]
```
Output sums to 1 — a probability distribution.

**Why softmax is a reduction:**
Every element needs the sum of the whole row before it can divide. Can't compute one output independently — must first collapse whole row into one number (sum). This is a reduction.

GeLU = each element independent. Softmax = all elements depend on each other.

**Why grid = rows:**
Each row is one independent softmax job. Row 0 has nothing to do with Row 1.
```
num_blocks = M    ← one block per row
block_size = N    ← one block handles all columns of that row
```
One block owns one entire row — loads all N values, computes sum (reduction), divides each value. Everything in shared memory within one block.

**The wrapper code:**
```python
M, N = x.shape
block_size = triton.next_power_of_2(N)  # round to nearest power of 2 for efficiency
num_blocks = M                           # one block per row

triton_softmax_kernel[(M,)](
    x_ptr=x, y_ptr=y,
    x_row_stride=x.stride(0),           # how many elements to skip to next row
    num_cols=N, BLOCK_SIZE=block_size
)
```

**Inside the kernel (conceptual):**
```python
row_start = pid * x_row_stride
offsets = row_start + tl.arange(0, BLOCK_SIZE)
mask = tl.arange(0, BLOCK_SIZE) < num_cols

x = tl.load(x_ptr + offsets, mask=mask)  # load whole row

x_max = tl.max(x, axis=0)               # reduction → one max value
x = x - x_max                           # subtract max for numerical stability

numerator = tl.exp(x)
denominator = tl.sum(numerator, axis=0) # reduction → one sum value
output = numerator / denominator

tl.store(y_ptr + offsets, output, mask=mask)
```

`tl.max` and `tl.sum` = reductions — collapse whole vector to one number, used by all elements.

**Why subtract max before exp:**
`exp(1000)` = infinity → overflow. Subtracting max keeps values safe:
```
softmax(x) = softmax(x - max(x))   ← mathematically identical, numerically stable
```
Same trick used in online softmax for FlashAttention.

---

## Questions / Follow-up
