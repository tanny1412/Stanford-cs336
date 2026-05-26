# Lecture 5 — CS336: GPUs

## Summary
GPU architecture internals — threads, cores, memory hierarchy, and why GPUs are so effective for deep learning.

---

## Key Concepts

### 1. What is a Thread

A thread = smallest unit of execution — one sequence of instructions running on a processor.

```
Thread: read value → multiply by 2 → store result
```

- CPU: 8-16 threads simultaneously (one per core)
- GPU: thousands of threads simultaneously

---

### 2. Threads in Matrix Multiply

Each output element of C = A × B needs a dot product. One thread per output element:

```
Thread 1  → computes C[0,0]
Thread 2  → computes C[0,1]
Thread 3  → computes C[1,0]
...
Thread N  → computes C[N,N]
```

All threads run simultaneously → entire matrix computed in parallel. That's why matrix multiplies are so fast on GPUs.

---

### 3. CPU vs GPU Cores

One core = one thread running at a time:
```
CPU: 8-16 cores   → 8-16 threads simultaneously
GPU: thousands of cores → thousands of threads simultaneously
```

**CPU cores** — powerful, complex, fast clock speed. Good at complex logic, branching, sequential tasks.

**GPU cores** — simple, slow individually. Only good at doing the same simple operation repeatedly. Massive total throughput from sheer number.

```
CPU: few powerful workers  → great for complex varied tasks
GPU: thousands of simple workers → great for same operation on lots of data
```

Matrix multiply = same operation on lots of data → perfect fit for GPU.

---

### 4. SMs and SPs

**SP (Streaming Processor)** = one GPU core. Does one floating point operation at a time. Also called a CUDA core.

**SM (Streaming Multiprocessor)** = group of SPs + shared resources:
```
SM
├── 128 SPs (CUDA cores)
├── Shared memory (fast, small)
├── Registers
└── Warp schedulers
```

H100: 132 SMs × 128 SPs = ~16,896 CUDA cores total.

**Hierarchy:**
```
GPU
└── many SMs
    └── each SM has many SPs
        └── each SP runs one thread
```

---

### 5. Blocks, Warps, and Shared Memory

**Block** = group of threads that run on one SM and share fast shared memory. Threads in different blocks can't share memory or synchronize.

Each block gets its own private slice of the SM's shared memory:
```
SM shared memory: 48KB total
  Block 1 → its own slice
  Block 2 → its own slice
```

All blocks execute the **same kernel** — same operation on different chunks of data. Different operations run sequentially (one kernel finishes, next launches).

---

### 6. Warps

**Warp = 32 threads that run truly simultaneously in lockstep** — all execute the exact same instruction at the same clock cycle, just on different data.

```
Block of 256 threads = 8 warps of 32 threads each

SM with 128 SPs runs 4 warps simultaneously (4 × 32 = 128)
Other warps wait in queue
```

Hierarchy:
```
Block → divided into warps (32 threads each)
Warp  → actual scheduling unit that runs on SPs
SP    → executes one thread from the active warp
```

Warps belong to one block — a warp never crosses block boundaries. But the warp scheduler picks from all warps across all blocks on the SM.

```
SM
├── Block 1 (its own shared memory)
│   ├── Warp 1, Warp 2, Warp 3
└── Block 2 (its own shared memory)
    ├── Warp 4, Warp 5, Warp 6
```

Scheduler can switch between any of these warps to keep SPs busy.

---

### 7. Warp Scheduler

Hardware inside the SM — runs automatically, not programmable.

Logic:
```
look at all warps on this SM
→ which are READY (not waiting for memory)?
→ pick one → run it on the SPs
```

Having more threads than SPs is intentional — when one warp stalls on memory, scheduler instantly switches to another ready warp. Keeps SPs busy.

**Occupancy** = how many warps are active on an SM. Higher occupancy = more choices for scheduler = SPs stay busier.

As a programmer you control: threads per block, number of blocks, what each thread does. Hardware handles the rest.

---

### 8. GPU Memory Hierarchy

One SM has one pool of shared memory divided among blocks running on it:
```
SM → one pool (e.g. 48KB total)
  Block 1 gets a slice
  Block 2 gets a slice
  Block 3 gets a slice
```
More shared memory per block = fewer blocks fit on SM at once.

**Full hierarchy (fastest → slowest):**
```
Registers:     ~1 cycle
Shared Memory: ~5 cycles
L1:            ~30 cycles
L2:            ~200 cycles
HBM:           ~600 cycles
```

**Intuition for each level:**

**Registers** — each thread's own scratchpad. Data is right there on the SP itself — no distance to travel, zero wait time.

**Shared Memory** — threads in a block often work on the same data. Without it, every thread fetches the same data from HBM separately — paying 600 cycles each. Shared memory means one thread fetches it once, stores it close by, all threads grab it instantly.

**L1 Cache** — automatic version of shared memory. Hardware automatically keeps recently used HBM data close to the SM. Access same data twice → second access is fast because L1 caught it. No programmer control needed.

**L2 Cache** — reuse layer across SMs. If SM 1 already paid 600 cycles to fetch data from HBM, it stays in L2. SM 2 needs same data → pays 200 cycles instead of 600. Without L2, every SM independently fetches same data from HBM — paying full cost every time.

**HBM** — where everything lives permanently. Huge but far away. Everything else in the hierarchy exists just to avoid going here as much as possible.

**One line:** data that was recently used or shared will probably be needed again — so keep it closer. That's the whole memory hierarchy.

---

### 9. SRAM vs DRAM

SRAM/DRAM = underlying hardware technology, not separate memory levels.

**SRAM (Static RAM)** — fast, expensive, takes more space per bit.
Used for: registers, shared memory, L1, L2.

**DRAM (Dynamic RAM)** — slow, cheap, dense.
Used for: HBM, regular CPU RAM.

Mapped to hierarchy:
```
Registers     → SRAM
Shared Memory → SRAM
L1 Cache      → SRAM
L2 Cache      → SRAM
HBM           → DRAM (stacked for high bandwidth)
```

**Why HBM = "High Bandwidth Memory":**
Regular DRAM is slow. HBM stacks DRAM chips vertically with thousands of wires — each bit still slow to access but many bits read simultaneously → very high total bandwidth.

---

### 10. L2 Cache — One Shared Across All SMs

```
GPU
├── SM 1 (own L1 + shared mem)
├── SM 2 (own L1 + shared mem)
...
├── SM 132
└── L2 Cache (one, shared by all SMs)
        ↓
      HBM (80GB)
```

Memory lookup order for a thread:
1. Registers → not there
2. L1 (own SM's cache) → not there
3. L2 (shared across all SMs) → not there
4. HBM → slowest, 600 cycles

L2 benefit: if one SM fetched data from HBM, it stays in L2 → other SMs grab it from L2 instead of going to HBM again.

---

### 11. Summary — Why Blocks and Warps Exist

**Block** → benefit is shared memory across warps in that block (thread cooperation)

**Warp** → benefit is keeping SPs busy via warp scheduler (hide memory latency)

Each SM has its own warp scheduler managing its own warps independently:
```
SM 1: own warp scheduler → keeps its 128 SPs busy
SM 2: own warp scheduler → keeps its 128 SPs busy
```

---

### 12. Making ML Workloads Fast — Compute Intensity, Tiling, Wave Quantization

GPU performance for matrix multiply is not simple — three key concepts:

**Compute Intensity (small matrices):**
Small matrices don't have enough work to keep all SPs busy — most SPs sit idle. Bottlenecked by not having enough compute, not by memory speed. Performance is low and grows slowly with matrix size.

**Tiling (the jumps upward):**
Split matrix into chunks (tiles) that fit in shared memory. Load a tile → compute on it → move to next tile. Each tile reuses data from shared memory instead of going back to HBM repeatedly.
- Without tiling: every thread fetches from HBM → slow
- With tiling: data loaded once into shared memory, reused by all threads in block → fast
Performance jumps dramatically at certain sizes when tiling kicks in.

**Wave Quantization (the drops at 128, 2048 etc.):**
GPUs process work in "waves" — all SMs working together on one wave of blocks. If matrix size doesn't perfectly fill a wave, last wave is only partially full → some SMs sit idle → performance drops. The dips in the chart happen at these boundary sizes.

**Takeaway:** even for a simple matrix multiply, peak GPU performance requires understanding tiling and wave quantization. This is why writing efficient GPU kernels (like FlashAttention) is a whole engineering discipline.

---

### 13. How to Make GPUs Go Fast — 6 Techniques

**1. Control Divergence**
Warps run 32 threads in lockstep — all must execute the same instruction. If threads take different branches (if/else), warp runs both branches sequentially with half threads idle each time → wasted SPs.
Fix: keep all 32 threads doing the same thing — avoid branching within a warp.

**2. Low Precision Computation**
bf16/fp16 instead of fp32:
- Half the memory → more fits in cache → fewer HBM reads
- GPU tensor cores do bf16 matrix multiplies much faster than fp32
Why mixed precision exists — not just memory savings but raw compute speed.

**3. Operator Fusion**
Instead of separate kernels writing to HBM between each operation:
```
without fusion: LayerNorm → HBM → ReLU → HBM → Dropout
with fusion:    LayerNorm → ReLU → Dropout  (one kernel, stays in shared memory)
```
Fewer HBM round trips = much faster. FlashAttention is a famous example.

**4. Recomputation**
Backprop needs forward pass activations. Normally stored in HBM — expensive.
Recomputation = discard activations during forward pass, recompute during backward pass when needed.
Trades extra compute for less memory. Same idea as gradient checkpointing.

**5. Coalescing Memory**
Threads in a warp should read consecutive memory addresses so GPU fetches them in one transaction:
```
Good (coalesced): Thread 1 → addr 0, Thread 2 → addr 1, Thread 3 → addr 2 ...  (1 transaction)
Bad (scattered):  Thread 1 → addr 0, Thread 2 → addr 512 ...  (multiple transactions)
```
Scattered reads = multiple slow HBM transactions. Coalesced = one fast transaction for whole warp.

**6. Tiling**
Split matrix into tiles that fit in shared memory. Load tile once → compute on it → move to next tile. Avoids repeated HBM reads by reusing data in shared memory.

---

### 14. torch.compile and Operator Fusion

`torch.compile` does operator fusion automatically — looks at sequence of operations, figures out which can be fused, compiles them into one optimized kernel instead of separate kernels with HBM round trips between them.

**Two ways to get fusion:**

**1. torch.compile** — automatic. Write normal PyTorch, compile adds fusion on top. Convenient but not always optimal — can't always find the best fusion strategy. ~20-30% speedup for free.

**2. Hand-written CUDA kernels (e.g. FlashAttention)** — manual. Write a custom kernel with perfect fusion, tiling, and memory coalescing by hand. Much more performant but extremely hard to write. 5-10× speedup but weeks of engineering work.

**FlashAttention example:**
Naive attention: QK^T → write to HBM → softmax → write to HBM → ×V
FlashAttention: entire attention in one fused kernel, never writes intermediate results to HBM → 5-10× faster.

---

### 15. Memory Coalescing and DRAM Burst Mode

DRAM reads in **burst mode** — reading one address delivers the entire burst section it belongs to:
```
Burst section 0: addresses 0, 1, 2, 3
Burst section 1: addresses 4, 5, 6, 7
```
Read address 1 → automatically get 0, 1, 2, 3. Burst sections are 128 bytes on real GPUs.

**Why coalescing matters:**

Coalesced (consecutive addresses):
```
Thread 1 reads addr 0
Thread 2 reads addr 1  → all in same burst section → ONE read → all threads satisfied
Thread 3 reads addr 2
Thread 4 reads addr 3
```

Scattered (random addresses):
```
Thread 1 reads addr 0   → burst section 0
Thread 2 reads addr 8   → burst section 2   → 4 separate reads, massive waste
Thread 3 reads addr 4   → burst section 1
Thread 4 reads addr 12  → burst section 3
```

Scattered = pay full cost of multiple bursts but use only 1 value from each → wasted bandwidth.

**The goal:**
```
Best case:  32 threads → all in 1 burst section → 1 read
Worst case: 32 threads → all different burst sections → 32 reads
```

128 byte burst ÷ 4 bytes per fp32 value = 32 values per burst = perfect fit for one warp of 32 threads reading consecutive addresses.

Arrange data so a warp's 32 threads read consecutive addresses → fit into one burst → one read → done.

---

### 16. Should You Learn CUDA?

No — not worth it unless specifically going into GPU kernel engineering (very niche role).

CUDA is C++ based, very low level, takes months to get productive in.

**Better alternatives given Python background:**

- **Triton** — Python-based GPU programming by OpenAI. Write GPU kernels in Python-like syntax, handles CUDA underneath. Right tool if you ever need custom kernels.
- **torch.compile** — one line of code, free 20-30% speedup, just use it.
- **Conceptual understanding** — knowing *why* FlashAttention is fast without writing it is already valuable.

**Realistic path:**
```
Python → PyTorch → GPU concepts (done) → Triton if needed → leave CUDA to specialists
```

---

### 17. Tiling — How It Works

**Problem without tiling:**

Each output element C[i,j] needs a dot product of row i of A and col j of B — reads from HBM every time:
```
Thread for C[0,0] → reads row 0 of A from HBM, col 0 of B from HBM
Thread for C[0,1] → reads row 0 of A from HBM again, col 1 of B from HBM
Thread for C[0,2] → reads row 0 of A from HBM again, col 2 of B from HBM
```
Row 0 of A read from HBM repeatedly — massive waste.

**The insight:** multiple output elements share the same input data. Load it once into shared memory and reuse.

**How tiling works:**

Split A and B into small tiles (T×T) that fit in shared memory. For each output tile of C, loop over tiles of A and B:
```
Step 1: Load tile of A into shared memory   ← one HBM read
        Load tile of B into shared memory   ← one HBM read
Step 2: All threads compute using shared memory  ← fast (5 cycles)
Step 3: Move to next tile, repeat
```

**The memory math:**

Without tiling (N×N matrix):
```
Total HBM reads = N² × 2N = 2N³
```

With tiling (tile size T):
```
Total HBM reads = 2N³/T
```
Tiling reduces HBM reads by factor of T — bigger tiles = fewer HBM reads.

**Why shared memory is the key:**
```
Without tiling: every thread → HBM for every value → 2N³ slow reads
With tiling:    load tile once → shared memory → all threads reuse → 2N³/T HBM reads
```
Shared memory = manually managed cache. You explicitly load what you need, use it many times, load the next thing.

---

### 18. Tiling — The Partial Accumulation Pattern

Total HBM reads don't go to zero — but instead of each thread going to HBM individually, threads share each read:

```
1. Load tile from HBM into shared memory   ← one HBM read for whole tile
2. All threads do partial multiply using shared memory  ← fast
3. Accumulate into partial result
4. Load next tile from HBM
5. Add to partial result
6. Repeat until all tiles done
7. Final output = sum of all partial results
```

Key: each tile gives a **partial dot product**. Keep adding tile by tile until full row/column is covered. Final output = accumulated sum of all partial results.

HBM reads: `2N³` → `2N³/T` — same total work but T threads share each HBM read instead of each paying separately.

---

### 19. Tiling Complexity — Memory Alignment

Tiles only load fast when they align with burst section boundaries.

**Aligned layout (good):**
Tile perfectly matches burst section boundaries → loading tile = reading complete burst sections → one clean read per row of tile.

**Unaligned layout (bad):**
Tile starts in middle of burst section → need to read TWO burst sections per row → fetch way more data than needed → twice as many HBM reads.

**Fix — padding:**
Add extra columns so matrix dimensions become a multiple of 32 (128 bytes ÷ 4 bytes per fp32 = 32 values per burst). Padding values never used but force alignment so every tile loads cleanly.

This is why matrix dimensions in ML code are always multiples of 64, 128, or 256 — not random, it's for memory alignment and efficient burst reads.

---

### 20. Online Softmax

Normal softmax needs full row first:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)   ← need all x_j to compute sum
```
Problem: full row can be huge (sequence_length values) — must load everything into memory before computing anything.

**Online softmax:** process one tile at a time, keeping two running values:
```
m = running maximum so far   (for numerical stability)
d = running sum of exp values
```

For each new tile:
```
1. Update m → new max if current tile has larger values
2. Rescale old d → max changed so old exp values need adjusting
3. Add new exp values to d
```

At the end, `d` holds correct denominator for full row — without ever loading the whole row at once.

**Why it matters — FlashAttention:**
FlashAttention uses online softmax to compute attention tile by tile without materializing the full N×N attention matrix in HBM. QK^T scores for one tile → shared memory → online softmax → accumulate weighted values → next tile. Full attention computed staying in shared memory, never writing giant matrix to HBM. That's the core trick behind FlashAttention's speed.

---

## Questions / Follow-up
