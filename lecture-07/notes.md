# Lecture 7 — CS336: Parallelism 1

## Summary
Distributed training — how to split work across multiple GPUs and nodes.

---

## Key Concepts

### 1. Node vs Rack

**Node** = one physical machine with multiple GPUs:
```
Node = one server
  └── typically 4-8 GPUs (e.g. 8× H100s)
  └── CPUs, RAM, NVLink connecting GPUs internally
```

**Rack** = physical cabinet holding multiple nodes:
```
Rack
├── Node 1 (8× H100s)
├── Node 2 (8× H100s)
├── Node 3 (8× H100s)
...
```

**Why distinction matters:**
- Within node → GPUs connected via **NVLink** (~900 GB/s, very fast)
- Across nodes → machines connected via **InfiniBand** (~200 GB/s, much slower)

Communication within a node is ~4-5× faster than across nodes. Distributed training keeps frequently communicating things on the same node.

---

### 2. NVLink vs NVSwitch

**NVLink** = the physical high speed connection technology between chips (the road).

**NVSwitch** = a hub chip that uses NVLink ports to connect ALL GPUs to ALL other GPUs (the roundabout/hub).

Without NVSwitch — limited direct connections between GPU pairs, must hop through other GPUs:
```
GPU 0 ←→ GPU 1 ←→ GPU 2 ←→ GPU 3   (must hop, slow)
```

With NVSwitch — every GPU has direct path to every other GPU through the switch:
```
GPU 0 ←→ NVSwitch ←→ GPU 1
GPU 0 ←→ NVSwitch ←→ GPU 2   (one hop, full speed)
GPU 0 ←→ NVSwitch ←→ GPU 3
```

**H100 node:** 4 NVSwitches, each connected to all 8 GPUs. More switches = more parallel bandwidth paths = faster all-to-all communication.

NVLink still exists when NVSwitch is present — NVSwitch is built on top of NVLink:
```
GPU ←→ NVLink cables ←→ NVSwitch ←→ NVLink cables ←→ GPU
```

Analogy: NVLink = roads, NVSwitch = central hub airport. Every GPU connects to the hub, reaches any other GPU in one hop.

---

### 3. Communication Collectives

**1. Broadcast** — one GPU sends same data to all:
```
GPU 0: [1,2,3] → all GPUs get [1,2,3]
```
Used for: sending initial model weights to all GPUs at start of training.

**2. Scatter** — one GPU splits data, sends different chunks to each GPU:
```
GPU 0: [A,B,C,D] → GPU 0:A, GPU 1:B, GPU 2:C, GPU 3:D
```
Used for: distributing different data batches to different GPUs.

**3. Gather** — collect chunks from all GPUs onto one GPU:
```
GPU 0:A, GPU 1:B, GPU 2:C, GPU 3:D → GPU 0: [A,B,C,D]
```
Used for: collecting results from all GPUs onto one.

**4. Reduce** — combine same-shaped data from all GPUs into one result on one GPU:
```
GPU 0:[1,2], GPU 1:[3,4], GPU 2:[5,6], GPU 3:[7,8] → GPU 0: [16, 20]
```
Used for: summing gradients onto one GPU.

**5. All-Reduce** — same as reduce but result goes to ALL GPUs:
```
GPU 0:[1,2], GPU 1:[3,4], GPU 2:[5,6], GPU 3:[7,8] → ALL GPUs: [16, 20]
```
Most important collective for training. Each GPU has gradients for its batch → all-reduce sums them → every GPU has complete gradient → all update weights identically.
Used by: **DDP**.

**6. All-Gather** — each GPU has a chunk, every GPU ends up with all chunks:
```
GPU 0:A, GPU 1:B, GPU 2:C, GPU 3:D → ALL GPUs: [A,B,C,D]
```
Used by: **FSDP** — weights sharded across GPUs, all-gather reconstructs full weights before forward pass.

**7. Reduce-Scatter** — reduce AND scatter simultaneously, each GPU gets one chunk of reduced result:
```
GPU 0:[1,2], GPU 1:[3,4], GPU 2:[5,6], GPU 3:[7,8]
→ GPU 0: [16]  (sum of first elements)
→ GPU 1: [20]  (sum of second elements)
```
Used by: **FSDP** gradient sync — each GPU gets only the gradient chunk it's responsible for. Uses half the bandwidth of all-reduce.

**The big picture:**
```
DDP:   all-reduce gradients after every backward pass
FSDP:  all-gather weights before forward → reduce-scatter gradients after backward
```

---

### 4. TPUs vs GPUs — Communication Design

**TPU networking — Toroidal Mesh:**
Each chip connects to neighbors (left, right, up, down), edges wrap around like a donut:
```
Chip ←→ Chip ←→ Chip ←→ (wraps back)
 ↕         ↕        ↕
Chip ←→ Chip ←→ Chip ←→ (wraps back)
```
Data hops through intermediate chips to reach distant ones. Simple and cheap to build at scale but not all-to-all — distant chips have higher latency.

**GPU networking — All-to-all (up to 256):**
NVLink + NVSwitch within node (fully connected). InfiniBand between nodes. H100 SuperPOD: 32 nodes (256 GPUs) fully connected — every GPU talks to every other GPU at high bandwidth directly.

More expensive to build but much faster for all-to-all communication patterns (all-reduce etc.).

**Why the difference matters:**
- TPU mesh → cheaper, scales to thousands of chips, designed for specific Google communication patterns
- GPU all-to-all → faster, more flexible, supports any communication pattern, dominates general purpose training
- Trade-off: TPU mesh cheaper at scale, GPU all-to-all gets expensive beyond a few hundred GPUs

---

### 5. NVSwitch vs NVLink Switch — Two Levels

**Within a node (8 GPUs):**
```
GPUs ←→ NVSwitch chips (inside node) ←→ GPUs
fully connected at ~900 GB/s
```

**Across 32 nodes (256 GPUs):**
```
Nodes ←→ NVLink Switches (external rack-level switches) ←→ Nodes
fully connected, slightly lower bandwidth
```

- **NVSwitch** = chip inside the node connecting 8 GPUs
- **NVLink Switch** = external switch connecting multiple nodes together
- Both use NVLink as underlying connection technology, just at different scales

**H100 SuperPOD** uses NVLink Switch between nodes → full NVLink fabric across all 256 GPUs.
**A100 SuperPOD** used InfiniBand between nodes → slower inter-node communication.

---

### 6. Collectives Run Behind the Scenes

You never call collectives directly — frameworks handle them automatically:

```python
model = DDP(model)      # DDP handles all-reduce automatically
model = FSDP(model)     # FSDP handles all-gather + reduce-scatter automatically
loss.backward()
optimizer.step()
```

**DDP behind the scenes:**
```
Each GPU: forward pass on own batch
      ↓
Each GPU: backward pass → own gradients
      ↓ (automatic)
all-reduce gradients across all GPUs  ← overlaps with backward for efficiency
      ↓
Every GPU: identical averaged gradients → identical weight update
```

**NCCL** (NVIDIA Collective Communications Library) — implements all collectives optimally for NVLink and InfiniBand. PyTorch calls NCCL, NCCL talks to hardware. Never touched directly.

---

### 7. Three Ways to Parallelize LLMs

**1. Data Parallelism** — each GPU has full model, processes different data:
- **Naïve (DDP)** — all-reduce gradients after backward. Simple but every GPU holds full weights.
- **ZeRO/FSDP** — shard optimizer states/gradients/weights across GPUs for memory savings. Still data parallelism because each GPU runs full model on different data (weights reconstructed temporarily via all-gather).

**2. Model Parallelism** — model itself split across GPUs permanently:
- **Pipeline parallel** — different layers on different GPUs. GPU 0: layers 1-8, GPU 1: layers 9-16 etc.
- **Tensor parallel** — individual weight matrices split across GPUs. Each GPU holds a slice of each layer.

**3. Activation Parallelism:**
- **Sequence parallel** — split sequence length across GPUs. Reduces activation memory for long sequences.

**When to use each:**
```
Full model fits on one GPU      → data parallel only (DDP/FSDP)
One layer fits but not all      → pipeline parallel
Even one layer too big for GPU  → tensor parallel
Massive scale (1000s of GPUs)   → all three combined (3D parallelism)
```

---

### 8. FSDP is Data Parallelism — Why

FSDP shards weights but reconstructs them temporarily before each forward pass:
```
all-gather → GPU 0 has full model → forward on batch A
all-gather → GPU 1 has full model → forward on batch B
```
Every GPU still runs the full model on different data → data parallelism. Sharding is just a memory trick.

Tensor/Pipeline parallel: GPUs permanently hold different parts, must cooperate on every forward pass — neither can run full computation alone.

---

### 9. Pipeline + Tensor Parallelism Together (3D Parallelism)

Two dimensions of splitting simultaneously:

```
Pipeline dimension → which layers does this group handle?
Tensor dimension   → how is each layer split within the group?
```

Example (32 GPUs, 32 layer model):
```
Group A (GPUs 0-3): layers 1-8
  GPU 0: left half of layers 1-8
  GPU 1: right half of layers 1-8

Group B (GPUs 4-7): layers 9-16
  GPU 4: left half of layers 9-16
  GPU 5: right half of layers 9-16
```

Communication:
- Within tensor parallel group → all-reduce (fast NVLink within node)
- Between pipeline stages → send activations point-to-point (across nodes, less communication)

This is why tensor parallel stays within a node and pipeline parallel goes across nodes.

---

### 10. ZeRO Stages 1, 2, 3

In normal DDP each GPU holds all 3 things:
```
Parameters     → full copy on every GPU
Gradients      → full copy on every GPU
Optimizer states → full copy on every GPU  (m and v = 8 bytes/param)
```

**ZeRO Stage 1 — shard optimizer states only:**
```
Parameters     → full copy on every GPU
Gradients      → full copy on every GPU
Optimizer      → each GPU holds only its shard
```
Each GPU still computes full gradients (all-reduce happens). Each GPU updates only its slice of weights using its optimizer shard. Then broadcast updated weights to sync all GPUs.

**ZeRO Stage 2 — shard optimizer + gradients:**
```
Parameters     → full copy on every GPU
Gradients      → each GPU holds only its shard
Optimizer      → each GPU holds only its shard
```
After backward: reduce-scatter instead of all-reduce → each GPU gets only the gradient shard it needs. Less memory, less communication.

**ZeRO Stage 3 — shard everything (= FSDP):**
```
Parameters     → each GPU holds only its shard
Gradients      → each GPU holds only its shard
Optimizer      → each GPU holds only its shard
```
All-gather weights before forward, reduce-scatter gradients after backward. Maximum memory savings — this is what cut per-rank memory 75% (77GB → 15.8GB) in the Llama-3-8B benchmark.

**Summary:**
```
ZeRO 1 → shard optimizer states only    (~4x memory saving)
ZeRO 2 → shard optimizer + gradients   (~8x memory saving)
ZeRO 3 → shard everything (FSDP)       (~64x memory saving)
```

---

### 11. Data Parallelism Scaling Limit

As you add more GPUs, effective batch size grows automatically:
```
1 GPU:    effective batch = 32
8 GPUs:   effective batch = 256
1000 GPUs: effective batch = 32,000
```

**The bottleneck is effective batch size, not per-GPU batch size.** Per-GPU batch stays fixed. Effective batch grows and eventually crosses the noise scale threshold → gradients become redundant → adding more GPUs stops helping.

Two regimes:
- **Perfect scaling** — batch size below noise scale, each batch adds useful gradient info, linear speedup
- **Ineffective scaling** — batch size above noise scale, gradients redundant, diminishing returns

Beyond the wall → switch to model/tensor/pipeline parallelism for real speedups.

---

### 12. Activations — FSDP vs Model Parallelism

**FSDP:** weights/gradients/optimizer sharded, but activations replicated — each GPU runs full forward pass on its own batch, generates its own full activations:
```
FSDP:
  weights      → sharded (1/N per GPU)
  gradients    → sharded (1/N per GPU)
  optimizer    → sharded (1/N per GPU)
  activations  → full copy on every GPU
```

**Tensor parallelism:** each GPU computes only part of each layer → only generates part of activations → real activation memory reduction:
```
Tensor parallel:
  weights      → split across GPUs
  activations  → split across GPUs
```

**Sequence parallelism:** splits sequence dimension across GPUs → each GPU holds activations for part of the sequence only. Specifically targets activation memory for long sequences.

Summary:
- FSDP → solves weight/gradient/optimizer memory, activations still full
- Tensor parallel → solves activation memory too
- Sequence parallel → specifically targets activation memory for long sequences

---

### 13. Pipeline Parallelism — Forward/Backward and Microbatching

**Forward pass:** activations flow GPU to GPU:
```
GPU 0 (layers 1-8) → send activations → GPU 1 (layers 9-16) → ... → GPU 3
```

**Backward pass:** gradients flow in reverse:
```
GPU 3 → send dL/dx → GPU 2 → send dL/dx → GPU 1 → send dL/dx → GPU 0
```

**Pipeline bubble problem (naive):**
```
GPU 0: [batch1 fwd][  idle  ][  idle  ][  idle  ][batch1 bwd]
GPU 1:             [batch1 fwd][  idle  ][  idle  ][batch1 bwd]
GPU 2:                        [batch1 fwd][  idle  ][batch1 bwd]
GPU 3:                                   [batch1 fwd][batch1 bwd]
```
GPU 0 sits idle for 3 steps — wasted GPU time.

**Microbatching fix:**
Split batch into micro-batches (m1, m2, m3, m4) — while GPU 1 processes m1, GPU 0 already processes m2:
```
GPU 0: [m1 fwd][m2 fwd][m3 fwd][m4 fwd][m4 bwd][m3 bwd][m2 bwd][m1 bwd]
GPU 1:         [m1 fwd][m2 fwd][m3 fwd][m4 fwd][m4 bwd][m3 bwd][m2 bwd][m1 bwd]
GPU 2:                 [m1 fwd][m2 fwd][m3 fwd][m4 fwd]...
GPU 3:                         [m1 fwd][m2 fwd]...
```

Pipeline fills up to steady state where all GPUs work simultaneously. Bubble only exists at start (filling) and end (draining).

More micro-batches = smaller bubble as fraction of total time:
```
4 stages, 4 micro-batches:   bubble = 3/7   (large)
4 stages, 16 micro-batches:  bubble = 3/19  (small)
4 stages, 100 micro-batches: bubble = 3/103 (tiny)
```

---

### 14. Micro-batch Size Tradeoff

Total batch size is fixed. Micro-batch size controls how many micro-batches you get:
```
smaller micro-batch → more micro-batches → smaller bubble ✓
                    → less compute per step → lower GPU utilization ✗

larger micro-batch → fewer micro-batches → bigger bubble ✗
                   → more compute per step → better GPU utilization ✓
```

Tune micro-batch size to balance pipeline bubble vs per-step GPU compute utilization. Neither extreme is good.

---

### 15. Pipeline Performance and Batch Size

Larger batch size = more micro-batches = smaller bubble = better performance.

Chart proof:
```
batch 128 → 128/8 = 16 micro-batches → tiny bubble → performance stays ~160 TFLOPs
batch 8   → 8/8 = 1 micro-batch      → huge bubble → performance drops to ~90 TFLOPs
```

Micro-batch = dividing one full batch into smaller pieces fed through pipeline one by one. Gradients accumulated across all micro-batches → one optimizer step at end. Model sees same effective batch size.

---

### 16. 3D Parallelism — All Three Together

```
Pipeline parallel  → split layers across GPU groups (vertical split)
Tensor parallel    → split weight matrices within each GPU group (horizontal split)
Data parallel      → each GPU group processes different data
```

Example (4 GPUs, 32 layers):
```
Group A (GPUs 0-1): layers 1-16
  GPU 0: left half of weights
  GPU 1: right half of weights

Group B (GPUs 2-3): layers 17-32
  GPU 2: left half of weights
  GPU 3: right half of weights
```

Used for 70B+ models trained from scratch. For fine-tuning, FSDP alone is enough.

---

### 17. Sequence Parallelism — Making Memory Truly Linear

**Problem:** tensor parallelism splits weights but LayerNorm/Dropout are pointwise ops — can't be tensor parallelized. Their activations still fully replicated on every GPU.

**Solution:** alternate between sequence parallel and tensor parallel:
```
LayerNorm/Dropout → Sequence Parallel (split across sequence dimension)
Attention/FFN     → Tensor Parallel   (split across model dimension)
```

Transition between modes ('g' in diagram):
- Forward pass: g = all-gather (reconstruct full sequence before tensor parallel ops)
- Backward pass: g̃ = reduce-scatter (scatter back to sequence parallel)

LayerNorm/Dropout are pointwise — each token processed independently → split sequence across GPUs, each handles its own tokens, no communication needed.

**Result:** every part of transformer now split across GPUs — either by sequence or model dimension. No fully replicated activations. Memory scales linearly with number of GPUs.

---

### 18. Activation Memory Scaling — Full Picture

Variables: s=sequence length, b=batch size, h=hidden dim, t=tensor parallel size, a=attention heads

| Configuration | Activation Memory Per Layer |
|---|---|
| No parallelism | sbh(34 + 5as/h) |
| Tensor parallel | sbh(10 + 24/t + 5as/ht) |
| Tensor + Sequence parallel | sbh(34/t + 5as/ht) |
| Tensor + Activation recomputation | sbh(10 + 24/t) |
| Tensor + Sequence + Activation recomputation | sbh(34/t) |

**Best combination — all three:**
```
Tensor parallel       → splits weight-related activations by t
Sequence parallel     → splits LayerNorm/Dropout activations by t
Activation recomputation → throws away expensive activations, recomputes during backward
```

Result: `sbh(34/t)` — everything divided by t. Add more GPUs → memory drops proportionally. True linear scaling.

Cost: activation recomputation adds ~33% extra compute. Worthwhile when memory is the bottleneck.

---

### 19. Other Parallelism Strategies

**Context Parallel / Ring Attention:**
For very long sequences (100k+ tokens) — even sequence parallelism isn't enough. KV cache and attention too large for one GPU.

Each GPU holds a chunk of the sequence. KV blocks rotate around the ring — GPU 0 sends KV to GPU 1, GPU 1 sends to GPU 2 etc. Each GPU accumulates partial attention results as it receives each KV block:
```
GPU 0: tokens 0-1000     → computes attention against all KV blocks as they rotate
GPU 1: tokens 1000-2000  → same
GPU 2: tokens 2000-3000  → same
```
Used by: Claude (200k), Gemini (1M tokens) for long context.

**Expert Parallel:**
MoE experts stored on different GPUs — each token routed to its assigned expert GPU:
```
Expert 1 → GPU 0
Expert 2 → GPU 1
Expert 3 → GPU 2
```
Same as DeepSeek MoE expert parallelism from Lecture 4 — this is the distributed training version.

**At maximum scale — 5D parallelism:**
Data parallel + Tensor parallel + Pipeline parallel + Sequence/Context parallel + Expert parallel — all combined simultaneously.

---

### 20. FSDP + Model Parallelism Together

FSDP reconstructs weights ONE LAYER AT A TIME, not the entire model at once:
```
Layer 1: all-gather weights → compute → discard
Layer 2: all-gather weights → compute → discard
```
Peak memory = one layer's weights, not all layers simultaneously.

FSDP fails when even ONE layer is too big for one GPU after all-gather → add tensor parallelism to split that layer's weight matrix across GPUs.

---

### 21. 3D Parallelism — Practical Decision Framework

**Rule 1 — Get model to fit in memory first:**
```
Step 1: Tensor parallel within a node
        (frequent all-reduce → needs fast NVLink)
Step 2: Pipeline parallel across nodes
        (only sends activations → slower InfiniBand is fine)
        OR ZeRO-3/FSDP depending on bandwidth
```

**Rule 2 — Once model fits, scale with data parallel:**
Use all remaining GPUs for data parallelism — process more data in parallel for throughput.

**Small batch tip:** use gradient accumulation — run multiple micro-batches, accumulate gradients without syncing, sync once. Effectively increases batch size without increasing memory → better communication efficiency.

**Layout:** 2 data parallel ranks × 3 pipeline stages × tensor parallel within each stage = classic 3D parallelism used by Megatron-LM for GPT-3 scale models.

---

### 22. 3D Parallelism — Concrete Example

32 GPUs across 4 nodes (8 GPUs per node), training a 70B model:

**Tensor parallel within each node (8 GPUs):**
Each node's 8 GPUs split each layer's weight matrix:
```
Node 0: GPU 0-7 each hold 1/8 of every weight matrix
Node 1: GPU 8-15 each hold 1/8 of every weight matrix
Node 2: GPU 16-23 each hold 1/8 of every weight matrix
Node 3: GPU 24-31 each hold 1/8 of every weight matrix
```

**Pipeline parallel across nodes (4 nodes):**
Each node handles different layers:
```
Node 0: layers 1-20
Node 1: layers 21-40
Node 2: layers 41-60
Node 3: layers 61-80
```

**Data parallel on top:**
A second set of 32 GPUs is an identical copy processing different data simultaneously → all-reduce gradients between the two sets.

**What's happening at any moment:**
- Within Node 0: 8 GPUs collaborating via NVLink on same layer (tensor parallel)
- Across nodes: activations flowing Node 0 → Node 1 → Node 2 → Node 3 (pipeline parallel)
- Across two sets of 32 GPUs: all-reduce gradients (data parallel)

---

### 23. Gradient Accumulation

Different from activation recomputation — solves a different problem.

**What it is:** run multiple forward+backward passes, accumulate gradients before optimizer step:
```
micro-batch 1 → forward → backward → accumulate gradients (no update)
micro-batch 2 → forward → backward → accumulate gradients (no update)
micro-batch 3 → forward → backward → accumulate gradients (no update)
micro-batch 4 → forward → backward → accumulate gradients → optimizer step
```

**Why:** simulates larger batch size without needing more memory:
```
GPU fits batch 32, want effective batch 128
→ accumulation steps = 4
→ run 4 × batch 32 = effective batch 128
→ uses memory of batch 32
```

**In pipeline parallelism:** accumulate gradients across all micro-batches, sync once at the end instead of after every micro-batch:
```
micro-batch 1,2,3,4 → accumulate gradients → one all-reduce → one optimizer step
```
Pay communication cost once instead of 4× — 4× less communication overhead.

**PyTorch:** `.backward()` automatically adds to `.grad` each call. `zero_grad()` clears it when ready to start fresh.

**Key difference:**
```
Activation recomputation → saves activation memory during forward pass
Gradient accumulation    → simulates larger batch size using less memory, reduces communication
```

---

## Questions / Follow-up
