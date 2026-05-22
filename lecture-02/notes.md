# Lecture 2 — CS336

## Summary
Tokenization strategies, embeddings, and training compute estimation (napkin math).

---

## Key Concepts

### 1. Character vs Byte Tokenizer

**Character tokenizer:**
- Splits text into individual characters
- `"hello"` → `["h", "e", "l", "l", "o"]` — 5 tokens
- Problem: vocab size depends on languages seen during training; fails on unseen characters

**Byte tokenizer:**
- Splits text into UTF-8 bytes (values 0–255)
- `"hello"` → `[104, 101, 108, 108, 111]` — 5 tokens
- `"café"` → `[99, 97, 102, 195, 169]` — 5 tokens (é = 2 bytes)

Why bytes beat characters:
1. Universal — always exactly 256 possible tokens, works for every language
2. No unknown tokens — everything is representable in bytes
3. Fixed vocab size — always 256, regardless of training data

---

### 2. BPE (Byte Pair Encoding)
- Starts from bytes, then iteratively merges frequent byte pairs into single tokens
- So `"hello"` becomes one token instead of five — much more sequence-efficient
- Used by GPT-2/3/4 → called "byte-level BPE"
- Byte tokenization is the foundation BPE builds on top of

---

### 3. Embedders
- Deep learning models (transformer-based) that convert text → dense vectors (e.g. 768 or 1536 dimensions)
- Key property: semantically similar text lands near each other in vector space
- Trained with contrastive learning (push similar meanings together, dissimilar apart)
- In RAG: embed knowledge chunks → nearest-neighbor search at query time

**How to choose an embedder:**
- Does the modality match? (text, code, multimodal)
- Does the language match? (multilingual-e5 for non-English)
- Domain: specialized domains benefit from domain-specific models (CodeBERT, etc.)
- Check the MTEB leaderboard for retrieval benchmarks
- Practical: latency, cost, vector size, context window (some cap at 512 tokens)

Rule of thumb:
- Prototyping → `text-embedding-3-small` or `nomic-embed-text`
- Production → benchmark top MTEB retrieval models on your actual data

---

### 4. Training Compute — The 6N Rule

**FLOPs per parameter per token:**
- Forward pass = **2 FLOPs/param** (one multiply + one add per weight)
- Backward pass = **4 FLOPs/param** (two gradients, each ~2 FLOPs)
  - `dL/dW` — gradient w.r.t. weights → used by optimizer to update weights
  - `dL/dx` — gradient w.r.t. inputs → passed to previous layer so backprop can continue
- **Total = 6 FLOPs/param/token**

Why dL/dx is needed: each layer only receives gradients from the layer ahead. Without dL/dx, the chain rule stops and earlier layers never get updated.

---

### 5. Napkin Math — Training Llama 2 70B

**Given:** 70B params, 15T tokens, 1024 H100 GPUs

**Step 1 — Total FLOPs:**
```
total_flops = 6 × 70e9 × 15e12 = 6.3 × 10²⁴ FLOPs
```

**Step 2 — FLOPs available per day:**
```
H100 peak = 1979 TFLOPS (BF16)
MFU = 0.5  (real efficiency ~50% due to communication overhead, memory bandwidth)
effective = ~989 TFLOPS/GPU

flops_per_day = 989e12 × 1024 × 86400 ≈ 8.75 × 10¹⁹ FLOPs/day
```

**Step 3 — Days:**
```
days = 6.3e24 / 8.75e19 ≈ 72 days
```

Real-world: Llama 2 70B took ~1.7M GPU-hours → at 1024 GPUs = ~69 days ✓

**MFU (Model FLOP Utilization)** is ~0.5 in practice because:
- GPUs wait on each other (distributed training communication)
- Memory bandwidth bottlenecks
- Load imbalance

---

### 6. Memory Budgeting — Largest Model on 8 H100s (AdamW)

AdamW needs **16 bytes per parameter**:

| What | Bytes | Why |
|------|-------|-----|
| Parameters | 4 | fp32 master copy |
| Gradients | 4 | fp32 |
| Adam 1st moment (m) | 4 | running mean of gradients |
| Adam 2nd moment (v) | 4 | running variance of gradients |

Calculation for 8 × 80GB H100s:
```
total_memory     = 80e9 × 8 = 640e9 bytes
num_parameters   = 640e9 / 16 = 40B parameters
```

This is **across all 8 GPUs combined** (model parallelism — weights are sharded, not replicated).

Caveat: activations during the forward pass add significant memory on top of this, so you'd fit fewer parameters in practice.

---

### 7. Tensors

A tensor is an n-dimensional array of numbers. A vector is just a special case (1D tensor).

```
0D (scalar):   42.0                        shape: ()
1D (vector):   [1, 2, 3]                   shape: (3,)
2D (matrix):   [[1, 2],                    shape: (3, 2)   ← rows, cols
                [3, 4],
                [5, 6]]
3D:            cube-like                   shape: (2, 3, 4)
4D:            batch of images             shape: (32, 3, 224, 224)
                                                    ↑   ↑   ↑    ↑
                                                 batch RGB  H    W
```

Shape convention: always `(rows, columns)` for 2D, outermost dimension first.

The `(3,)` trailing comma is Python notation — distinguishes a 1-element tuple from the integer `3`.

In deep learning everything is a tensor:
- Model weights → 2D (matrix)
- Batch of tokens → 2D (batch_size × sequence_length)
- Attention scores → 4D (batch × heads × seq × seq)

---

### 8. Sequence Length

How many tokens are in an input:
```
"The cat sat on the mat"
→ ["The", "cat", "sat", "on", "the", "mat"]  → sequence length = 6
```

Batch of text tensor shape:
```
(batch_size, sequence_length)  e.g. (32, 512)
```

Why it matters:
- Models have a max sequence length = **context window** (GPT-4 = 128k tokens)
- Longer sequences = more memory + more compute
- Attention cost scales as **sequence_length²** — doubling sequence length quadruples attention compute

---

### 9. Memory Calculation by Scenario

**bytes_per_parameter = parameters + gradients + optimizer state**

| Scenario | Bytes/param | Calculation (7B model) | Memory |
|----------|-------------|------------------------|--------|
| Inference fp16 | 2 | 7e9 × 2 | 14 GB |
| Inference fp32 | 4 | 7e9 × 4 | 28 GB |
| Training AdamW | 16 | 7e9 × 16 | 112 GB |

Inference is cheap — just weights, no gradients or optimizer state.
Training needs all three simultaneously → 8× more memory than fp16 inference.

---

### 10. Activations — Inference vs Training

**Inference:** activations computed and discarded layer by layer — only current layer in memory at any time.

**Training:** all activations must be kept in memory during the forward pass because backprop needs them to compute gradients (chain rule). All layers simultaneously → expensive.

**Gradient checkpointing:** discard activations during forward pass, recompute during backward pass.
```
normal training:      save all activations  → less compute, more memory
grad checkpointing:   recompute on the fly  → more compute, less memory
```

---

### 11. bf16 vs fp16 — Underflow

Floating point = `mantissa × 2^exponent`
- **mantissa** → precision
- **exponent** → range (how large/small numbers can get)

```
           sign  exponent  mantissa
fp16:        1      5        10
bf16:        1      8         7
fp32:        1      8        23
```

**Underflow** = a number so small it rounds to zero because the format can't represent it.

fp16 has only 5 exponent bits → small range → gradients deep in the network can underflow to zero.

bf16 has 8 exponent bits (same as fp32) → much larger range → tiny gradients survive.

For training, **range matters more than precision** → bf16 preferred over fp16.

---

### 12. Mixed Precision Training — What Runs in bf16 vs fp32

**bf16:**
- Forward pass
- Backward pass
- Gradients (initially)

**fp32:**
- Master copy of weights
- Optimizer states (Adam m and v)
- Weight update step

Why fp32 for the update: `w = w - lr × gradient` produces extremely tiny changes. bf16 lacks the precision to accumulate them — they round to zero and weights never update. fp32 has enough precision.

**The full flow:**
```
fp32 master weights
      ↓ cast to bf16
forward pass (bf16)
      ↓
backward pass (bf16) → gradients in bf16
      ↓ cast to fp32
optimizer states (m, v) computed in fp32
weight update in fp32
      ↓ cast to bf16
next forward pass
```

This is why training still costs 16 bytes/param even with mixed precision — fp32 master weights always exist alongside bf16 working copy.

---

### 13. Mixed Precision — Exact Memory Breakdown

At peak memory during training, both bf16 and fp32 copies of weights exist simultaneously:

```
bf16 weights         2 bytes/param   (forward/backward pass)
fp32 master weights  4 bytes/param   (optimizer step)
fp32 gradients       4 bytes/param
fp32 m               4 bytes/param
fp32 v               4 bytes/param
──────────────────────────────────
total               18 bytes/param
```

The rough estimate of 16 bytes/param ignores the bf16 working copy. Strictly it's 18.

The tradeoff is still worth it because:
- Forward/backward is much faster in bf16
- Activations are stored in bf16 → big memory saving since activations can be huge
- Extra 2 bytes for the bf16 copy is a small price for the speed gain

**Exact AdamW optimizer step:**
```
1. forward pass             (bf16)
2. backward pass            (bf16)
3. gradients computed       (bf16)
         ↓ cast to fp32
4. m = β₁×m + (1-β₁)×grad  (fp32)  ← optimizer state update
   v = β₂×v + (1-β₂)×grad² (fp32)  ← optimizer state update
         ↓
5. w = w - lr × m/√v        (fp32)  ← weight update
         ↓ cast to bf16
6. next forward pass        (bf16)
```

---

### 14. Tensors — Under the Hood

A tensor is just a **pointer to a flat contiguous block of GPU memory** + metadata:

```
pointer   → where in GPU memory the data starts
shape     → (3, 224, 224)
dtype     → bf16, fp32 etc
stride    → how to step through memory for each dimension
```

**Strides example:**

Flat memory: `[1, 2, 3, 4, 5, 6]`

Shape `(2, 3)` with strides `(3, 1)` means:
- Next row → jump 3 elements
- Next column → jump 1 element

So the 2D matrix is just a view into flat memory — no data is duplicated.

**Key insight:** reshaping a tensor doesn't move any data — it just changes shape and stride metadata. Reshape is basically free.

This is why GPU memory management matters — contiguous memory, in-place operations, and memory fragmentation all affect performance.

---

### 15. Tensor Views — Shared Memory

Two tensors can point to the same GPU memory. Changing one changes the other:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = x.view(3)       # same memory as x
x[0] = 99.0
print(y)            # tensor([99., 2., 3.])
```

Operations that return **views** (same memory):
```python
x.view(...)
x.reshape(...)      # usually a view
x.transpose(...)
x[:]                # slicing
```

Operations that return **copies** (new memory):
```python
x.clone()
x.contiguous()
```

Common bug: modifying a slice thinking it's independent but it's a view — silently mutates original tensor. Use `.clone()` when in doubt.

---

### 16. Batch Size vs Sequence Length

- **Sequence length** — how many tokens one sentence has
- **Batch size** — how many sentences in one batch

```
(batch_size, sequence_length)
     ↑               ↑
how many          how many tokens
sentences         per sentence
```

All sequences in a batch must be the same length → shorter ones get padded with `[PAD]` token:
```
"The cat sat"  → [The, cat, sat]
"Hello world"  → [Hello, world, PAD]

tensor shape: (2, 3)
```

---

### 17. Einops — Motivation and rearrange

**Problem:** `permute(0, 2, 1, 3).reshape(...)` is unreadable and error-prone. Hard to track what each dimension index means.

**Einops solution:** name your dimensions explicitly.

```python
# old way
x.permute(0, 2, 1, 3).reshape(batch, seq, heads*d_head)

# einops
rearrange(x, 'batch heads seq d_head -> batch seq (heads d_head)')
```

**rearrange examples:**
```python
# transpose
rearrange(x, 'batch seq -> seq batch')

# merge dimensions
rearrange(x, 'batch seq features -> batch (seq features)')

# split a dimension
rearrange(x, 'batch (seq features) -> batch seq features', seq=3, features=4)
```

Pattern: `'input_shape -> output_shape'` with named dimensions. Also validates shapes at runtime — throws error immediately if tensor doesn't match.

---

### 18. MFU and Multi-GPU Communication

**MFU = actual FLOPs/sec ÷ theoretical peak FLOPs/sec**

More GPUs → more gradient synchronization + communication overhead → GPUs sit idle waiting → actual FLOPs/sec drops → MFU goes down.

```
1 GPU:     MFU ~0.5
8 GPUs:    MFU ~0.45
64 GPUs:   MFU ~0.4
1024 GPUs: MFU ~0.3
```

Scaling is not linear — communication cost grows with GPU count.

Why interconnect speed matters:
- **NVLink** (GPU→GPU inside a node) — fast
- **InfiniBand** (GPU→GPU across nodes) — slower → MFU drops when going multi-node

Keeping MFU high at thousands of GPUs is one of the core engineering challenges in large scale training.

---

### 19. Backward Pass — Gradient Flow Between Layers

For every layer, two gradients are computed:
- **dL/dW** — gradient w.r.t. weights → used to update that layer's weights
- **dL/dx** — gradient w.r.t. inputs → passed to the previous layer

The `dL/dx` received from the layer ahead is used for **both**:

```
received from Layer 3:   dL/dx  (how loss changes w.r.t. Layer 2's output)
Layer 2 knows:           its own input (saved activation from forward pass)

computes:
  dL/dW₂ = dL/dx × input₂ᵀ    ← update W₂
  dL/dx₂ = W₂ᵀ × dL/dx        ← pass back to Layer 1
```

**Important:** `W₂ᵀ` used here is the **original W₂** from the forward pass, not the updated one. Weights are only updated after the entire backward pass is complete:

```
1. forward pass   → use W₂ (save activations)
2. backward pass  → use same W₂ to compute gradients
3. optimizer step → NOW update W₂ using dL/dW₂
```

dL/dx is the messenger — each layer consumes the gradient from the layer ahead and produces a new one for the layer behind.

---

### 20. AdamW Optimizer Step (corrected)

`dL/dW` is the input to the optimizer — m and v are computed using it each step:

```
backward pass → collect dL/dW for every layer
                      ↓
optimizer step → for each layer:
                   1. m = β₁×m + (1-β₁)×dL/dW    ← update 1st moment
                   2. v = β₂×v + (1-β₂)×(dL/dW)² ← update 2nd moment
                   3. W = W - lr × m/√v            ← update weights
```

dL/dx — only exists to carry signal backwards, never used to update anything.
dL/dW — the only thing that matters, one per layer, fed into the optimizer.

---

### 21. Parameter Initialization — Xavier

**The problem:**

Each output element is a sum of `input_dim` terms:
```
output_j = x₁w₁ + x₂w₂ + ... + xₙwₙ
```

When you sum n random numbers, the result grows as √n. So with `input_dim = 2873`, outputs scale to `√2873 ≈ 53`. Large outputs → large gradients → unstable training.

**The fix — Xavier initialization:**

Divide weights by √(input_dim) to cancel the growth:
```python
w = torch.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
```

```
output grows by √n
weights shrunk by √n
────────────────────
net effect = 1   ✓
```

Output stays ~constant regardless of layer width.

**Truncate to [-3, 3]:** `torch.randn` can produce extreme outliers (e.g. 5, -6) which cause output spikes even after scaling. Truncating clips them.

---

## Questions / Follow-up
