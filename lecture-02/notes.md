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

## Questions / Follow-up
