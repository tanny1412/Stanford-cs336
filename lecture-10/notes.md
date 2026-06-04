# Lecture 10 — CS336

## Summary
Inference efficiency — how to serve LLMs fast and cheap. Core bottleneck is memory bandwidth during decode. Everything (KV cache, GQA, MLA, quantization, speculative decoding) is about reducing memory reads or hiding their cost.


## Key Concepts

### 1. Compute-Limited vs Memory-Limited

**H100 accelerator intensity:**
```
FLOPs per second   = 989 × 10¹²
Memory bandwidth   = 3.35 × 10¹²
Accelerator intensity = FLOPs/s ÷ memory bandwidth = 295
```

295 = for every byte moved from memory, the H100 can do 295 FLOPs.

**Two regimes:**
- Computation intensity > 295 → **compute-limited** (good) — GPU is doing real work, stays busy
- Computation intensity < 295 → **memory-limited** (bad) — GPU is idle, just waiting for data to arrive from memory

**Conclusion: compute-limited if and only if batch size B > 295**

Small batch → load weights, do tiny computation → memory limited → GPU mostly idle, wasted.
Large batch → load weights once, run many inputs through → lots of computation per load → compute limited → GPU fully busy.

Batch size is the knob. This is why inference servers like vLLM work hard to batch requests together — a single request is almost always memory-limited and wastes the GPU.

---

### 2. Naive Inference — Why It's Slow

Every time you generate a new token, you feed the entire history into the transformer from scratch:

```
Step 1: "never gonna give you"          → transformer → sample "up"
Step 2: "never gonna give you up"       → transformer → sample "never"
Step 3: "never gonna give you up never" → transformer → sample "gonna"
```

Each step re-computes attention over the full sequence even though most of it hasn't changed.

**Complexity:**
```
One forward pass        = O(T²) FLOPs  (attention is quadratic in sequence length)
Generating T tokens     = T forward passes
Total                   = O(T³) FLOPs
```

1000 tokens naively = 1000³ operations. Grows cubically — gets extremely slow for long sequences.

This is the problem KV cache solves — cache the key and value vectors from previous tokens and reuse them instead of recomputing from scratch every step.

---

### 3. KV Cache, Prefill and Decode

**KV Cache — the core idea:**
During attention, every token has Q, K, V vectors. When generating token N you need K and V from all previous tokens. Without cache you recompute them every step. KV cache saves them so you never recompute.

**One KV cache per attention layer:**
```
Llama 3 8B has 32 layers
→ 32 separate KV caches
→ each stores K and V for every token seen so far
```

For a sequence of 1000 tokens:
```
Layer 1  → K: [1000 vectors], V: [1000 vectors]
Layer 2  → K: [1000 vectors], V: [1000 vectors]
...
Layer 32 → K: [1000 vectors], V: [1000 vectors]
```

**Prefill phase:**
```
Run ONE forward pass over all prompt tokens simultaneously
→ compute and cache K, V for every prompt token at once
→ get logits for last token → sample first output token
```
Fast — processing many tokens in parallel. Compute-limited.

**Decode phase:**
```
Run forward pass for just 1 new token
→ look up cached K,V from all previous tokens
→ add new token's K,V to cache
→ sample next token → repeat
```
Slow — one token at a time, loading entire KV cache from memory every step. Memory-limited.

**Together:**
```
Prefill  → process prompt in parallel → cache K,V → first token
Decode   → one token at a time → reuse cache → each new token
```

**Why KV cache is memory hungry:**
```
memory = num_layers × seq_len × 2 (K and V) × d_model × bytes_per_param
```
For a 70B model with 128k context window = tens of GBs, sometimes more than the model weights themselves. That's the real bottleneck at inference. This is exactly the problem vLLM's PagedAttention solves.

---

### 4. Batching, MLP vs Attention Intensity

**MLP layers during decode:**
- All sequences in the batch share the same weights (Wup, Wgate, Wdown)
- Bigger batch → more sequences using same weights → more FLOPs per weight load
- Arithmetic intensity = B → scales with batch size → can become compute-limited

**Attention layers during decode:**
- Each sequence has its own KV cache — Q, K, V depend on the specific sequence
- Can't share work across sequences, each needs its own K,V loaded from memory
- Arithmetic intensity = 1 regardless of batch size → always memory-limited, batching doesn't help

```
Prefill   → compute-limited  (processing many tokens in parallel)
Decode    → memory-limited   (one token at a time)

MLP       → intensity = B    (gets better with more concurrent users)
Attention → intensity = 1    (stuck, batching doesn't fix it)
```

---

### 5. GQA and MQA — Reducing KV Cache Reads

Batching doesn't help attention reads → reduce KV size itself by sharing K,V across heads.

**Standard Multi-Head Attention (MHA):**
```
8 heads → 8 separate K caches, 8 separate V caches → load all 8 every decode step
```

**Multi-Query Attention (MQA):**
```
8 query heads, only 1 K cache, 1 V cache shared across all heads → 8x less KV reads
```

**Grouped Query Attention (GQA) — middle ground:**
```
8 query heads, 2 K caches, 2 V caches (groups of 4 heads share one K,V) → 4x less KV reads
```

GQA is now the standard — MQA too aggressive (hurts quality), MHA too slow at inference. Llama 3 uses GQA.

---

### 6. KV Cache Dimensions

KV cache has three dimensions all at once:
```
KV cache = num_layers × num_heads × num_tokens × d_head × 2 (K and V)
```

Concretely for Llama 3 8B with 1000 tokens:
```
32 layers × 32 heads × 1000 tokens × 128 d_head × 2 = ~262M numbers
```

**What varies across each dimension:**
- Layers → each layer has its own independent K, V (different weights)
- Heads → each head attends to different things (all separate in MHA)
- Tokens → each token has its own K, V vector stored

**What GQA reduces:** only the heads dimension — instead of 32 separate K,V per layer, maybe 8 groups. Layers and tokens stay the same.

Long sequences on big models eat enormous memory because all three dimensions are large simultaneously.

---

### 7. Why You Can't Scale Layers or Responses Infinitely

Two things grow as you scale and both hit the same wall — memory bandwidth:

**More layers → more KV caches to load every decode step:**
```
32 layers  → 32 KV caches loaded every step
128 layers → 128 KV caches loaded every step
```

**Longer responses → more tokens in KV cache:**
```
Every new token generated adds one entry to every layer's KV cache
1000 token response → 1000 entries per layer per head in memory
```

Longer responses get progressively slower — each new token takes longer than the previous one because the KV cache keeps growing and there's more to load from memory each step.

Bigger models (more layers, more heads) → more KV cache from the start.
Longer responses → KV cache keeps growing throughout generation.

Both hit memory bandwidth — you're moving more and more data from memory to GPU on every single decode step.

---

### 8. Memory, Latency and Throughput Formulas

**Total memory:**
```
memory = B × kv_cache_size + parameter_size
```
Model weights are fixed regardless of batch size. Only KV cache scales with B — each new sequence adds its own KV cache.

**Latency — time to generate one token:**
```
latency = memory / memory_bandwidth
```
Every decode step reads all parameters + all KV caches from memory. Latency is entirely determined by how fast you can move data, not compute speed.

**Throughput — tokens generated per second:**
```
throughput = B / latency
```
B tokens generated in parallel in one decode step. Throughput scales with batch size — more users batched = more tokens per second from the same hardware.

**The tradeoff:**
- Bigger B → better throughput (more tokens/sec)
- Bigger B → more KV cache → higher latency per step
- Parameters don't scale with B — same weights loaded regardless

```
B = 1:   load parameters + 1 KV cache  → low latency, low throughput
B = 100: load parameters + 100 KV caches → higher latency, much higher throughput
```
Throughput wins because you generate 100 tokens in the same step that B=1 generates 1 — the gain outweighs the latency increase.

---

### 9. Why Bigger GPUs Help Both Latency and Throughput

**More memory → fit more sequences in a batch:**
```
Small GPU: B = 10  → low throughput
Big GPU:   B = 100 → 10x more throughput
```

**More memory bandwidth → lower latency:**
```
latency = memory / memory_bandwidth
H100: 80GB, 3.35TB/s bandwidth
A100: 80GB, 2TB/s bandwidth
→ H100 moves data 1.6x faster → 1.6x lower latency on decode
```

Multi-GPU inference pools memory and bandwidth across cards — fit larger batches, lower latency simultaneously.

---

### 10. Linear Attention

**Standard attention:** every token attends to every other token — full attention matrix. O(T²) memory, KV cache grows with sequence length.

**Linear attention:** mathematical approximation (Taylor approximation) that rewrites attention as a fixed-size hidden state — like SSMs:
```
Standard: softmax(QKᵀ)V  → full matrix, O(T²)
Linear:   φ(Q)(φ(K)ᵀV)  → fixed state, O(T)
```
Never builds the full attention matrix — just maintains a running compressed state. Constant memory regardless of sequence length.

Tradeoff: approximate recall — can't perfectly remember exact past tokens.

**Three approaches:**
- Linear attention → great long range memory, approximate recall
- Sliding window → precise recall but only recent tokens
- BASED (Arora+ 2024) → linear attention for long range + local window for precise recent recall

MiniMax-01 uses linear attention + full attention in a 456B MoE model.

---

### 11. Quantization — PTQ vs QAT vs Mixed Precision

**Mixed precision training (NOT quantization):**
```
Store weights in fp32   (master copy)
Cast to fp16/bf16 for forward + backward pass (faster compute)
Update gradients in fp32 (stable)
```
Standard in all modern training — purely for speed and memory, not about inference compression.

**PTQ (Post Training Quantization) — the industry standard:**
- Model fully trained → compress weights after the fact
- fp16 → int8 → int4, no retraining needed
- For inference: less memory, faster memory reads during decode
- Examples: AWQ, GPTQ, bitsandbytes
- Downside: some quality loss, but acceptable especially at 4-bit with good calibration

**QAT (Quantization Aware Training):**
- During training, simulate quantization in forward pass
- Model learns to compensate for precision loss
- Better quality than PTQ at same bit width
- Expensive — requires retraining. Rarely used in practice.

**QLoRA:**
```
Base model → quantized to 4-bit (PTQ)
           → train LoRA adapters on top in fp16
```
Combines PTQ + LoRA — quantization for memory savings, LoRA for efficient fine-tuning.

**Practical reality:**
```
Production inference     → PTQ (AWQ, GPTQ)
Everyday fine-tuning     → QLoRA (PTQ + LoRA)
Extreme compression      → maybe QAT
```

---

### 12. Pruning

**Quantization vs Pruning:**
```
Quantization: keep all weights, reduce precision → fp16 to int4
Pruning:      remove weights entirely → 175B → 87.5B weights
```

Some weights contribute almost nothing (near zero) — safe to remove. Fewer weights = less memory to load, less compute.

**Two types:**
- Unstructured pruning — remove individual weights randomly. Hard to speed up on GPU (scattered pattern)
- Structured pruning — remove entire heads, layers, neurons. GPU-friendly, clean smaller matrices → less KV cache → faster decode

**Pruning alone hurts quality** — need distillation to recover it.

---

### 13. Distillation After Pruning

**The pipeline:**
```
Train big model
→ prune (remove heads/layers/weights)
→ distill from original into pruned model
→ quantize
→ deploy
```

**Distillation recipe (3 steps):**
1. Define faster model architecture (fewer layers, heads, smaller d_model)
2. Initialize weights from original model where possible — don't start from random, steal weights from teacher → already-good starting point
3. Repair via distillation — fine-tune student using teacher's output distributions as targets

**Why teacher's soft distributions beat hard labels:**
```
Hard label:    token X is correct (1 bit of info)
Soft distrib:  token X 60%, token Y 30%, token Z 10% (rich signal)
```
Student absorbs knowledge about relationships between tokens, not just correct answers.

**Why initialize from teacher weights:**
```
Random init  → student knows nothing → distillation teaches everything → slow
Teacher init → student already knows most things → distillation fixes gaps → fast
```

---

### 14. Speculative Decoding

Big model generates one token at a time — slow. Speculative decoding uses a cheap draft model to speed it up with no quality loss.

**The trick:**
```
Step 1: small draft model guesses next 4 tokens quickly
Step 2: big target model verifies all 4 in ONE parallel forward pass (like prefill)
Step 3: accept tokens big model agrees with, reject at first disagreement
        → all 4 accepted: got 4 tokens for cost of ~1 big model step
        → rejected at token 3: keep 1,2 use big model's token 3, restart
```

Output quality is identical to using just the big model — only accepting tokens the big model agrees with.

```
Without: 4 tokens = 4 slow big model steps
With:    4 tokens = 4 fast draft steps + 1 parallel verify step
```

2-3x speedup in practice with zero quality loss.

---

### 15. State Space Models and Diffusion Models (Glossary)

**State Space Models (SSMs) — e.g. Mamba:**
Alternative to attention that replaces growing KV cache with a fixed-size hidden state. No matter how long the sequence, state size stays constant → constant memory during inference. Tradeoff: lossy, can't perfectly recall distant past tokens like attention can.

**Diffusion Models:**
Generate full output by iteratively denoising from noise — no autoregressive loop, no KV cache. Start with noise, gradually clean it up into real text. Tradeoff: need many denoising steps per generation, still experimental for text.

**Linear Attention:**
Approximation of standard attention using Taylor expansion. Rewrites attention as a fixed-size running state — O(T) instead of O(T²). Great long range memory but approximate recall. BASED combines linear attention (long range) + sliding window (precise recent recall).

---

## Details


## Questions / Follow-up
