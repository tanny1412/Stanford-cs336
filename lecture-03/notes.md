# Lecture 3 — CS336: Architectures, Hyperparameters

## Summary
The original transformer architecture — normalization, residual connections, positional embeddings, and FFN design choices.

---

## Key Concepts

### 1. The Original Transformer (Attention is All You Need)

Key design choices:
- **Position embedding:** sines and cosines
- **FFN activation:** ReLU → `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`
- **Norm type:** LayerNorm, post-norm placement

---

### 2. Why Normalization?

As data flows through many layers, values can grow very large or very small → gradients explode or vanish → unstable training.

Normalization keeps values in a reasonable range at each layer.

What normalization does — make mean = 0, std = 1:
```
original:    [10, 20, 30]   mean=20, std=10
normalized:  [-1,  0,  1]
```

---

### 3. LayerNorm — Why Named "Layer"

Normalizes across the **features of one single token** (one layer's output for one example):
```
one token's vector:  [0.5, 2.1, -0.3, 1.8, ...]
                      ↑ normalize across all these values
```

Done independently for every token — no dependency on other samples.

**vs BatchNorm:** BatchNorm normalizes across the batch (many samples at once). LayerNorm normalizes across the layer's features for one sample.

Why LayerNorm over BatchNorm for transformers:
- BatchNorm struggles with variable sequence lengths
- BatchNorm struggles with small batch sizes
- LayerNorm works per token independently

---

### 4. Post-Norm vs Pre-Norm

**Post-norm** (original transformer): norm applied after sublayer + residual
```
output = LayerNorm(x + Sublayer(x))
```

**Pre-norm** (modern LLMs — GPT, Llama): norm applied before sublayer
```
output = x + Sublayer(LayerNorm(x))
```

"Add & Norm" in the diagram = residual (Add) then LayerNorm (Norm), after the sublayer → post-norm.

Modern LLMs use pre-norm because it trains more stably.

---

### 5. Residual Connections — Two Separate Benefits

```
output = x + MHA(x)
```

**Benefit 1 — Easier to learn:**

Without residual: each layer must produce the full correct output from scratch — hard.

With residual: each layer only learns a small correction on top of x — easier.

Example: if ideal output is `[2.1, 3.0, 1.8]` and x is `[2.0, 3.1, 1.7]`:
- Without residual: MHA must produce `[2.1, 3.0, 1.8]`
- With residual: MHA only needs to produce `[0.1, -0.1, 0.1]`

**Benefit 2 — Gradient highway:**

Gradient flowing back splits into two paths:
```
gradient coming back
        ↓
   ┌────┴────┐
   │         │
through    through
  MHA        x
(can get   (always
 small)     = 1)
```

The x path always passes gradient back with strength 1 — unchanged, no matter how many layers. Keeps gradients alive even in very deep networks.

---

### 6. Pre-Norm Flow

```
input x
   ↓
LayerNorm(x)              ← Norm first
   ↓
MHA(LayerNorm(x))         ← sublayer on normalized input
   ↓
x + MHA(LayerNorm(x))     ← Add original x back (bypasses norm)
```

Key: residual always adds back the **original x** — before the norm. x bypasses the norm entirely.

---

### 7. Why Pre-Norm is Preferred in Modern LLMs

**Post-norm problem:**
```
output = LayerNorm(x + MHA(x))
```
Gradient flows back through LayerNorm — distorts it. In very deep networks this causes instability at the start of training, requiring careful learning rate warmup.

**Pre-norm fix:**
```
output = x + MHA(LayerNorm(x))
```
The residual x bypasses the norm entirely → gradient highway is completely clean and untouched across all layers. Sublayer always receives normalized input so it never sees exploding values.

Summary:
- Post-norm: gradient highway disturbed by LayerNorm
- Pre-norm: gradient highway completely untouched → cleaner flow → more stable training

That's why GPT, Llama, and all modern LLMs use pre-norm.

---

### 8. Positional Embeddings — Why Needed

Transformers process all tokens in parallel — no inherent sense of order. Without positional info, "cat sat mat" and "mat cat sat" look identical.

Positional embeddings inject position information into each token so the model knows where it sits in the sequence.

---

### 9. Original — Sine/Cosine Embeddings

Fixed vectors added to token embeddings before the transformer:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

No learned parameters. Pattern lets model infer relative distances.

**Problem:** encodes absolute positions ("position 1 looks like this"). But language cares about relative positions — is this word 2 steps before that word, not whether it's at position 47 or 48.

---

### 10. Learned Positional Embeddings (GPT-2 style)

Learn a separate embedding for each position during training. Simple, works well, but has a hard cap — can't generalize to sequences longer than what was trained on.

---

### 11. RoPE (Rotary Position Embedding) — Modern Standard

Instead of adding a position vector to the token, RoPE **rotates** the query and key vectors in attention by an angle based on position.

Key insight: the dot product between two rotated vectors depends only on the **difference** in their positions, not absolute positions:
```
token at position 5  →  rotated by angle 5θ
token at position 8  →  rotated by angle 8θ

dot product  →  depends only on (8-5) = 3  ← relative distance
```

**Why RoPE over sine/cosine:**
1. **Relative positions** — model learns relationships between tokens, not absolute slots
2. **Extrapolation** — generalizes better to longer sequences than trained on
3. **No extra parameters** — computed not learned
4. **Works inside attention** — position encoded where it matters most (Q and K dot product), not bolted on at input

Used by: Llama, Mistral, GPT-4, and all modern LLMs.

---

### 12. RMSNorm vs LayerNorm

**LayerNorm does two things:**
1. Normalize — make mean = 0, std = 1
2. Re-scale — learned gamma and beta parameters

**RMSNorm drops mean subtraction:**
```
LayerNorm:  x̂ = (x - mean) / std   then scale with gamma, beta
RMSNorm:    x̂ = x / rms(x)         then scale with gamma only
```

Mean centering contributes very little to stability in practice. What matters is controlling the scale, not the mean.

RMSNorm is:
- Simpler — fewer operations
- Faster — ~10-15% faster
- Same quality empirically

Used by Llama, Mistral, and most modern LLMs.

---

### 13. SwiGLU vs ReLU

**ReLU:**
```
ReLU(x) = max(0, x)
```
Hard on/off gate. Negative input → output = 0, gradient = 0 → dying neuron problem.

**Swish** — smoother ReLU:
```
Swish(x) = x × sigmoid(x)
```
Smoothly curves near zero. Negative values contribute a little → gradients always flow.

**GLU (Gated Linear Unit):**
```
GLU(x) = Linear₁(x) × sigmoid(Linear₂(x))
```
One path produces values, other produces a learned gate (0-1) controlling how much passes through.

**SwiGLU = Swish + GLU:**
```
SwiGLU(x) = Linear₁(x) × Swish(Linear₂(x))
```

Why better than ReLU:
1. No dying neurons — smooth curve, gradients always flow
2. Gating — network learns to selectively activate, more expressive than hard on/off
3. Empirically better — Google PaLM showed consistent quality improvement over ReLU

Intuition: ReLU = blunt on/off switch. SwiGLU = smooth learnable gate.

---

### 14. d_model vs d_ff

**d_model** — size of every token's vector throughout the transformer. Every token is always this size.

**d_ff** — hidden dimension inside the FFN. Expands then shrinks:
```
token (d_model) → expand → hidden (d_ff) → shrink → token (d_model)
```

**Why d_ff = 4 × d_model:**
Expansion gives FFN more capacity to compute. 4× is empirically what works best — consensus across GPT, BERT, original transformer. Think of d_model as "working memory" and d_ff as "scratch space."

---

### 15. SwiGLU — Name Clarification

**SwiGLU = Swish + GLU** (not Swish + GeLU)

- **Swi** → Swish activation
- **GLU** → Gated Linear Unit

**GeLU** is separate — used in BERT and GPT-2:
```
GeLU(x) = x × Φ(x)    where Φ is the Gaussian CDF
```

Both smoother than ReLU, both solve dying neuron problem. SwiGLU adds gating on top → more expressive.
- GeLU → BERT, GPT-2
- SwiGLU → Llama, PaLM (newer, generally better)

---

### 16. Multi-Head Attention — Why Not More Expensive

```
d_model = num_heads × head_dim
```

Multiple heads cost the same as one big head because:
```
compute XQ → shape (n, d_model)    ← one big matrix multiply
reshape    → shape (h, n, d/h)     ← split into h heads for free
```

One big matrix multiply, then reshape. Total matrix sizes stay the same — heads are free.

Why multiple heads: each head attends to different aspects (syntax, coreference, position) — multiple views of the sequence at no extra cost.

---

### 17. Aspect Ratios — Deep vs Wide

**d_model / n_layers** tells you how wide each layer is relative to depth.

| Model | d_model/n_layers |
|-------|-----------------|
| BLOOM | 205 |
| PaLM (540B) | 156 |
| GPT3/Mistral/Qwen | 128 |
| Llama/Llama2/Chinchilla | 102 |
| GPT2 | 33 |

**Sweet spot: 100-128.** Modern models all cluster here.

Rule of thumb: `n_layers ≈ d_model / 128`

Too wide + shallow → capacity without depth to build complex reasoning.
Too narrow + deep → can't represent enough information per layer.

---

### 18. Gradient Norm Spikes — Why Bad Even if Loss Looks Smooth

**L2 norm of gradient** = overall size of all gradients combined. A spike = gradients suddenly very large.

When gradients spike:
```
W = W - lr × gradient   ← giant gradient → giant weight update
```

Weights jump to a bad region. Loss may recover quickly (looks smooth) but internally weights took a violent detour — can destabilize future training, cause later loss spikes, waste training steps recovering.

**Gradient clipping** — cap gradient norm at a max value (e.g. 1.0) before the update step. Cuts off spikes before they cause damage.

---

### 19. Logit Soft-Capping

Another stability trick — keep attention logits bounded using tanh:

```
logits = soft_cap × tanh(logits / soft_cap)
```

Keeps logits between -soft_cap and +soft_cap (e.g. 50 for attention, 30 for final layer).

**Why tanh over hard clipping:** tanh smoothly compresses large values — nothing abruptly cut off, better gradient flow.

**Tradeoff:** limits what attention scores can express — sometimes a token genuinely deserves a very high score. May hurt performance.

Used by Gemma models. Not universally adopted because of the performance tradeoff.

---

### 20. MQA and GQA — Reducing KV Cache Memory

**KV Cache:** during inference, K and V for all previous tokens are cached to avoid recomputation. Grows with sequence length, stored in GPU memory.

**MHA (Multi-Head Attention):** every head has its own Q, K, V:
```
8 heads → 8 sets of K and V cached   ← large cache
```

**MQA (Multi-Query Attention):** all heads share one K and V, each head has its own Q:
```
8 heads → 1 set of K and V cached   ← 8× smaller cache
```
Faster inference, but quality drops — less expressiveness.

**GQA (Grouped Query Attention):** middle ground — heads grouped, each group shares K and V:
```
8 heads, 4 groups → 4 sets of K and V cached   ← 2× smaller
```
Better quality than MQA, smaller cache than MHA. Used by Llama 2, Mistral.

Why cache size affects speed: smaller KV cache = fewer memory reads = faster generation, especially for long sequences.

---

### 21. Sparse / Sliding Window Attention

**Problem:** full attention is quadratic — every token attends to every other token:
```
seq length 1000  → 1M computations
seq length 10000 → 100M computations   (4× for 2× sequence length)
```

**Fix:** each token only attends to a subset of tokens.

**(a) Standard Transformer** — full attention, every token sees every other (full triangle).

**(b) Sparse Transformer (strided)** — two patterns:
- Local window → attend to nearby tokens
- Strided → attend to every nth token across sequence
Lets information travel long distances (strided) while keeping local context (window).

**(c) Sparse Transformer (fixed)** — fixed patterns instead of strided jumps.

**Tradeoff:** cheaper compute but some token pairs never directly attend → less expressive. Information still travels indirectly across layers so quality doesn't drop much.

Used by GPT-3, Longformer for long context efficiency.

---

## Questions / Follow-up
