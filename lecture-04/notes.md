# Lecture 4 — CS336: Mixture of Experts

## Summary
Mixture of Experts (MoE) — how replacing a single dense FFN with multiple expert FFNs gives more capacity at the same compute cost.

---

## Key Concepts

### 1. Problem with Dense FFN Scaling

Every token goes through the same FFN every time:
```
token → W1 (expand to d_ff) → W2 (shrink to d_model) → output
```
Every parameter is used for every token — "dense." Making the model better means making it bigger, but bigger = more compute for every single token. Cost grows linearly with parameters.

---

### 2. MoE — Core Idea

Instead of one FFN, have many FFNs called **experts**. A **router** decides which 1-2 experts each token goes to:
```
Expert 1: W1, W2
Expert 2: W1, W2
...
Expert N: W1, W2

token "cat"   → router → Expert 3, Expert 7
token "Paris" → router → Expert 1, Expert 5
```

---

### 3. Why "Sparse"

Out of say 8 experts, each token only uses 2 → 6 experts completely inactive per token. Most parameters are OFF at any given moment → sparse.

---

### 4. Why MoE is Powerful

Same compute per token, but access to a much larger pool of parameters:
```
Dense 7B model:  7B params, all active every token
MoE equivalent: 56B total params, only 7B active per token
```
More capacity, same compute. More knowledge baked in across all experts combined.

**Specialization:** experts naturally divide the work — one gets good at grammar, another at facts, another at code. Better than one FFN trying to do everything.

Analogy: Dense = one doctor for everything. MoE = hospital with specialists, each patient routed to the right 1-2 specialists.

---

### 5. Combining Expert Outputs

Router assigns confidence weights alongside expert selection:
```
router scores: Expert 3 = 0.7, Expert 7 = 0.3

output = 0.7 × Expert3(token) + 0.3 × Expert7(token)
```
Weighted sum of the two expert outputs → passed to the next layer like a normal FFN output. Rest of the transformer doesn't know MoE happened.

---

### 6. Expert Parallelism

Each expert stored on a different device:
```
Device 1 → Expert 1
Device 2 → Expert 2
...
Device 8 → Expert 8
```
Router sends token to Expert 3 and 7 → only Device 3 and 7 do compute → other 6 sit idle. Memory and compute both distributed.

**Challenge — load balancing:** if router sends many tokens to the same expert, some devices are overloaded while others sit idle. Fixed with a **load balancing loss** — a penalty encouraging the router to distribute tokens evenly across all experts.

**Tradeoff:**
- Tokens must be sent across devices → network communication overhead
- All experts must be loaded in GPU memory even though only 2 fire per token → memory expensive
- Trickier to deploy than dense models despite same per-token compute

Used by: Mixtral, GPT-4 (likely).

---

### 7. Router — What It Actually Is

The router is just a small **linear layer** (matrix multiply):
```
router = Linear(d_model, num_experts)

token vector (d_model)
      ↓  multiply by router weights
scores (num_experts)     e.g. [0.2, 0.8, 0.1, 0.5, ...]
      ↓  softmax
probabilities
      ↓  pick top-2
Expert 6 (0.20) + Expert 2 (0.18)  ← these fire
```

Router weights are learned during training via backprop — starts random, gradually learns which token types belong to which experts.

**Collapse problem:** early training router is random. As it learns, one expert can dominate → gets more tokens → learns faster → gets even more tokens → others starve. Load balancing loss prevents this.

---

### 8. Linear Transformations

A linear layer = just a matrix multiply:
```
output = input × W
```
No activation function. Adding ReLU/SwiGLU after makes it nonlinear.

---

### 9. Top-K Routing in Detail

**s_i,t — router score for expert i, token t:**
```
s_i,t = Softmax(u_t^T × e_i)
```
Dot product of token vector with each expert's embedding → softmax → probabilities.

**g_i,t — gate (who actually fires):**
```
g_i,t = s_i,t   if expert i is in top-K
        0        otherwise
```
Only top-K experts get non-zero gate → sparse.

**h_t — final output:**
```
h_t = Σ (g_i,t × FFN_i(u_t)) + u_t
```
Weighted sum of top-K expert outputs + residual. Most gates = 0 so most experts contribute nothing.

**Two variants:**
- **DeepSeek/Grok/Qwen** — softmax over ALL experts first, then pick top-K
- **Mixtral/DBRX/DeepSeek v3** — pick top-K first, then softmax only over those K

Difference: how scores are normalized — across all experts vs only among the winners.

---

### 10. MoE Variants — DeepSeek and Chinese LMs

**(a) Conventional Top-2 Routing:** N experts, router picks top 2. Simple.

**(b) Fine-grained Expert Segmentation:** Many smaller experts, pick more of them (e.g. top-4 from 2N experts). More granular specialization — tiny specialists mixed and matched more precisely per token.

**(c) Shared Expert Isolation (DeepSeekMoE):**
- **Shared experts** (always on) — handle general knowledge every token needs
- **Routed experts** (router picks top-K) — handle specialized knowledge

Router no longer wastes slots on general things — all K slots go to specialists. Shared experts act as always-available foundation. Used by DeepSeek, Qwen.

---

### 11. Fine-Grained Ratio

**Fine-grained ratio** = expert size relative to a standard FFN.

Ratio of 1/4 → each expert is 4× smaller than a standard FFN. Same total parameters, but more experts to choose from → more flexibility.

**Expert routing table for recent MoEs:**

| Model | Routed | Active | Shared | Fine-grained ratio |
|-------|--------|--------|--------|-------------------|
| Mixtral | 8 | 2 | 0 | — |
| DBRX | 16 | 4 | 0 | — |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OLMoE | 64 | 8 | 0 | 1/8 |
| Llama 4 | 128 | 1 | 1 | 1/2 |

- **Routed** — how many experts to route between
- **Active** — how many fire per token (top-K)
- **Shared** — always-on experts for every token
- **Fine-grained ratio** — how small each expert is vs standard FFN

Trend: newer models go finer-grained — more experts, each smaller, more active at once.

---

### 12. Heuristic Load Balancing Loss

Forces router to distribute tokens evenly across experts.

**Two quantities per expert i:**

**f_i** — fraction of tokens actually dispatched to expert i:
```
f_i = tokens routed to expert i / total tokens in batch
```

**P_i** — fraction of router probability allocated to expert i:
```
P_i = average softmax score for expert i across all tokens
```

**Auxiliary loss:**
```
loss = α × N × Σ(f_i × P_i)
```
If one expert dominates → both f_i and P_i high → product large → loss high → penalized. Gradient pushes router to lower P_i for overused experts.

More frequent use = stronger downweighting. Self-correcting mechanism.
`α` = small hyperparameter controlling balancing strength vs main loss.

---

### 13. Auxiliary Loss and Z-Loss

**Auxiliary loss** = extra loss added on top of main loss to nudge model towards desired behavior the main loss doesn't enforce:
```
total loss = main loss + α × auxiliary loss
```

**Main loss** — cross entropy, how wrong was the prediction.

**Load balancing auxiliary loss** — nudges router to distribute tokens evenly.

**Z-loss** — penalizes router logits from getting too large:
```
z_loss = β × log²(Σ exp(logits))
```
Keeps logits in reasonable range → softmax doesn't become too peaky → more stable routing.

**The pattern:** whenever training has a stability problem or desired behavior the main loss ignores, add a small auxiliary loss with tiny weight. Model ends up optimizing multiple objectives at once:
- Predict next token well (main loss)
- Keep experts balanced (load balancing loss)
- Keep logits from exploding (z-loss)

---

### 14. Z-Loss — Why Named "Z"

"Z" = standard math notation for pre-softmax logits:
```
z = router logits (raw scores before softmax)
softmax(z) = probabilities
```
Z-loss penalizes the z values from getting too large. Named after the notation, not anything special.

---

### 15. Upcycling — MoE from a Dense Model

**Question:** can we use a pre-trained dense model to initialize a MoE model instead of training from scratch?

**How it works:**
Copy the trained dense FFN weights into ALL experts:
```
Dense FFN weights → copied into Expert 1
                  → copied into Expert 2
                  → copied into Expert 3 ...
```
All experts start identical. Continue training → experts gradually diverge and specialize. Router initialized randomly and trained from scratch.

**Why it works:** starts from a model that already predicts text well. Training only needs to teach specialization — much less work than learning everything from scratch.

**Result:** upcycling reaches same accuracy as a much larger dense model in far less compute. Consistently outperforms training MoE from scratch given the same compute budget.

**Tradeoff:** need a good dense model to start with. But if you have one, upcycling is very compute-efficient.

---

### 16. DeepSeek MoE v3 — Aux-loss-free + Seq-wise Aux

**Aux-loss-free balancing:**

Problem with auxiliary loss: it conflicts with the main loss — router wants to pick the best expert but aux loss forces even spread. Two objectives fighting each other → hurts quality.

DeepSeek v3 removes auxiliary loss entirely. Instead uses a **bias term** per expert:
```
routing score = sigmoid(u_t · e_i) + bias_i
```
- Expert getting too many tokens → bias_i decreases → fewer tokens routed to it
- Expert getting too few tokens → bias_i increases → more tokens routed to it

Bias updated directly based on load, not through backprop. Main training loss stays completely clean — no conflicting objectives. Better quality + balanced routing.

**Seq-wise auxiliary:**

Original load balancing measures balance across the whole batch — tokens within one sequence could all go to the same expert (fine for batch average, but creates compute hotspots).

Seq-wise aux measures balance **within each individual sequence** — forces even distribution at sequence level, not just batch level. Finer-grained load control.

---

### 17. MLA (Multi-Head Latent Attention)

**Problem:** KV cache stores full K and V for every token — very large, grows with sequence length.

**Core idea:** compress K and V into a small latent vector, only cache that:
```
c = h × W_DKV        ← compress input into small latent
K = c × W_UK         ← decompress latent → K (only when needed)
V = c × W_UV         ← decompress latent → V (only when needed)
```

Cache only `c` during inference — much smaller than full K and V.

---

**What is a projection:**
Just a matrix multiply that changes dimension:
```
input (d_model) × W_Q (d_model, d_head) = Q (d_head)
```
Nothing fancy — linear transformation to a new size.

---

**Why merge W_UK into Q projection:**

Two chained matrix multiplies = one merged matrix:
```
K = h × W_DKV × W_UK = h × W_merged
```

In attention you compute Q·K^T. Absorb W_UK into the Q side:
```
Q · K^T = (h × W_Q) · (c × W_UK)^T
        = (h × W_Q × W_UK^T) · c^T
        = Q_merged · c^T
```

Now dot product is between Q_merged and cached latent c directly — never need to decompress c into full K. Cache c, compute slightly modified Q. Full K reconstruction skipped entirely.

**Memory saving:** cache small latent c instead of full K and V → dramatic reduction in KV cache size.

---

**RoPE conflict:**
RoPE needs to rotate actual Q and K vectors, not the compressed latent. Solution: keep a few "non-latent" key dimensions at full size just for RoPE, while everything else stays compressed.

---

## Questions / Follow-up
