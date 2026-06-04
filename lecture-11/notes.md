# Lecture 11 — CS336: Scaling Laws 2

## Summary
Case studies of how real labs (Cerebras, MiniCPM, DeepSeek) implement scaling laws in practice. Key new ideas: WSD learning rate schedule and muP derivation. Core lesson: scaling laws are a tool, not a magic formula — labs use them to get hyperparameters in the right order of magnitude, not exact values.

---

## Key Concepts

### 1. How Labs Actually Scale — Common Ingredients

Three approaches seen across Cerebras-GPT, MiniCPM, DeepSeek:

**Cerebras-GPT + MiniCPM approach:**
- Use muP to make hyperparameters (LR, init) stable across scale
- Tune hyperparameters on tiny proxy model (40M params)
- Scale up using muP — optimal LR transfers automatically
- Use WSD schedule for cheap Chinchilla analysis

**DeepSeek approach (no muP):**
- Assume most hyperparameters don't change with scale
- Directly fit scaling laws on batch size and LR across different compute scales
- Use isoflops analysis to replicate Chinchilla and get token/model size tradeoffs
- More direct but requires strong belief in scaling laws

**Both approaches share:**
- Fix aspect ratio (d_model / n_layers), only scale total parameter count
- Replicate Chinchilla isoflops analysis to verify token-to-parameter ratio
- Use WSD or similar schedule to avoid N² training runs

---

### 2. WSD Learning Rate Schedule (Warmup-Stable-Decay)

**Problem with cosine LR:** each data target needs its own LR schedule — can't reuse checkpoints. Doing Chinchilla analysis requires N² training runs.

**WSD solution:**
```
Phase 1: Warmup → ramp up LR
Phase 2: Stable → flat LR (long middle phase, reusable)
Phase 3: Decay  → rapid cooldown to near zero
```

Rewind to earlier checkpoint in stable phase → apply short decay → get model at any data size. Turns N² runs into ~1 run.

Training curve looks weird (loss stays high during stable, drops sharply at decay) — this is normal. Cooldown phase is where most loss improvement happens. Final loss matches cosine but far cheaper for scaling experiments.

Popularized by MiniCPM, now widely adopted.

---

### 3. Chinchilla 20:1 Is Just a Starting Point

```
Chinchilla (2022): ~20 tokens per parameter
Llama 3 (2024):    ~39 tokens per parameter
MiniCPM (2024):    ~192 tokens per parameter
```

Ratio keeps shifting with better data quality and architectures. Don't treat 20:1 as a constraint — recent models consistently train at much higher token-to-parameter ratios. Small models trained longer = cheaper to serve at inference.

---

### 4. muP — Why It Works (Conceptual, Skip Math)

**The problem:** as model width increases, optimal LR shifts downward. Tune LR on small model → scale up → LR is now wrong → expensive retuning.

**muP fixes this with two conditions:**
1. Activations at initialization should stay constant as width grows (don't blow up or vanish)
2. After one gradient step, activation updates should also stay constant

These two conditions force specific initialization and per-layer LR scaling rules:
```
Initialization: 1 / sqrt(fanin)   ← same as Kaiming, already standard
Learning rate:  1 / fanin          ← per layer, different from global constant LR
```

**The practical difference from standard parameterization:**
- Initialization: already roughly correct if using Kaiming
- LR: normally one global constant → muP uses per-layer LR scaled by 1/width → this is the real change

**Result:** optimal LR stays the same as model scales up. Tune once on small model, use forever.

Used by: Cerebras-GPT, MiniCPM, Llama 4 (Meta calls it similar thing internally).

---

### 5. When muP Breaks

From empirical ablations (large scale muP study):

**Works fine with:**
- Different nonlinearities (SwiGLU, squared ReLU)
- Batch size variations (4x up or down)
- Different query initializations
- Embedding scaling variations

**Breaks with:**
- Learnable gains in LayerNorm (remove biases and it works again)
- Very exotic optimizers (Lion etc.) — muP is designed for AdamW
- Strong weight decay

**Key validation:** optimal LR remained stable scaling from small model all the way to 10B parameters in one large-scale experiment.

---

### 6. Scaling Law Reliability — What's Trustworthy vs Not

From looking across multiple labs:

**Very reliable:**
- Isoflops / Chinchilla analysis — fits always look clean, consistent across labs
- Token-to-parameter optimal ratio direction (even if exact number varies)

**Noisier / more suspicious:**
- LR scaling laws (DeepSeek's fit looked tenuous — could have fit a horizontal line)
- Exact hyperparameter predictions from scaling

**Lesson:** use scaling laws to get order of magnitude right, not exact values. The isoflops analysis is the most trusted piece. Hyperparameter scaling is noisier — that's why labs use muP to avoid needing it.

---

---

## Glossary

**WSD Schedule (Warmup-Stable-Decay):** LR schedule with three phases — warmup, flat stable phase, rapid decay. Stable phase can be rewound to simulate different data sizes without retraining, making Chinchilla analysis cheap. Now standard in most labs.

**muP (Maximal Update Parametrization):** Initialize and set per-layer LRs so optimal LR stays the same as model width scales up. Key change: instead of one global LR, each layer gets its own LR scaled by 1/width. Tune on small model, works at any scale.

**Isoflops Analysis:** Fix a compute budget, train models of different sizes, find which size gives lowest loss. The minimum gives optimal parameter-to-token ratio for that budget. Most reliable piece of scaling law work — always fits cleanly.

**Chinchilla Ratio:** Optimal tokens-to-parameters ratio. Chinchilla said 20:1 but keeps shifting — Llama 3 uses 39:1, others go higher. Treat as a starting point not a rule. Direction is clear: more tokens per parameter than Chinchilla suggested.

**Proxy Model:** A tiny model (e.g. 40M params) used to tune hyperparameters cheaply before scaling up. Works because muP makes those hyperparameters transfer to larger models.

---

## Questions / Follow-up
