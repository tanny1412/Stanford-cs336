# Lecture 9 — CS336: Scaling Laws 1

## Summary
Scaling laws — the mathematical relationship between compute, model size, and data that lets you make optimal training decisions before spending millions of dollars.

---

## Key Concepts

### 1. What Scaling Laws Actually Are

Core question: given a fixed compute budget, what's the optimal model size and number of training tokens?

Scaling laws are not just "follow the graph" — they're a mathematical relationship between:
- **C** — compute budget (FLOPs)
- **N** — model size (parameters)
- **D** — dataset size (tokens)

They let you predict model loss before training and answer:
- What size model should I train with $X compute?
- If I 10× my compute, how should I split it between bigger model vs more data?
- What loss will this model achieve?

---

### 2. Chinchilla Finding (DeepMind, 2022)

Most labs (including OpenAI with GPT-3) were training models that were too big and undertrained.

GPT-3: 175B parameters, 300B tokens → compute-suboptimal.

**Chinchilla result: optimal tokens ≈ 20× number of parameters**

```
1B parameter model  → train on ~20B tokens
7B parameter model  → train on ~140B tokens
70B parameter model → train on ~1.4T tokens
```

You get better performance by training a smaller model on more tokens with the same compute budget than training a large model on fewer tokens.

---

### 3. Why Log Axes in Scaling Law Plots

Numbers span many orders of magnitude (1M to 100B params). On linear scale, small models look identical and large models dominate. Log scale makes each order of magnitude equally spaced — patterns visible across the full range.

**Log-log means both axes are in log scale:**
```
X axis: log(compute) or log(model size) or log(tokens)
Y axis: log(loss)
```

**Why power laws become straight lines on log-log:**

The relationship is: `loss = a × compute^b`

Take log of both sides:
```
log(loss) = log(a) + b × log(compute)
          = mx + c  ← straight line!
```

On log-log axes, a power law = straight line. Easy to fit, easy to extrapolate to much larger models you haven't trained yet. That's why scaling law papers always use log-log plots.

---

### 4. BLEU vs ROUGE

Both are n-gram overlap metrics — measure how many words/phrases match between model output and a reference text.

- **BLEU** → machine **translation** (Bilingual Evaluation Understudy)
- **ROUGE** → **summarization** (Recall-Oriented Understudy for Gisting Evaluation)

Memory trick: BLEU = Bilingual = translation between languages. ROUGE = Gisting = condensing/summarizing.

---

### 5. Derivative → Gradient → Jacobian → Hessian

**Derivative** — single input, single output:
```
f(x) → scalar
df/dx → one number
```

**Gradient** — multiple inputs, single output (like loss function):
```
f(x₁, x₂, x₃) → scalar
∇f = [df/dx₁, df/dx₂, df/dx₃]  → vector
```
This is what backprop computes — gradient of loss w.r.t. each parameter.

**Jacobian** — multiple inputs, multiple outputs:
```
f(x₁, x₂) → [y₁, y₂, y₃]
J = matrix of all partial derivatives
    [[dy₁/dx₁, dy₁/dx₂],
     [dy₂/dx₁, dy₂/dx₂],
     [dy₃/dx₁, dy₃/dx₂]]
```
How each output changes w.r.t. each input. Used in backprop through vector-valued functions.

**Hessian** — second derivatives, multiple inputs, single output:
```
H = matrix of second partial derivatives (curvature of loss surface)
    [[d²f/dx₁², d²f/dx₁dx₂],
     [d²f/dx₂dx₁, d²f/dx₂²]]
```
Tells you how the gradient itself is changing. Used in second-order optimizers but too expensive for large models (N² parameters to compute).

**Hessian-vector product** — trick to use curvature information without computing full Hessian matrix. Used in some advanced optimizers.

---

### 6. Asymptotic

Describes behavior as something approaches a limit, usually infinity. "In the long run" or "at the limit."

**In ML:**
- **Asymptotic performance** — how good does the model get as you keep scaling? Loss keeps improving then eventually flattens — that floor is the asymptotic behavior
- **Asymptotic complexity** — how does compute grow as input size grows? Attention is O(n²) — doubling sequence length quadruples compute
- **Asymptotic loss** — theoretical minimum loss you'd approach if trained forever on infinite data. You approach it but never quite reach it

"Asymptotically optimal" = optimal when things get very large.

---

### 7. Scaling Laws — Power Law Relationships (Log-Log Graphs)

All three graphs are log-log scale (both axes in log). Straight lines on log-log = power law relationships.

**Loss vs Compute:** each model size is a different line, all roughly straight → power law
**Loss vs Dataset Size:** `L = (D/5.4×10¹³)^(-0.095)` — more tokens = lower loss
**Loss vs Parameters:** `L = (N/8.8×10¹³)^(-0.076)` — bigger model = lower loss

**Key insight:** all three are power laws → loss is predictable. Fit lines on small experiments → extrapolate to predict what loss a 100B model will achieve before training it.

Power laws hold even when train ≠ test domain — robust across different datasets.

---

### 8. Kaplan 2020 — Original Scaling Laws Paper (OpenAI)

First paper to show clean power law relationships between loss, compute, model size, and data.

**Key recommendation:** given fixed compute, scale model size much faster than data:
```
10× compute → 5.5× bigger model, only 1.8× more data
```
Conclusion: bigger models matter more than more data.

**Why Chinchilla (2022) overturned it:**
Hoffmann et al. (DeepMind) ran more careful experiments at larger scale and found:
```
10× compute → ~3× bigger model AND ~3× more data (equal scaling)
Optimal: 20 tokens per parameter
```

GPT-3 followed Kaplan — 175B params, 300B tokens. By Chinchilla: should have trained on 3.5T tokens. GPT-3 was massively undertrained.

**Why Kaplan was wrong:** ran experiments at smaller scale and extrapolated. The optimal model size to data ratio shifts as you scale — only visible at larger scale which Chinchilla ran.

---

### 9. Why Scaling Laws Exist — Toy Example

Simple example: estimate the average of n numbers.

Error = σ² / n
- σ² = how spread out the data is
- n = how many samples

More samples → smaller error. Double samples → halve error.

Take log of both sides:
```
log(Error) = -log(n) + 2log(σ)
```
Straight line on log-log plot → power law. Error decreases as a power law with more data.

**Key insight:** scaling laws aren't magic. Even the simplest estimation problem (computing an average) has a power law between error and data size. When LLMs show power law scaling, it's a fundamental property of learning from samples — not something special about neural networks.

---

### 10. Gaussian/Normal Distribution and ~ Notation

**`~` means "drawn from" or "sampled from":**
```
x ~ N(μ, σ²)
```
x is a random number from a Normal distribution with mean μ and variance σ².

**What Normal distribution is:** bell curve — most values cluster around mean, fewer far away:
```
μ = center of the bell (average value)
σ = how wide the bell is (standard deviation)
σ² = variance (σ squared)
```

**The 3σ rule:** ~99.7% of values fall within 3 standard deviations of the mean:
```
range = μ ± 3σ
```

**Examples:**
```
x ~ N(0, 1):  μ=0, σ=1  → most values between 0 ± 3×1 = -3 to 3
x ~ N(5, 4):  μ=5, σ²=4 → σ=√4=2 → most values between 5 ± 3×2 = -1 to 11
```

**Why it shows up everywhere in ML:**
- Weight initialization: `W ~ N(0, 1/n)` — Xavier initialization
- `torch.randn()` samples from N(0,1)
- Errors in predictions often modeled as Gaussian
- Noise in data assumed to follow Gaussian

**Universal reading rule:** whenever you see `~ N(μ, σ²)` just read it as "random number with average μ and spread σ."

---

### 11. Common ML Notation Reference

```
~         → drawn from / sampled from       x ~ N(0,1)
∈         → belongs to / is in              x ∈ [0,1]
∑         → sum of                          ∑ xᵢ
∏         → product of                      ∏ xᵢ
∝         → proportional to                 y ∝ x²
≈         → approximately equal to          π ≈ 3.14
argmax    → value that maximizes            argmax f(x)
argmin    → value that minimizes            argmin f(x)
E[x]      → expected value (average) of x
Var[x]    → variance of x
∇         → gradient                        ∇L = gradients of loss
||x||     → norm (size) of vector x
```

---

### 12. Why Neural Network Scaling Exponents Are So Small

Classical models (regression etc.) have error ∝ 1/n → slope of -1 on log-log.

Real neural network scaling laws are much shallower:
```
Machine translation:  error ∝ n^(-0.11)
Language modeling:    error ∝ n^(-0.095)
```

Much slower improvement than classical models. Why?

**The explanation — high dimensionality:**

Error in d dimensions: `Error ≈ n^(-1/d)`
```
2D:  error = n^(-1/2)   → fast improvement
10D: error = n^(-1/10)  → slow improvement
∞D:  error = n^(-0.095) → very slow improvement
```

Language is essentially infinite dimensional — every word, context, concept adds a new dimension. So adding more data helps very slowly.

**The takeaway:**
```
Simple model (low dimensions):     add 10× data → error drops a lot
Neural network (high dimensions):  add 10× data → error barely drops
```

This is why LLMs need trillions of tokens — not because neural nets are bad learners, but because language is genuinely a very high dimensional problem. Improvement per extra token is tiny by mathematical necessity.

---

### 13. Data Composition and Distribution Shift

**Data composition affects offset, not slope:**

Different data sources (Wikipedia, Books, Common Crawl) produce parallel lines on loss vs model size chart — same slope, different heights.
- Rate of improvement from scaling = same regardless of data source
- Starting point (offset) differs — higher quality data = lower loss at every model size

Better data shifts the whole curve down but doesn't change how fast scaling helps.

**Distribution shift:**
When training data doesn't match test data, error intercept goes up. More diverse training data = smaller distribution shift at deployment. Why frontier labs scrape the entire internet — not just more tokens but more distributions.

**Counterintuitive finding:** a 56% mix of out-of-distribution data (q=0.56) gives lower loss than pure in-distribution (q=0). Moderate out-of-distribution data forces the model to generalize more broadly instead of overfitting to one source. Too much hurts, but a moderate mix is better than pure in-distribution.

**Practical summary:**
```
More data (same quality)  → moves along scaling curve (lower loss)
Better data quality       → shifts whole curve down
More diverse data         → reduces distribution shift at deployment
```

---

### 14. Scaling Laws Under Data Repetition

**Key finding:** up to ~4 epochs of repeating data is almost as good as seeing new data. Beyond that, rapidly diminishing returns.

Left chart: loss keeps improving up to 4 epochs, then flattens fast.
Right chart: optimal point with finite data loses only tiny amount vs infinite fresh data (loss 2.359 vs 2.376).

**Effective data formula:** `D' = U_D + U_D × R_D*(1 - e^(-R_D/R_D*))` — repeated data worth less than fresh but not worthless.

**Practical takeaway:** if you've exhausted unique data, repeating up to ~4 epochs is fine. Beyond that, find more diverse data. This is why data quality and diversity matter — you literally run out of good data at some point.

---

### 15. Data Selection Should Be Adaptive to Scale

Web data is non-homogeneous — some buckets high quality (A, B, C), others lower quality (D, E).

**Quality-Quantity Tradeoff (QQT):**
- Small compute → use only high quality data (aggressive filtering, Pool 1)
- Large compute → expand to include lower quality buckets (less filtering, Pool 2)

The optimal data pool changes with compute budget — no single best dataset. Right dataset depends on how much you can train.

**Estimated scaling curves:**
```
Small compute:  highly aggressive filtering is best
Medium compute: mildly aggressive filtering is best
Large compute:  less aggressive filtering is best
```

At large scale, more data beats cleaner data — less filtered data curve eventually wins.

**Takeaway:** don't pick a fixed dataset and scale forever. As compute budget grows, expand data pool to include progressively lower quality sources. Optimal data strategy is adaptive to scale.

---

### 16. Data Scaling Laws — Conclusion

More data always helps but with diminishing returns (power law, very slow because language is high dimensional).

- **Data quantity** → moves along the scaling curve (lower loss)
- **Data quality** → shifts the whole curve down (better starting point, same slope)
- **Data diversity** → reduces distribution shift at deployment
- **Data repetition** → fine up to ~4 epochs, rapidly diminishing after that
- **Data selection** → optimal strategy changes with scale — small compute wants quality, large compute wants quantity

There is no single best dataset. The right data depends on your compute scale. As you scale up compute, expand your data pool to include progressively lower quality sources.

---

### 17. Scaling Laws for Model Engineering

**Questions scaling laws answer:**
1. LSTMs vs Transformers — which architecture scales better?
2. Adam vs SGD — which optimizer scales better?
3. Train longer vs train bigger — given fixed compute, which is optimal?
4. More data vs more GPUs — where should you spend your budget?

**How to answer without training big models:**
Train small versions, measure where they fall on the scaling curve, extrapolate to large scale. Saves enormous compute for architecture and hyperparameter decisions.

This is how labs make architectural decisions — not by training 70B models to compare, but by running small scaling experiments and reading the curves. The architecture with the better slope at small scale wins at large scale.

---

### 18. Depth vs Width — Number of Layers

- 1→2 layers = huge improvement
- 3, 6, 6+ layers = barely different at small scale (<10⁷ params)
- At large scale (10⁷+) = more layers start mattering again
- Sweet spot: d_model/n_layers ≈ 100-128 (Lecture 3)

---

### 19. Embedding vs Non-Embedding Parameters

**Embedding parameters** — lookup table converting token IDs to vectors:
```
vocab_size × d_model = 50,000 × 4096 = 200M params (just a lookup table)
```
No computation — just memorization. Don't scale the same way.

**Non-embedding parameters** — attention weights, FFN weights, LayerNorm — actually do computation and reasoning.

Scaling law papers always plot against non-embedding parameters — embedding params distort the relationship. A bigger vocab just means a bigger lookup table, not a smarter model.

Example — Llama 3 8B:
```
Total:         8B
Embedding:    ~0.5B
Non-embedding: ~7.5B  ← what matters for reasoning
```

---

### 20. Hyperparameter Shape Barely Matters

When total non-embedding parameter count N is fixed, exact model shape barely affects loss:
- FFN ratio (8 to 64): loss varies only a few percent
- Aspect ratio: can vary by 40× while only slightly impacting performance
- Attention head dimension: 22% extra compute compensates for 1% loss increase

**Key insight:** focus on getting parameter count and data right — those matter far more than architecture shape. You can deviate significantly from "consensus" hyperparameters without hurting much.

---

### 21. Critical Batch Size

**Critical batch size B*** = min examples needed for target loss / min steps needed for target loss

The batch size where you get maximum useful information per example. Beyond B*, examples give redundant gradient information.

```
Below B*: perfect scaling — doubling batch halves steps needed
Above B*: ineffective scaling — doubling batch doesn't halve steps, redundant gradients
```

Training at exactly B* is optimal — smaller is too slow (too many steps), larger is wasteful (redundant gradients). Finding B* matters for large scale training to be at the knee of the curve.

---

### 22. Critical Batch Size Changes During Training

As loss decreases, critical batch size increases:
```
Early training (high loss) → small batches fine, every example teaches something new
Late training (low loss)   → need bigger batches, gradients become similar across examples
```

**Why:** early in training gradients are noisy and varied — small batches give enough signal. Late in training the model has learned most patterns — need bigger batches to get useful gradient updates.

**Practical takeaway:** don't use fixed batch size throughout training. Some labs increase batch size during training to stay near B* — smaller early, larger late.

---

### 23. muP — Scale-Aware Learning Rate

**The problem:** optimal learning rate shifts as model width increases. Tune LR on small model → scale up → LR is now wrong → expensive retuning at every scale.

**muP (Maximal Update Parametrization, Yang et al. 2022):**
A specific initialization and LR scaling scheme where optimal LR stays the same regardless of model width:
```
Normal: tune LR on small model → scale up → LR wrong → retune
muP:    tune LR on small model → scale up → LR still optimal → no retuning
```

Key rule: learning rate for matrix-like parameters scales as `1/r` where r = width multiplier. As model gets wider, LR shrinks proportionally — automatically.

**Why it matters:** saves enormous compute — no need to retune hyperparameters at each scale. Tune once on a cheap small model, transfer to any larger model for free.

Used by: Microsoft Phi models, increasingly adopted across labs.

---

### 24. Caution — Scaling Laws Don't Predict Downstream Performance

**Reading negative log perplexity charts:** y axis is negative log perplexity — sign flipped. Lower perplexity (better model) = higher on y axis. Moving up = model improving.
```
more params → better model → perplexity drops → negative log perplexity rises → moves up
```

**Loss scales predictably** — clean power law relationship between params and perplexity. Models cluster neatly by size.

**Downstream task performance (e.g. SuperGlue) is unpredictable** — same model sizes produce wildly scattered accuracy scores. Power law breaks down.

**Why:** loss is smooth and continuous. Downstream tasks require specific capabilities that emerge suddenly at certain scales (not gradually) — non-linear jumps break the power law.

**Consequence:** you can predict loss at scale reliably. You cannot predict downstream task performance from loss alone. Lower loss ≠ always better at specific tasks.

This is why evaluation (Lecture 12) matters — can't just look at training loss and declare success. Must measure what you actually care about.

---

### 25. Joint Data-Model Scaling Laws

**Key observation:** lots of data is wasted on small models. A 700M model trained on 10T tokens won't reach the loss of a 7B model on the same data — model size is the bottleneck.

**Joint formula (Rosenfeld+ 2020):**
```
Error = n^(-α) + m^(-β) + C
```
- `n^(-α)` → contribution from data (more tokens → lower error)
- `m^(-β)` → contribution from model size (bigger model → lower error)
- `C` → irreducible error floor (can't go below even with infinite data + model)

Both reduce error independently. You need to scale both.

**Answer to "more data or bigger model?"**
Neither alone is sufficient. Scale both together. Chinchilla says ~20 tokens per parameter is optimal. The joint formula tells you exactly how much each contributes.

The 3D loss surface shows: too small a model OR too few tokens = you're on the steep slope. Optimal = the valley where both are balanced.

---

### 26. Caution — 'Optimal' Scaling Laws Are Hard to Get

The chart answers: *for a given compute budget, how many parameters should your model have?*

**Kaplan (dashed line):** steep slope — as compute goes up, make the model much bigger. Don't worry as much about data. GPT-3, Gopher, Megatron were all built following this.

**Chinchilla's 3 approaches (solid lines):** shallower slope — the optimal model size grows slower with compute. You should be scaling data much more aggressively alongside model size.

**Where real models land:** GPT-3 (175B), Gopher (280B), Megatron (530B) all sit above the Chinchilla lines — too many parameters for their compute budget. They were undertrained. Chinchilla 70B sits near the optimal — smaller model, more data, same compute, better results.

**Why Kaplan was wrong:** ran experiments at small scale and extrapolated. The optimal parameter-to-data ratio shifts as you scale up — something only visible at the larger scale Chinchilla ran at.

**The caution:** scaling law fits are sensitive. Get the experiments slightly wrong → extrapolate → build the wrong model at billion-dollar cost.

---

### 27. Accounting for LR Schedules When Measuring Scaling Laws

Most models use a cosine LR schedule — LR starts high and decays to near zero by end of training. The schedule is tuned to total training steps.

**The problem:** if you compare runs of different lengths, each has a different LR schedule shape. A model that looks worse might just have a worse LR schedule, not actually be a worse model.

**Concrete example:**
- Run A: 100k steps, cosine decays over 100k steps → fully cooled down at end
- Run B: 200k steps, cosine decays over 200k steps → at step 100k, LR is still halfway through decay (still high)

Compare loss at step 100k: Run B looks worse because its LR hasn't finished decaying yet — high LR late in training = noisier updates = higher loss. Nothing to do with model quality.

From the charts: the spread between lines is ~0.15-0.20 loss units just from different cosine cycle lengths — a huge gap caused purely by LR schedule, not model size or data.

**Why this breaks scaling laws:** if you're fitting a curve across 10 runs of different lengths, some cooled down and some not, your curve is measuring "did this run finish its LR decay?" not actual model quality.

**The fix — cooldown at each checkpoint:**
Whenever you want to measure a checkpoint mid-training, branch off and do a short cosine decay to zero from that point, then measure loss.

Example:
- Training run goes 200k steps
- Want to measure at steps 50k, 100k, 150k, 200k
- At each: branch off, run a short cosine decay (~5k steps), measure final loss
- Now all 4 measurements are on equal footing — fully cooled down every time

This ensures your scaling curve measures actual model quality at each compute level, not LR schedule artifacts.

---

---

## Bonus — RLHF vs RLVR vs DPO vs GRPO

### RLHF
- Humans rate outputs (A vs B)
- Train a reward model on those ratings
- Use RL to make the LLM score higher on that reward model
- Problem: model can game the reward model (reward hacking)

### DPO
- You have (good output, bad output) pairs for the same prompt
- Directly pushes model to prefer good over bad
- Offline — fixed dataset, no exploration
- Still needs human preference pairs
- About shaping style and tone, not developing reasoning

### RLVR
- Reward comes from verifiable correctness — math answer right/wrong, code passes tests or not
- Online — model generates attempts during training, gets feedback, learns in real time
- No reward model needed, no human labels needed
- The model explores and discovers solutions
- What gave DeepSeek-R1 and o1 their reasoning ability

**RLHF vs RLVR:** RLHF = reward from human opinion (subjective, can be hacked). RLVR = reward from objective correctness (verifiable, can't be faked).

**DPO vs RLVR:** DPO teaches from fixed examples of good/bad. RLVR lets model explore and discover. DPO can't teach a model to reason through new problems — only imitate existing good answers.

---

### How RLVR Actually Works (GRPO)

**Dataset:** just prompts + correct answers. No reasoning chains, no model outputs.
```
Prompt: "Solve: x² + 5x + 6 = 0"
Answer: x = -2 or x = -3   ← just ground truth to check against
```

**Training loop:**
```
1. Take a problem
2. Run inference on current model → generate 8 attempts
3. Check each attempt against ground truth → reward 1 (correct) or 0 (wrong)
4. Compute group average (say 0.75)
5. Normalize: each attempt score = reward - group average
6. Gradient update: reinforce above-average, penalize below-average
7. KL penalty: don't drift too far from SFT model
8. Repeat for millions of problems
```

**Why reasoning chains emerge:** nobody put reasoning chains in the prompt or target. The model learned step-by-step reasoning during pretraining (from textbooks, math solutions, proofs). RLVR just selects for it — reasoned attempts are correct more often → score above average → get reinforced → reasoning dominates over time.

---

### Why GRPO and not PPO or DPO?

**PPO needs two models:**
```
Policy model  → generates text
Value model   → estimates future reward (same size as policy) ← expensive
```
Double the memory, double the compute, unstable.

**DPO:** offline, needs fixed (chosen, rejected) pairs. Doesn't fit RLVR where you're generating outputs on the fly and checking correctness in real time.

**GRPO:** replaces the value model with the group average as a baseline:
```
PPO:  baseline = value model (separate trained model)
GRPO: baseline = group average of the 8 attempts → free, no extra model
```

No value model → half the memory, more stable training, works naturally with verifiable rewards.

**Note — PPO has two separate things:**
- Reward model → checks if output is good
- Value model → estimates future reward, used as baseline to reduce variance

GRPO replaces the value model (with group average). In RLVR, the reward model is also replaced by verifiable correctness. So both are gone.

---

---

### 28. WSD Learning Rate Schedule (Warmup-Stable-Decay)

**The problem with cosine LR for scaling experiments:**
To do Chinchilla analysis you need to train models on different amounts of data. With cosine, each data target needs a completely different LR schedule — you can't reuse checkpoints from one run to measure different data sizes. That means N² training runs.

**WSD fixes this:**
```
Phase 1: Warmup   → ramp up LR (same as cosine)
Phase 2: Stable   → flat LR (the long middle part)
Phase 3: Decay    → rapid cooldown to near zero
```

The stable phase can be reused. To measure different data sizes:
```
Run one long training run (warmup → stable → end)
Want to know: "how would my model look with less data?"
→ rewind to an earlier checkpoint in the stable phase
→ apply a short decay from there
→ done — exact WSD shape without retraining from scratch
```

This turns N² runs into roughly 1 run. Popularized by MiniCPM, now widely adopted.

**Training curve looks weird with WSD:** loss stays high during stable phase, then drops sharply during decay. Normal — the cooldown phase is where most of the loss improvement happens. Cosine and WSD give roughly similar final loss, but WSD is far cheaper for scaling experiments.

---

### 29. Chinchilla 20:1 Is Just a Starting Point

The 20 tokens per parameter ratio from Chinchilla is not a hard rule — it keeps shifting as architectures and data quality improve:

```
Chinchilla (2022):  ~20 tokens per parameter
Llama 3 (2024):     ~39 tokens per parameter
MiniCPM (2024):     ~192 tokens per parameter (outlier, but direction is clear)
```

Why it shifts:
- Better data quality → model learns more per token → can push ratio higher
- Better architectures → more efficient learners → same compute, more tokens worth it
- Models served at inference want to be small → train smaller model longer = cheaper to serve

**Key takeaway:** don't treat 20:1 as a constraint. With careful optimization, you can go far beyond it. Recent frontier models consistently train at much higher token-to-parameter ratios.

---

## Questions / Follow-up
