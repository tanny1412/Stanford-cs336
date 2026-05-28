# Lecture 8 — CS336: Parallelism 2

## Summary
Code-level implementation of distributed training primitives — DDP, tensor parallel, pipeline parallel on MLPs. Mostly a concretization of Lecture 7 concepts in code.

---

## Key Concepts

### 1. Key Terminology

- **World size** — total number of devices (e.g. 4 GPUs = world size 4)
- **Rank** — the ID of each device (rank 0, rank 1, rank 2, rank 3)
- **Barrier** — synchronization point that waits for all processes to reach it before continuing

---

### 2. All-Reduce Bandwidth in Practice

Benchmarked all_reduce on 4 H100s:
```
Achieved bandwidth: ~277 GB/s
Theoretical NVLink peak: ~900 GB/s
```
You never hit theoretical peak — wave quantization, tensor sizes, and NCCL overhead all reduce actual bandwidth. Always benchmark your actual workload.

Why all_reduce has a 2× factor in bandwidth calculation:
- Each rank sends its tensor out (reduce step)
- Each rank receives the result back (broadcast step)
- Total bytes = 2 × tensor_size

reduce_scatter has no 2× factor — you only reduce, you don't get it back to everyone.

---

### 3. DDP is Just One Line Added to SGD

The entire DDP implementation is normal SGD with one injection after backward:

```python
# normal forward + backward
loss.backward()

# DDP: just this one line per parameter
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

# normal optimizer step
optimizer.step()
```

All-reduce is a synchronization point — all ranks pause here until everyone is ready. If one rank is missing an all_reduce call, the whole job hangs waiting.

Losses are different across ranks (different data) but after all_reduce, gradients are identical — so all ranks do the exact same optimizer step and stay in sync.

---

### 4. Tensor Parallelism in Code

Each rank gets 1/world_size of the hidden dimension:
```
local_num_dim = num_dim / world_size  # e.g. 1024/4 = 256

Each rank's weight matrix: num_dim × local_num_dim  (not num_dim × num_dim)
```

After each layer's forward pass, activations are partial (batch_size × local_num_dim). To continue to the next layer, must all_gather activations so every rank has the full activations:
```
partial activations → all_gather → full activations → next layer
```

This all_gather happens every layer — explains why tensor parallelism needs fast NVLink. Lots of communication per layer.

---

### 5. Pipeline Parallelism in Code

Each rank gets `num_layers / world_size` layers. Data flows via point-to-point send/recv (not collective operations):

```python
# rank 0: start with data, compute layers, send activations to rank 1
dist.send(activations, dst=rank+1)

# rank 1: receive activations from rank 0, compute layers, send to rank 2
dist.recv(activations, src=rank-1)
dist.send(activations, dst=rank+1)
```

Naive implementation is synchronous — each rank waits for the previous. Async send/recv (`isend`, `irecv`) needed to overlap communication and computation properly.

---

### 6. JAX/TPU Alternative

JAX allows declarative sharding — you just specify which dimension to shard, the compiler figures out the collectives:

```python
# FSDP in ~10 lines with JAX
model = shard(model, mesh, partition_spec)
# JAX handles all-gather, reduce-scatter automatically
```

Much cleaner than PyTorch's explicit collective calls. But PyTorch lets you see under the hood — useful for understanding. In production, use FSDP or Accelerate, not manual collectives.

---

### 7. Key Insight — Recompute vs Store vs Communicate

Three options when you need data during backward:
```
1. Recompute from scratch        → costs compute, saves memory
2. Store in memory (same GPU)   → costs memory, saves compute
3. Store on another GPU         → costs communication (slowest option)
```

Recomputation is often worth it — communication across GPUs is slower than recomputing. This is the same trade-off as activation checkpointing but at the multi-GPU level.

---

### 8. Hardware Will Always Have Hierarchy

Even as hardware improves, models will always scale to the limit of what hardware can do. The memory hierarchy (L1 → HBM → NVLink → InfiniBand) will always exist — just at different scales and speeds.

Understanding how to work with this hierarchy is a permanent skill, not something that becomes obsolete.

---

## Questions / Follow-up
