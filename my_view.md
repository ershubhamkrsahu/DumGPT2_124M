# My View — Building DumGPT2 124M

A concept-first log of this project. Each entry covers a meaningful commit: **what** changed, **why** it was needed, and the **idea** behind it. Trivial wiring/typo commits are intentionally folded into the relevant entries instead of getting their own section.

> **How to maintain this file:** after each meaningful commit, add one entry below with the commit hash, title, and the three sections. Skip commits that only fix a typo or rewire an import — mention them inside the nearest conceptual entry.

---

## Phase 1 — The model

### `f73d110` — Project scaffold
**What:** Created a `uv`-managed Python 3.12 project with PyTorch, `tiktoken`, and NumPy, plus a `.gitignore` for virtualenvs, caches, checkpoints, and datasets.

**Why:** A reproducible environment first, so every later experiment runs against the same dependency graph. The lockfile pins transitive dependencies, not just the ones written in `pyproject.toml`.

**Concept:** Model weights and raw datasets are large binary artifacts; they belong in storage, not Git. A lockfile is what makes "works on my machine" reproducible.

---

### `3c57cd0` — GPT-2 architecture from scratch
**What:** Implemented `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, and `GPT` in `train_gpt2.py` instead of importing `transformers`.

**Why:** The goal is to understand GPT-2, not just call it. Writing every tensor shape by hand forces the architecture to make sense.

**Concept:**
- GPT-2 124M = 12 layers, 12 heads, 768 embedding dims, vocabulary 50,257, context 1,024.
- Attention projects to Q, K, V in one `Linear(768, 3*768)`, then splits the result. Each head has size `768 / 12 = 64`, so tensors become `(B, n_head, T, head_size)`.
- The causal mask is lower-triangular: position `t` may only attend to positions `<= t`. That is what makes the model autoregressive.
- The MLP expands 4× (`768 -> 3072 -> 768`) with GELU in between.
- Each block is pre-LayerNorm with residuals: `x = x + attn(ln_1(x))`, then `x = x + mlp(ln_2(x))`. Residual connections give gradients a direct path to earlier layers.

---

### `5df7188` — Loading pretrained GPT-2 weights
**What:** Added `GPT.from_pretrained()` for `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`, copying weights from Hugging Face into the custom model.

**Why:** A known-good checkpoint is the strongest correctness test for a from-scratch implementation. If the architecture or weight mapping is wrong, the outputs will be nonsense.

**Concept:**
- OpenAI's original checkpoint stores attention/MLP projections as `Conv1D`, whose weight layout is `(in, out)`. PyTorch `Linear` uses `(out, in)`, so those four weight matrices must be transposed when copying.
- Non-parameter buffers like `attn.bias` / `attn.masked_bias` are excluded, and every remaining key and shape is asserted to match.
- Loading a checkpoint is not "training"; it is a validation tool. The project later switches back to random initialization because the actual goal is training from scratch.

---

### `e5c3b71` — Forward pass
**What:** Token embedding + position embedding, pass through all blocks, final LayerNorm, then `lm_head` to vocabulary logits. Also set up eval mode, device transfer, and generation constants.

*(The accompanying rename of `float` to `forward` is a wiring fix, not a concept — `nn.Module` only calls a method named `forward`.)*

**Why:** Complete the path from token IDs to next-token scores so the model can actually be run and tested.

**Concept:**
- `wte(idx)` maps `(B, T)` token IDs to `(B, T, 768)`.
- `wpe(pos)` adds order information. Attention by itself is permutation-invariant, so without positional embeddings the model would see a bag of tokens.
- Logits have shape `(B, T, vocab_size)`: one score per vocabulary item at every position.
- During inference the model only needs the final position's logits to predict the next token.

---

### `05ef05f` — Autoregressive generation
**What:** Tokenize a prompt, repeatedly take the last-position logits, apply softmax, keep the top 50 probabilities, sample one token, and append it to the sequence.

**Why:** Generation is the most intuitive way to see whether the model has learned anything, and sampling (rather than always taking the argmax) produces varied text.

**Concept:**
- Autoregressive decoding feeds the model's own output back as input, one token at a time.
- `torch.topk` truncates the long tail of unlikely tokens. `torch.multinomial` then samples from the remaining distribution, so the result is stochastic but not nonsensical.
- `torch.no_grad()` during generation avoids building an unnecessary autograd graph.

---

## Phase 2 — Training

### `0748c03` + `dabd799` — Train from scratch, on any device
**What:** Commented out pretrained loading and initialized `GPT(GPTConfig())` from scratch. Added device autodetection with priority CUDA → MPS → CPU.

**Why:** The project's actual objective is to train a 124M model from random weights. Pretrained GPT-2 remains useful only as a reference. Device detection keeps the same script usable on a GPU, an Apple laptop, or CPU.

**Concept:** Model parameters and input batches must live on the same device, otherwise PyTorch raises a device mismatch. Random initialization means the initial loss should be near `ln(vocab_size) ≈ ln(50,257) ≈ 10.8` — a useful sanity check.

---

### `d509d48` + `1676116` — Dataset and next-token batches
**What:** Added `dataset_dowload.py` to fetch Tiny Shakespeare, tokenized it with the GPT-2 BPE encoding, and built `x` / `y` batches of shape `(B=4, T=32)`, later `T=1024`.

**Why:** Language modeling needs a continuous stream of text and a self-supervised target: the next token. No human labels are required.

**Concept:**
- `y` is `x` shifted by one position. For every input token, the target is simply the token that came after it.
- A batch of `B × T` tokens contains `B × T` independent next-token predictions, all processed in parallel.
- BPE (`tiktoken`, GPT-2 encoding) splits text into subword units. The model never sees characters or words directly, only integer IDs.
- `input.txt` is ~1.1 MB, roughly 338k tokens — only a few hundred batches per epoch at this batch size.

---

### `02b8331` — Cross-entropy loss
**What:** Added an optional `targets` argument to `GPT.forward`, flattened logits and targets, and computed `F.cross_entropy`.

**Why:** This is the training objective. Without a scalar loss there is nothing to backpropagate.

**Concept:**
- Cross-entropy measures `-log(probability assigned to the correct token)`. Lower is better; it is the same as the negative log-likelihood of the data under the model.
- `cross_entropy` expects `(N, num_classes)` scores and `(N,)` targets, hence the `view(-1, vocab_size)` and `view(-1)`.
- The loss is the average over all `B × T` positions in the batch.

---

### `531b76f` + `adfd144` — Training loop and `DataLoaderLite`
**What:** Added an AdamW optimizer at `lr=3e-4` and a 50-step loop doing zero-grad → forward → backward → step. Replaced the manual slice with `DataLoaderLite`, which streams sequential batches and wraps to the start at the end of the token stream.

**Why:** Start with the simplest correct training loop, and keep data handling deterministic so experiments are easy to reason about.

**Concept:**
- AdamW keeps per-parameter first and second moment estimates, giving each parameter an adaptive effective learning rate. Its "W" means weight decay is decoupled from the gradient update.
- Correct order matters: clear old gradients, recompute them, then update parameters.
- Sequential batches preserve long stretches of context and make runs reproducible. Random sampling would be better for generalization later, but adds complexity now.
- The buffer slice is `B*T + 1` long because targets need one extra token beyond the inputs.

---

### `8b1963e` — Weight tying
**What:** Reused `transformer.wte.weight` as `lm_head.weight`.

**Why:** GPT-2 ties input and output embeddings. It reduces parameter count and couples a token's representation with the scores it produces as a prediction.

**Concept:**
- `wte` maps token ID → vector; `lm_head` maps vector → logits over the vocabulary. Tying them means the same matrix is used in both directions.
- A token whose input embedding is similar to the current hidden state naturally receives a higher output logit, so semantically related tokens influence each other.
- This is part of how GPT-2 124M reaches ~124M parameters. Without tying, `lm_head` alone would add ~38.6M duplicate parameters.

---

### `73a8cbf` — Weight initialization and residual scaling
**What:** Initialize Linear/Embedding weights from `N(0, 0.02)`, zero all biases, and scale the `c_proj` layers in attention and MLP by `(2 * n_layer)^-0.5`.

**Why:** Deep residual networks are sensitive to initialization. This is GPT-2's recipe for keeping activations and gradients stable at depth.

**Concept:**
- Every residual branch adds its output to the residual stream. If each branch contributes unit-scale noise, the stream's variance grows with depth.
- Scaling the projection that writes back into the residual stream by `1/sqrt(2 * n_layer)` keeps the accumulated variance roughly constant. The factor `2` accounts for the two branches (attention and MLP) per block.
- Zeroing biases is safe because LayerNorm follows and because there is no symmetry-breaking role for biases here.

---

## Phase 3 — Making it fast

### `a06d052` — Full context, TF32, and throughput measurement
**What:** Raised `T` from 32 to 1,024 (GPT-2's full context), enabled TF32 matmuls with `torch.set_float32_matmul_precision('high')`, and logged step time plus tokens/second.

**Why:** Training at `T=32` does not exercise the real cost profile. Attention is quadratic in sequence length, so realistic context is where optimization matters. Measuring first gives a baseline.

**Concept:**
- Attention cost is `O(T²)` in time and memory per head; at `T=1024` that is a 1024×1024 score matrix per head.
- TF32 is a tensor-core format with an 8-bit exponent (same range as FP32) and a 10-bit mantissa. On NVIDIA Ampere and newer it speeds up FP32 matmuls with negligible quality impact for training.
- Tokens/second (`B * T / dt`) is a better hardware-utilization metric than steps/second, because it normalizes for batch size and context length.

---

### `a2c40fb` — Honest timing and cheaper gradient clearing
**What:** Added `torch.cuda.synchronize()` around the timed region and switched `optimizer.zero_grad()` to `set_to_none=True`.

**Why:** The first timing change makes measurements real; the second removes unnecessary memory traffic.

**Concept:**
- CUDA kernel launches are asynchronous. Without `synchronize()`, the host measures only how long it took to enqueue work, not how long the GPU actually ran.
- By default `zero_grad()` writes zeros into every gradient tensor. `set_to_none=True` drops the tensors instead, so the allocator can reuse that memory, and the backward pass can allocate fresh gradients only where needed.

---

### `495f740` — `torch.compile` and bfloat16 autocast
**What:** Wrapped the model in `torch.compile()` and ran the forward pass inside `torch.autocast(device_type, dtype=torch.bfloat16)`. Also added `utils/check_gpu.py` to inspect hardware.

**Why:** The default eager-mode PyTorch execution launches many small kernels and pays Python overhead per operation. Compilation and mixed precision are the two biggest wins available before changing the algorithm.

**Concept:**
- `torch.compile` traces the model into a graph, fuses elementwise operations, and generates optimized kernels. The first step is slow (compilation); later steps are faster.
- Autocast runs eligible ops (matmuls, convolutions) in bfloat16 while keeping sensitive ops and master weights in FP32.
- bfloat16 has the same exponent range as FP32, so it does not need loss scaling the way FP16 does. It trades mantissa precision for speed and memory.
- Compilation on Windows requires Triton, which is why the dependency was added in the next commit.

---

### `2615b27` — Flash Attention
**What:** Replaced the hand-written attention with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. Simplified `torch.compile` to `mode="default"` and added `triton-windows`.

**Why:** The manual version materializes a full `(B, n_head, T, T)` attention matrix, which at `T=1024` is both a memory bottleneck and a lot of slow reads/writes to GPU memory. Flash Attention computes the same result without ever writing that matrix out.

**Concept:**
- Standard attention needs `O(T²)` memory per head for the score matrix, and the softmax forces a round trip to high-bandwidth memory.
- Flash Attention tiles Q, K, and V into blocks that fit in SRAM, computes attention block by block, and uses an online softmax to combine partial results. Memory drops to `O(T)`, and the backward pass recomputes tiles instead of storing them.
- The math is exact, not an approximation — only the order of operations and memory movement change.
- `is_causal=True` lets the kernel skip masked-out blocks entirely. PyTorch's SDPA picks the best available backend (Flash, memory-efficient, or math) for the hardware.
- Triton is the kernel compiler behind `torch.compile` on Windows; without it, compiled training is unavailable on this platform.

---

## Current state

- The script trains a randomly initialized GPT-2 124M on Tiny Shakespeare.
- Training runs for 50 steps, with compile, bf16 autocast, Flash Attention, and throughput logging, then exits before the generation block.
- The generation code remains as a reference/debug path and can be reached by removing `sys.exit(0)`.

## Things to explore next

- Add evaluation: measure validation loss on a held-out split instead of only training loss.
- Add checkpoint saving/resuming, including optimizer state.
- Scale to gradient accumulation and distributed data parallel for larger effective batches.
- Implement a proper learning-rate schedule (warmup + cosine decay) and gradient clipping.
- Feed real data into generation after training so samples can be inspected.
