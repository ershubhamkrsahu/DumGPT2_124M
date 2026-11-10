# my_view

This is my personal log for DumGPT2 124M. Every time I make a commit that actually teaches me something, I write down what I did, why I did it, and what clicked for me. If a commit was just a typo or a broken import, I don't give it its own entry — I just fold it into whatever entry it belongs to.

The point isn't to document every line. The point is that future me can open this and remember how the whole thing works.

---

## Phase 1 — building the model

### f73d110 — setting up the project

I started with a uv-managed Python 3.12 project and added torch, tiktoken, and numpy. I also wrote a proper `.gitignore` before writing any real code.

Why: I wanted the environment to be reproducible from day one. `pyproject.toml` only lists the packages I directly use, but `uv.lock` pins the whole dependency tree, so the exact same versions come back later. The `.gitignore` keeps virtualenvs, `__pycache__`, checkpoints, and datasets out of git.

What I picked up: model weights and raw data are big binary files, they don't belong in a repo. Lockfiles are what actually make "it works on my machine" reproducible.

### 3c57cd0 — writing GPT-2 from scratch

I implemented `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, and `GPT` myself instead of just importing `transformers`.

Why: the whole point of this project is to understand GPT-2. If I just call `GPT2LMHeadModel`, I learn nothing about what's inside it.

What I picked up:
- GPT-2 124M is 12 layers, 12 heads, 768 embedding dims, 50,257 vocab, and 1,024 context.
- Attention makes Q, K, V with one `Linear(768, 3*768)` and then splits the output three ways. Each head is 64 dims, so the tensors become `(B, n_head, T, head_size)`.
- The causal mask is lower triangular. Position `t` can only look at positions `<= t`, and that's literally what makes the model autoregressive.
- The MLP goes 768 → 3072 → 768 with GELU in the middle.
- Every block is pre-LayerNorm with a residual: `x = x + attn(ln_1(x))`, then `x = x + mlp(ln_2(x))`. The residual is a shortcut for gradients, so deep stacks still train.

### 5df7188 — loading the real GPT-2 weights

I wrote `GPT.from_pretrained()` so I could copy weights from Hugging Face into my model for `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`.

Why: loading a checkpoint I trust is the best way to check whether my architecture is actually correct. If a shape or a name was wrong, the outputs would be garbage and I'd know immediately.

What I picked up:
- OpenAI's original checkpoint uses `Conv1D`, which stores weights as `(in, out)`. PyTorch `Linear` stores them as `(out, in)`, so `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, and `mlp.c_proj` all need a transpose when copying.
- The `attn.bias` and `attn.masked_bias` buffers don't exist in my model, so I filter those keys out.
- I assert every remaining key and shape before copying. If the counts don't match, I want it to blow up instead of silently loading a half-broken model.
- This was only ever a correctness check. I go back to random init right after, because the actual goal is training from scratch.

### e5c3b71 — the forward pass

I wired up the full path: token embedding + position embedding, through all the blocks, final LayerNorm, then `lm_head` to get logits. I also set eval mode, moved the model to the device, and set up the generation constants.

There was also a dumb bug where I named the method `float` instead of `forward`. That one doesn't deserve its own entry — `nn.Module` only calls a method called `forward`, so renaming it was the fix.

What I picked up:
- `wte(idx)` turns `(B, T)` token IDs into `(B, T, 768)`.
- `wpe(pos)` adds position information. Attention is permutation-invariant on its own, so without this the model would see a bag of tokens and have no idea about order.
- Logits come out as `(B, T, vocab_size)`. For next-token prediction I only care about the last position.
- In inference I only need the final logits, not the whole sequence.

### 05ef05f — generating text

I tokenized a prompt, then looped: take the last position's logits, softmax them, keep the top 50, sample one token, and append it. Repeat until I hit the max length.

Why: generation is the fastest way to see if the model is learning anything at all, and sampling makes the output varied instead of the same sentence every time.

What I picked up:
- Autoregressive decoding just means feeding the model's own output back into it, one token at a time.
- Top-k cuts off the long tail of nonsense tokens. `torch.multinomial` then samples from what's left, so results are random but still plausible.
- `torch.no_grad()` during generation stops autograd from building a graph I'll never use.

---

## Phase 2 — actually training it

### 0748c03 + dabd799 — training from scratch on any device

I commented out the pretrained load and switched to `GPT(GPTConfig())`, so the model starts from random weights. I also added device detection: CUDA first, then MPS, then CPU.

Why: the real goal of the project is to train my own model. The pretrained GPT-2 is just a reference point. Device detection means the same script runs on my GPU, on a Mac, or on CPU without edits.

What I picked up: the model and the data have to be on the same device or PyTorch throws a mismatch error. And when initialization is correct, the very first loss should be around `ln(50,257) ≈ 10.8`. If it isn't, something is wrong before training even starts.

### d509d48 + 1676116 — Tiny Shakespeare and shifted batches

I added `dataset_dowload.py` to grab Tiny Shakespeare, tokenized it with the GPT-2 BPE encoding, and started making `x`/`y` batches. They started at `B=4, T=32` and later went to `T=1024`.

Why: next-token prediction is self-supervised. I don't need labels, I just need a stream of text and the token that comes next.

What I picked up:
- `y` is just `x` shifted by one. Every input token's target is whatever followed it in the text.
- One batch of `B × T` tokens gives me `B × T` predictions in parallel.
- BPE splits text into subwords, so the model never sees characters or whole words, only token IDs.
- Tiny Shakespeare is only ~1.1 MB, roughly 338k tokens, so an epoch is just a few hundred batches at this batch size.

### 02b8331 — cross-entropy loss

I added an optional `targets` argument to `forward` and computed `F.cross_entropy` over the flattened logits and targets.

Why: without a scalar loss there's nothing to backpropagate. This is the entire training objective.

What I picked up:
- Cross-entropy is `-log(probability of the correct token)`. Lower is better, and it's the same thing as negative log-likelihood.
- `cross_entropy` wants `(N, num_classes)` scores and `(N,)` targets, which is why I flatten with `view(-1, vocab_size)` and `view(-1)`.
- The reported loss is the average over all `B × T` positions in the batch.

### 531b76f + adfd144 — training loop and DataLoaderLite

I set up AdamW at `lr=3e-4` and a 50-step loop: zero grad, forward, backward, step. Then I replaced the manual slicing with `DataLoaderLite`, which walks through the token stream in order and wraps back to the start at the end.

Why: I wanted the simplest correct loop first. Sequential batches also keep the data deterministic, which makes it way easier to compare runs.

What I picked up:
- AdamW keeps a running first and second moment for every parameter, so each one gets its own adaptive step size. The "W" means weight decay is decoupled from the gradient update.
- Order matters: clear old grads, compute new ones, then update.
- I slice `B*T + 1` tokens because the targets need one token beyond the inputs.
- Sequential data preserves long-range context and is reproducible. Random sampling would generalize better, but that's a later problem.

### 8b1963e — tying the embeddings

I pointed `transformer.wte.weight` and `lm_head.weight` at the same tensor.

Why: GPT-2 does it, it cuts parameters, and it ties a token's input representation to the scores it produces as an output.

What I picked up:
- `wte` goes token ID → vector. `lm_head` goes vector → vocabulary scores. Tying them means one matrix does both jobs.
- If a token's embedding is close to the current hidden state, that token naturally gets a higher score. Related tokens end up influencing each other.
- This is part of how GPT-2 counts as 124M params. Without tying, `lm_head` alone would add ~38.6M duplicates.

### 73a8cbf — initialization and residual scaling

I initialized Linear and Embedding weights from `N(0, 0.02)`, zeroed all biases, and scaled the `c_proj` layers by `(2 * n_layer)^-0.5`.

Why: deep residual nets are really sensitive to init. This is GPT-2's recipe for keeping activations and gradients under control at depth.

What I picked up:
- Every residual branch dumps its output into the residual stream. If each one contributes unit-scale noise, the variance grows with depth.
- Scaling the projection that writes back into the stream by `1/sqrt(2 * n_layer)` keeps that growth roughly constant. The 2 is because each block has two branches, attention and MLP.
- Zero biases are fine here because LayerNorm comes right after and there's no symmetry to break.

---

## Phase 3 — making it fast

### a06d052 — full context, TF32, and tok/s

I bumped `T` from 32 to 1,024, turned on TF32 with `torch.set_float32_matmul_precision('high')`, and started logging step time and tokens per second.

Why: training at `T=32` doesn't look anything like real GPT-2 training. Attention is quadratic in sequence length, so the full context is where things actually get expensive. Measure first, optimize second.

What I picked up:
- Attention is `O(T²)` in both time and memory per head. At `T=1024` that's a 1024×1024 score matrix per head.
- TF32 has the same exponent range as FP32 but only a 10-bit mantissa. On Ampere and newer it speeds up FP32 matmuls with basically no quality hit for training.
- Tokens per second (`B*T/dt`) is a better metric than steps per second because it accounts for batch size and context length.

### a2c40fb — timing that actually means something

I put `torch.cuda.synchronize()` around the timed section and switched `optimizer.zero_grad()` to `set_to_none=True`.

Why: my timing was lying to me, and I was doing unnecessary memory work every step.

What I picked up:
- CUDA launches are asynchronous. Without `synchronize()`, I'm timing how long it takes to queue the work, not how long the GPU runs.
- `zero_grad()` normally writes zeros into every gradient. `set_to_none=True` throws the tensors away instead, so the allocator can reuse that memory and backward only allocates what it needs.

### 495f740 — torch.compile and bf16 autocast

I wrapped the model in `torch.compile()` and ran the forward pass inside `torch.autocast(..., dtype=torch.bfloat16)`. I also added `utils/check_gpu.py` to see what my GPU actually supports.

Why: eager PyTorch launches a ton of tiny kernels and pays Python overhead for each op. Compiling and using mixed precision are the two biggest wins I can get before touching the algorithm.

What I picked up:
- `torch.compile` traces the model into a graph, fuses elementwise ops, and generates faster kernels. The first step is slow because it's compiling.
- Autocast runs matmuls and similar ops in bfloat16, but keeps sensitive ops and master weights in FP32.
- bfloat16 has the same exponent range as FP32, so it doesn't need the loss scaling that FP16 does. It just gives up mantissa bits for speed and memory.
- Compiling on Windows needs Triton, which is why I added it in the next commit.

### 2615b27 — Flash Attention

I replaced my manual attention with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, simplified `torch.compile` to `mode="default"`, and added `triton-windows`.

Why: my version materializes the whole `(B, n_head, T, T)` attention matrix. At `T=1024` that's a huge memory cost and a lot of slow reads and writes to GPU memory. Flash Attention gets the same answer without ever writing that matrix out.

What I picked up:
- Normal attention needs `O(T²)` memory per head for the scores, and softmax forces them out to high-bandwidth memory and back.
- Flash Attention tiles Q, K, V into blocks that fit in SRAM, computes attention block by block, and uses an online softmax to combine the pieces. Memory drops to `O(T)`, and backward recomputes tiles instead of storing them.
- The math is exact. It's the same attention, just a different order of operations and way less memory traffic.
- `is_causal=True` lets the kernel skip the masked-out blocks entirely. PyTorch picks the best backend available for the hardware.
- Triton is the compiler behind `torch.compile` on Windows. Without it, compiled training just isn't available there.

### 3b4ee32 — padding the vocabulary to 50,304

I changed the vocabulary size from 50,257 to 50,304.

Why: 50,257 is an ugly number that doesn't line up with the sizes the GPU likes. 50,304 is the next multiple of 128, so the embedding and `lm_head` matmuls split evenly into tensor-core/warp tiles instead of leaving a ragged edge. This gave me about a 4% speed boost. The extra 47 rows are dead logits — the loss never touches them and sampling can't pick them.

What I picked up:
- GPU matmuls are happiest when dimensions are multiples of 128 (or at least 64 or 32). Anything else leaves partial tiles and wasted work at the boundary.
- When training from scratch, vocab size is basically a free architectural choice, so rounding it up is a cheap win.
- The padded rows do add a tiny bit of parameter and compute cost, but 47 out of 50,257 is ~0.1%, way less than the waste it removes.
- I first made this change inside `from_pretrained`, which didn't touch the training path. I later moved it into the training config with `GPTConfig(vocab_size=50304)`, so the speedup is active now.

---

## Phase 4 — tuning the training run

The model was reasonably fast by this point. What I hadn't touched yet was whether it trains *well* — learning rate, gradient stability, and optimizer settings. Those aren't speed wins, they're tuning.

### c2c53d3 — learning rate schedule, gradient clipping, and GPT-3 optimizer settings

I added a real learning rate schedule, gradient clipping, and the GPT-3 optimizer hyperparameters. I also made the log line show LR and grad norm instead of just loss and speed.

Why: a constant `3e-4` from step zero is a bad idea. At the start the weights are random, so early gradients are noisy and a full-size step can destabilize things. Near the end, the same big step keeps bouncing around the minimum instead of settling into it. Warmup fixes the first problem, cosine decay fixes the second.

What I picked up:
- Warmup ramps the LR linearly from near zero to `max_lr` over 10 steps. It gives Adam time to build decent moment estimates before taking big steps.
- Cosine decay goes from `max_lr` to `min_lr` (10% of max) along a cosine curve. It spends more time at small LRs near the end, which is good for converging.
- The scheduler is applied after `backward()` and before `optimizer.step()`. PyTorch gets the LR from each param group, so I loop over `optimizer.param_groups` and set `lr` there every step.
- `clip_grad_norm_` rescales all gradients if their global norm goes above 1.0. This stops one bad batch — especially with bf16 — from blowing up the weights. It returns the norm *before* clipping, which is why logging it is useful: if it keeps sitting at 1.0, clipping is doing real work.
- `betas=(0.9, 0.95)` is the GPT-3 setting. A lower `beta2` than the usual 0.999 makes the second moment adapt faster to recent gradient variance, which suits large models and larger batches.

### 5d5be6c — splitting params for weight decay and using fused AdamW

I moved optimizer creation into `GPT.configure_optimizers()` and now build two param groups: everything with 2D+ shape gets `weight_decay=0.1`, and everything 1D (biases and LayerNorm gains) gets zero decay. I also bumped `max_lr` to `6e-4` and switched to the fused AdamW kernel when CUDA is available.

Why: weight decay is a pull toward zero, which is fine for weight matrices because that's the regularization I actually want. It makes no sense for biases and LayerNorm gains — zeroing a bias just removes a degree of freedom, and LayerNorm gains have a natural scale around 1. I could also push the LR up to `6e-4` because warmup, clipping, and the decay groups made the early steps stable enough.

What I picked up:
- `p.dim() >= 2` is the usual heuristic: matrices and embeddings get decayed, vectors (biases, LayerNorm weights) don't.
- A param group in AdamW is just a dict with `'params'` plus overrides. That's why my LR loop can set `param_group['lr']` every step.
- Fused AdamW does the whole update in one CUDA kernel instead of many small ones, which cuts memory traffic. It only works on CUDA, so I check for `fused` in `inspect.signature(torch.optim.AdamW)` and also check the device before using it.
- Decoupled weight decay is not the same as L2 in the loss. It shrinks weights directly, which behaves better with Adam's adaptive scaling.
- Filtering on `requires_grad` matters if I ever freeze layers, since frozen params shouldn't be in the optimizer at all.

### e0c4ca8 — gradient accumulation for a 32k-token batch

I set `total_batch_size = 32768` and kept the micro-batch at `B=4, T=1024`. That gives `grad_accum_steps = 8`. The training loop now runs 8 forward/backward passes, divides every loss by 8, and only calls `optimizer.step()` once at the end.

Why: a 32k-token batch is 8× bigger than what fits in memory at `B=4`. Accumulation lets me get the optimization behavior of a big batch without holding all of its activations at once.

What I picked up:
- Dividing the loss by `grad_accum_steps` before `backward()` is essential. Without it, gradients sum across micro-batches instead of averaging, which silently multiplies the effective learning rate by 8.
- `zero_grad()` stays outside the inner loop. Gradients are supposed to accumulate; only the parameter update happens once.
- `loss_accum` is detached so logging doesn't keep the graph alive.
- Throughput has to include `grad_accum_steps`, otherwise tok/s looks 8× worse than it really is.
- The math is equivalent to one big batch, but not bit-for-bit identical: it's 8 sequential chunks, so the order and gradient noise can differ slightly.
- Clip after the whole accumulation window, not after each micro-batch, otherwise I'd be clipping partial gradients.
- Memory stays at one micro-batch's activations; the only thing that persists is the accumulated gradient buffer.

I also started adding a `device` argument to `DataLoaderLite` but didn't pass it at the call site yet, so that's still a loose end.

---

## Phase 5 — distributed training (attempted)

### b08b074 — DDP scaffolding, parked for now

I added the DDP plumbing: detect the `torchrun` environment variables, initialize the process group, give each rank its own device and strided slice of the token stream, turn off gradient sync except on the last micro-step, all-reduce the loss, log only from rank 0, and destroy the process group at the end. I also finally moved `vocab_size=50304` into the training config, so the padding optimization is actually active now.

Why: DDP is how the same model gets trained across multiple GPUs. Each rank sees different data, and the gradients get averaged, so the optimization behaves like one big batch instead of N independent runs.

The honest part: **I didn't get the DDP section working.** I kept getting errors and tried to fix them, but I couldn't get past them, so I've parked it for now. The biggest reason is that this machine only has one GPU — debugging multi-GPU communication when you can't actually run multiple processes is painful. On top of that, I'm on Windows, where the usual NCCL backend isn't really available. The normal single-GPU path (`ddp=False`) is untouched, so this doesn't block my training.

What I picked up:
- `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. If `RANK` isn't set, it's a plain single-process run.
- Each rank has to see different data. Starting at `B*T*rank` and advancing by `B*T*world_size` gives every rank a strided chunk and still covers the dataset once per global step.
- DDP averages gradients during backward. With gradient accumulation, syncing on every micro-step is wasted communication, so `require_backward_grad_sync` stays off until the final micro-step.
- The logged loss should be all-reduced with `AVG` across ranks, otherwise rank 0 is only reporting its own slice.
- Only the master process should print or checkpoint; otherwise N processes flood the console.
- `model.module` unwraps the DDP container so the optimizer is built on the real model, not the wrapper.
- Effective batch size is `B*T*grad_accum*world_size`, which is why `grad_accum` has to be divided by the world size.
- NCCL is NVIDIA's collective communication library. It's Linux-first, so Windows doesn't have the normal `nccl` backend, which is part of why I'm leaving this alone for now.

---

## Phase 6 — real data

### f8ea8e2 + 4fb187d — FineWeb-Edu, token shards, and getting data out of git

I replaced Tiny Shakespeare with FineWeb-Edu. I wrote `fineweb_edu.py` to stream the dataset, tokenize it, and write binary shards, then rewrote `DataLoaderLite` to read those shards instead of `input.txt`. I also stopped tracking `input.txt` and ignored the shard directory, so only the scripts live in git now.

Why: Tiny Shakespeare is about 1 MB, roughly 338k tokens. A 124M-parameter model can basically memorize that, and the loss would look great while meaning almost nothing. To train something that actually generalizes I need billions of tokens. FineWeb-Edu is open, high quality, and already used by nanoGPT-style runs, so it was the obvious step up.

**Where this data actually comes from:**
- **Common Crawl** is a nonprofit that crawls the public web every month and publishes raw WARC archives of the pages. It is not a clean dataset — it's just a huge pile of HTML.
- **FineWeb** (from Hugging Face) processes those crawls: extracts text from HTML, identifies the language, applies quality heuristics (Gopher/C4-style rules), deduplicates with MinHash, and scrubs PII.
- **FineWeb-Edu** adds one more stage: a classifier scores each page for educational value and keeps the high-scoring ones. The full dataset is ~1.3T tokens; the `sample-10BT` config I'm using is a 10B-token sample of it.
- So the chain is: Common Crawl → extracted and filtered web text → educational classifier → FineWeb-Edu → GPT-2 tokenizer → uint16 shards → training batches.

**How my pipeline works:**
- I stream the HF dataset instead of downloading all the raw text first.
- Each document is tokenized with GPT-2 BPE, with an EOT token prepended so the model can tell where one document ends and the next begins.
- Tokens are packed into 100M-token shards and stored as `uint16`. The GPT-2 vocabulary is 50,257, which fits comfortably below 65,536, so two bytes per token is enough and cuts storage in half compared to `int32`.
- 100 shards came out to about 18.5 GiB. Shard 0 is validation and the other 99 are training data.
- `DataLoaderLite` keeps only one shard in RAM at a time, slices `B*T + 1` tokens from it, and moves to the next shard when it runs out. It never holds the whole dataset.
- `input.txt` is now untracked and ignored, along with `edu_fineweb10B/`. The scripts are versioned; the data is not.

What I picked up:
- Raw Common Crawl is not training data. Most of the work in a real dataset is extraction, filtering, and dedup, not modeling.
- Quality filtering matters more than raw volume, especially for smaller models. A smaller clean dataset beats a giant noisy one.
- Tokenize once and store integers. Re-tokenizing text every epoch would be a huge waste of CPU.
- `uint16` is a nice fit here because the vocabulary is small enough. It halves the bytes per token versus `int32`.
- Sharding keeps RAM bounded and lets training stream from disk. The tradeoff is a stall whenever I switch shards; later I'd prefetch the next shard in the background.
- EOT tokens matter. Without them, the model would read unrelated web pages as one continuous document.
- Reality check on the schedule: 19,073 steps at a 32,768-token batch is only ~625M tokens, not the full 10B. nanoGPT's 19,073 assumes a ~524k-token effective batch. So right now I'm seeing roughly 6% of the sample unless I raise the effective batch size.

---

## Phase 7 — evaluation

### 6f63db8 — validation loss every 100 steps

I added a `reset()` method to `DataLoaderLite`, created a second loader for the validation split, and now measure validation loss every 100 steps over 20 batches. I also moved the old generation code inside the training loop (disabled for now) and deleted the dead block after `sys.exit(0)`.

Why: training loss by itself can lie. It's measured on the same shards the model is fitting, so it mostly tells me how well the model is memorizing. Validation loss on the held-out shard tells me whether it's actually learning patterns that transfer. If train loss keeps dropping while val loss stalls or rises, I'm overfitting.

What I picked up:
- `model.eval()` and `model.train()` change behavior for things like dropout and batch norm. GPT-2 here has neither, but I still switch modes so the habit is correct and future changes don't silently break.
- Validation runs under `torch.no_grad()` and autocast. No gradients means no graph and much less memory, which lets me use the same batch size without thinking about it.
- `reset()` on the validation loader makes every evaluation start from the same place, so the numbers are comparable across steps instead of drifting with the loader state.
- I average 20 validation batches (about 82k tokens) to reduce noise. With DDP I all-reduce that average across ranks.
- Every 100 steps is a compromise. Validation costs time, so I moved it before the timer starts — now `dt` and tok/s only measure the actual training step instead of being dragged down by evaluation.
- The validation shard is 100M tokens, which is way more than I need, but reusing the shard format kept the loader simple.
- Generation now runs every 100 steps. I had it disabled because `torch.compile` threw an error; after fixing the bad `gather` call I turned it back on, so I'm watching whether the compile issue still shows up. Every 100 steps keeps the sampling cost from dominating training. Each rank seeds its own generator so the ranks don't all sample the same text.

---

## where things stand

- The script trains a randomly initialized GPT-2 124M on shards from the FineWeb-Edu `sample-10BT` dataset.
- The dataset is tokenized into 100 `uint16` shards under `edu_fineweb10B/` (~18.5 GiB): shard 0 is validation, the rest are training. The shards and `input.txt` are ignored by git.
- It runs up to 19,073 steps with compile, bf16 autocast, Flash Attention, gradient accumulation to a 32k-token effective batch, warmup + cosine LR decay, gradient clipping, fused AdamW with decay/no-decay groups, and tok/s logging.
- Every 100 steps it also measures validation loss on the held-out shard, and every 100 steps it generates a few sample sequences.
- DDP scaffolding is in the file but parked — the single-GPU path is what actually runs today.

## stuff I want to try next

- Confirm generation is stable under `torch.compile`.
- Prefetch the next shard in the background so disk reads don't stall the GPU.
- Save and resume checkpoints, including optimizer state.
- Get the DDP path working on a machine with more than one GPU.
- Tune weight decay and the max learning rate further.
- Generate samples after training so I can actually read what the model learned.
