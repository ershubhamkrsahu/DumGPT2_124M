from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as f
import tiktoken
import sys
import time
import math
import inspect

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# ---------------------------------------------------------------------------------
# INITIALIZATIONS
# ---------------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.DUMGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # # attention (materialize the large (T, T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = f.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = f.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.DUMGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (similar tensors tend to have similar probabilities)
        self.transformer.wte.weight = self.lm_head.weight

        # init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'DUMGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T) and represents the token indices in the vocabulary
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)
        pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # apply the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads a pretrained GPT model from the Hugging Face model hub."""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "Invalid model type. Choose from 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'."

        from transformers import GPT2LMHeadModel, GPT2Config
        print(f"Loading pretrained model: {model_type}")

        # n_layer, n_head, and n_embd are determined by the model type
        config_args = {
            'gpt2': dict({'n_layer': 12, 'n_head': 12, 'n_embd': 768}),
            'gpt2-medium': dict({'n_layer': 24, 'n_head': 16, 'n_embd': 1024}),
            'gpt2-large': dict({'n_layer': 36, 'n_head': 20, 'n_embd': 1280}),
            'gpt2-xl': dict({'n_layer': 48, 'n_head': 25, 'n_embd': 1600}),
        }[model_type]

        config_args['vocab_size'] = 50257  # GPT-2 models have a vocab size
        config_args['block_size'] = 1024  # GPT-2 models have a block size

        # create a form-scratch initialized minGPT model

        config = GPTConfig(vocab_size=50304)    # (optimise) ugly_number 50257 -> 50304
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove the attention bias keys

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # remove the attention bias keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # remove the attention bias keys

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoint use a "Conv1D" module, but we only want to use a vanilla linear layer, so we need to transpose the weights when copying them over

        assert len(sd_keys) == len(sd_keys_hf), f"State dict keys length mismatch: {len(sd_keys)} vs {len(sd_keys_hf)}"

        for k in sd_keys_hf:
             if any([k.endswith(w) for w in transposed]):
                # special treatment for Conv1D weights: we need to transpose 
                assert sd[k].shape[::-1] == sd_hf[k].shape, f"Shape mismatch for {k}: {sd[k].shape} vs {sd_hf[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())    
             else:
                #  vanilla copy over the other parameters
                assert sd[k].shape == sd_hf[k].shape, f"Shape mismatch for {k}: {sd[k].shape} vs {sd_hf[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. any parameters that is 2D will be weight delayed
        # i.e. all parameters of dim 2 or more will be weight decayed, all params of dim 1 will not (bias and layernorms as well)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

    
# ---------------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, device, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as file:
            text = file.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = (buf[:-1].view(B, T)) # inputs
        y = (buf[1:].view(B, T)) # targets

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bound, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y

# ---------------------------------------------------------------------------------
# DISTRIBUTED DATA PARALLEL
# ---------------------------------------------------------------------------------
# !only helps if you have multiple GPUs in your system 
# torchrun commands sets the env vars RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?)
if ddp:
    # uses the torch.distributed package to handle communication between processes
    assert torch.cuda.is_available(), "For DDP, CUDA is required."
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, no-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to auto detect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}") 

# calculate gradient accumulation steps
total_batch_size = 32768
B = 4 # micro batch size
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------------
# GET A DATA BATCH
# ---------------------------------------------------------------------------------
train_loader = DataLoaderLite(B=B, T=T, device=device, process_rank=ddp_rank, num_processes=ddp_world_size)

# enable mixed precision training
torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------------
# CREATE THE MODEL
# ---------------------------------------------------------------------------------
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# add compile for faster training on GPU (if available)
model = torch.compile(model, mode="default")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP container if needed

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

# learning rate cosine decay scheduler
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps 
    # 2) if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimizer 
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()

        # use automatic mixed precision training  
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps # this is the "normalizer" for equivalent results
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clipping gradients prevents shocks
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    dt = t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt

    if master_process:
        print(
            f"step: {step:5d} | "
            f"loss: {loss_accum.item():.4f} | "
            f"lr: {lr:.2e} | "
            f"norm: {norm:.4f} | "
            f"dt: {dt * 1000:.2f} | "
            f"tok/sec: {tokens_per_sec:.2f}"
        )
if ddp:
    dist.destroy_process_group()

sys.exit(0)

# prefix tokens
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Once upon a time, in a land far away, there lived a")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (1, 8)
x = tokens.to(device)

# generate! right now x is (B, T)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, _ = model(x) # (B, T, vocab_size)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, vocab_size)
        # apply softmax to get probabilities
        probs = f.softmax(logits, dim=-1) # (B, vocab_size)
        # sample from the distribution or take the most likely
        topk_prob, topk_idx = torch.topk(probs, k=50, dim=-1) # (B, 50)
        # select a token form top-k probabilities
        ix = torch.multinomial(topk_prob, num_samples=1) # (B, 1)
        # gather the corresponding token indices
        xcol = topk_idx.gather(-1, ix) # (B, 1)
        # append to the sequence 
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

# print the generated sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"Generated sequence {i+1}: {decoded}")