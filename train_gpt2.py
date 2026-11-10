"""
DumGPT - GPT-2 style language model training.

This script keeps the original training structure:
- Custom GPT model
- FineWeb-Edu binary shards
- AdamW optimizer
- Learning-rate warmup + cosine decay
- Mixed precision
- Gradient accumulation
- Optional multi-GPU DDP
- HellaSwag evaluation
- Checkpoints and training graphs

Single GPU:
    python train_gpt2.py --max-steps 100

Two GPUs:
    torchrun --standalone --nproc_per_node=2 train_gpt2.py

Resume:
    python train_gpt2.py --resume checkpoints/latest.pt
"""

# =============================================================================
# IMPORTS
# =============================================================================

from dataclasses import dataclass
import argparse
import csv
import inspect
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Distributed training (DDP)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# HellaSwag helpers are kept in your existing separate script.
from hellaswag import iterate_examples, render_example, get_most_likely_row, evaluate_hellaswag


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent / "edu_fineweb10B"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Model configuration: GPT-2 124M.
BLOCK_SIZE = 1024
VOCAB_SIZE = 50304
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# Training configuration.
# These are conservative defaults for a 4 GB laptop GPU.
TOTAL_BATCH_SIZE = 8192
B = 1
T = 512

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
MAX_STEPS = 19073
WEIGHT_DECAY = 0.1

# Evaluation and checkpoint frequency.
VAL_INTERVAL = 250
VAL_STEPS = 10
HELLASWAG_INTERVAL = 0       # 0 disables periodic HellaSwag evaluation.
HELLASWAG_MAX_EXAMPLES = 500
SAVE_INTERVAL = 500
SAMPLE_INTERVAL = 500

# Gradient checkpointing saves VRAM at the cost of extra computation.
USE_GRADIENT_CHECKPOINTING = True


# =============================================================================
# TOKENIZER
# =============================================================================

import tiktoken
enc = tiktoken.get_encoding("gpt2")


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = VOCAB_SIZE
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Project input into query, key, and value in one operation.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.DUMGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split embedding dimension into attention heads.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch SDPA automatically uses an efficient attention kernel when available.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
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


# =============================================================================
# GPT MODEL
# =============================================================================

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie token embeddings and output projection weights.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "DUMGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # Gradient checkpointing trades a little speed for lower VRAM usage.
        for block in self.transformer.h:
            if USE_GRADIENT_CHECKPOINTING and self.training and x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Weight decay is applied to matrix weights, not biases or LayerNorm.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )


# =============================================================================
# DATA LOADER
# =============================================================================

def load_tokens(filename):
    """Read one uint16 shard. memmap avoids loading the whole shard into RAM."""
    return np.memmap(filename, dtype=np.uint16, mode="r")


class DataLoaderLite:

    def __init__(self, B, T, data_dir, split, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in ["train", "val"]

        self.shards = sorted(data_dir.glob(f"edu_fineweb_{split}_*.bin"))
        assert len(self.shards) > 0, f"No {split} shards found in {data_dir}"

        if process_rank == 0:
            print(f"Found {len(self.shards)} {split} shard(s).")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1

        # Move to the next shard before slicing if this batch would go out of bounds.
        if self.current_position + needed > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        if self.current_position + needed > len(self.tokens):
            raise RuntimeError(
                f"Shard {self.shards[self.current_shard]} is too small for "
                f"batch={B}, sequence_length={T}."
            )

        buf = np.asarray(
            self.tokens[self.current_position:self.current_position + needed],
            dtype=np.int64,
        )

        x = torch.from_numpy(buf[:-1].copy()).view(B, T)
        y = torch.from_numpy(buf[1:].copy()).view(B, T)

        # Each DDP process reads a different section of the token stream.
        self.current_position += B * T * self.num_processes
        return x, y


# =============================================================================
# DDP INITIALIZATION
# =============================================================================

def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA."
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        master_process = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True

    return ddp, rank, local_rank, world_size, device, master_process


# =============================================================================
# LEARNING-RATE SCHEDULE
# =============================================================================

def get_lr(step, max_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    if step >= max_steps:
        return MIN_LR

    decay_ratio = (step - WARMUP_STEPS) / max(max_steps - WARMUP_STEPS, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# =============================================================================
# CHECKPOINTS AND GRAPHS
# =============================================================================

def save_checkpoint(path, step, model, optimizer, scaler, config):
    """Save everything required to continue training."""
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.__dict__,
    }

    temp_path = path.with_suffix(".tmp")
    torch.save(state, temp_path)
    temp_path.replace(path)


def create_training_graphs(metrics_file, output_dir):
    """Create graphs from the CSV log after training."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install pandas and matplotlib to generate training graphs.")
        return

    if not metrics_file.exists():
        return

    data = pd.read_csv(metrics_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("learning_rate", "Learning Rate", "Learning rate"),
        ("grad_norm", "Gradient Norm", "Gradient norm"),
        ("tokens_per_sec", "Training Throughput", "Tokens / second"),
        ("hellaswag_accuracy", "HellaSwag Accuracy", "Accuracy"),
    ]

    for column, title, ylabel in graphs:
        if column not in data or data[column].dropna().empty:
            continue

        values = data[column].dropna()
        steps = data.loc[values.index, "step"]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values)
        plt.title(title)
        plt.xlabel("Training step")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}.png", dpi=160)
        plt.close()

    print(f"Training graphs saved to: {output_dir}")


# =============================================================================
# TRAINING
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=B)
    parser.add_argument("--sequence-length", type=int, default=T)
    parser.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--hellaswag-interval", type=int, default=HELLASWAG_INTERVAL)
    parser.add_argument("--hellaswag-max-examples", type=int, default=HELLASWAG_MAX_EXAMPLES)
    args = parser.parse_args()

    ddp, rank, local_rank, world_size, device, master_process = setup_distributed()

    try:
        random.seed(1337 + rank)
        np.random.seed(1337 + rank)
        torch.manual_seed(1337 + rank)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(1337 + rank)
            torch.set_float32_matmul_precision("high")

        # Select precision based on the actual GPU.
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_amp = device.type == "cuda"

        # FP16 needs gradient scaling; BF16 does not.
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(use_amp and amp_dtype == torch.float16),
        )

        total_batch_size = args.total_batch_size
        B = args.batch_size
        T = args.sequence_length

        assert total_batch_size % (B * T * world_size) == 0, (
            "total batch size must be divisible by B * T * number of GPUs"
        )
        grad_accum_steps = total_batch_size // (B * T * world_size)

        if master_process:
            print("=" * 72)
            print("DumGPT training")
            print("=" * 72)
            print(f"Device              : {device}")
            print(f"Number of GPUs      : {world_size}")
            print(f"Precision           : {'BF16' if use_bf16 else 'FP16' if use_amp else 'FP32'}")
            print(f"Micro batch         : {B}")
            print(f"Sequence length     : {T}")
            print(f"Total batch tokens  : {total_batch_size}")
            print(f"Gradient accum steps: {grad_accum_steps}")

        # Load training and validation shards.
        train_loader = DataLoaderLite(B, T, args.data_dir, "train", rank, world_size)
        val_loader = DataLoaderLite(B, T, args.data_dir, "val", rank, world_size)

        # Build the model.
        config = GPTConfig()
        model = GPT(config).to(device)

        if ddp:
            model = DDP(model, device_ids=[local_rank])

        raw_model = model.module if ddp else model

        optimizer = raw_model.configure_optimizers(
            weight_decay=WEIGHT_DECAY,
            learning_rate=MAX_LR,
            device=device,
        )

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Resume model + optimizer + scaler state if a checkpoint was supplied.
        start_step = 0
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_step = checkpoint["step"] + 1
            if master_process:
                print(f"Resuming from step {start_step}")

        metrics_file = OUTPUT_DIR / "metrics.csv"
        if master_process and not metrics_file.exists():
            with metrics_file.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "step", "train_loss", "val_loss", "learning_rate",
                    "grad_norm", "step_ms", "tokens_per_sec", "hellaswag_accuracy",
                ])

        # Autocast only on CUDA. CPU training remains FP32.
        from contextlib import nullcontext

        def autocast_context():
            if use_amp:
                return torch.autocast(device_type="cuda", dtype=amp_dtype)
            return nullcontext()

        for step in range(start_step, args.max_steps):
            t0 = time.time()
            model.train()

            lr = get_lr(step, args.max_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)

            # Accumulate gradients to reach the desired effective batch size.
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # DDP only synchronizes gradients on the final micro-batch.
                sync_context = (
                    model.no_sync()
                    if ddp and micro_step < grad_accum_steps - 1
                    else nullcontext()
                )

                with sync_context:
                    with autocast_context():
                        logits, loss = model(x, y)

                    loss_accum += loss.detach() / grad_accum_steps
                    scaler.scale(loss / grad_accum_steps).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.time() - t0
            tokens_per_sec = (
                B * T * grad_accum_steps * world_size
            ) / max(elapsed, 1e-9)

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            train_loss = loss_accum.item()
            val_loss = float("nan")
            hellaswag_accuracy = float("nan")

            # Validation loss.
            if step % VAL_INTERVAL == 0 or step == args.max_steps - 1:
                model.eval()
                val_loss_accum = torch.zeros((), device=device)

                with torch.no_grad():
                    for _ in range(VAL_STEPS):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)

                        with autocast_context():
                            _, loss = model(x, y)

                        val_loss_accum += loss.detach() / VAL_STEPS

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                val_loss = val_loss_accum.item()
                model.train()

            # HellaSwag evaluation is optional because it can take time.
            if args.hellaswag_interval > 0 and (
                step % args.hellaswag_interval == 0 or step == args.max_steps - 1
            ):
                if master_process:
                    correct, total = evaluate_hellaswag(
                        raw_model,
                        device,
                        max_examples=args.hellaswag_max_examples,
                        amp_dtype=amp_dtype if use_amp else None,
                    )
                    hellaswag_accuracy = correct / max(total, 1)
                    print(
                        f"HellaSwag: {correct}/{total} = "
                        f"{hellaswag_accuracy:.4f}"
                    )

                if ddp:
                    score = torch.tensor(
                        hellaswag_accuracy if master_process else 0.0,
                        device=device,
                    )
                    dist.broadcast(score, src=0)
                    hellaswag_accuracy = score.item()

            # Print and save metrics only from the main process.
            if master_process:
                print(
                    f"step: {step:5d} | "
                    f"loss: {train_loss:.4f} | "
                    f"val: {val_loss:.4f} | "
                    f"lr: {lr:.2e} | "
                    f"norm: {grad_norm:.4f} | "
                    f"dt: {elapsed * 1000:.2f} ms | "
                    f"tok/sec: {tokens_per_sec:.2f}"
                )

                with metrics_file.open("a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        step, train_loss, val_loss, lr, float(grad_norm),
                        elapsed * 1000, tokens_per_sec, hellaswag_accuracy,
                    ])

                # Generate a small sample to see how the model is learning.
                if step > 0 and SAMPLE_INTERVAL > 0 and step % SAMPLE_INTERVAL == 0:
                    model.eval()
                    tokens = enc.encode("Once upon a time, in a land far away,")
                    xgen = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

                    with torch.no_grad():
                        for _ in range(30):
                            logits, _ = raw_model(xgen[:, -BLOCK_SIZE:])
                            logits = logits[:, -1, :]
                            probs = F.softmax(logits, dim=-1)
                            topk_prob, topk_idx = torch.topk(probs, k=50, dim=-1)
                            next_token = topk_idx.gather(
                                -1,
                                torch.multinomial(topk_prob, num_samples=1),
                            )
                            xgen = torch.cat((xgen, next_token), dim=1)

                    print("Generated:", enc.decode(xgen[0].tolist()))
                    model.train()

            # Save resumable checkpoints periodically.
            if SAVE_INTERVAL > 0 and (
                (step + 1) % SAVE_INTERVAL == 0 or step == args.max_steps - 1
            ):
                if master_process:
                    save_checkpoint(
                        CHECKPOINT_DIR / "latest.pt",
                        step,
                        raw_model,
                        optimizer,
                        scaler,
                        config,
                    )
                    print(f"Checkpoint saved at step {step}")

        if master_process:
            print("\nTraining complete.")
            create_training_graphs(metrics_file, OUTPUT_DIR / "plots")

    finally:
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()