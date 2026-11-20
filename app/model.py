import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import tiktoken
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, channels // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, sequence_length = idx.size()
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"the context window is {self.config.block_size}."
            )
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def _config_from_dict(values: dict[str, Any]) -> GPTConfig:
    allowed = {field.name for field in fields(GPTConfig)}
    return GPTConfig(**{key: value for key, value in values.items() if key in allowed})


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accept raw weights and the common training-checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint is not a PyTorch state dictionary.")

    for key in ("model", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    if not checkpoint or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError("No model state dictionary was found in the checkpoint.")

    cleaned = {}
    for key, value in checkpoint.items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def load_dumgpt(
    repo_or_dir: str = "shubhamisok/DumGPT-124M",
    filename: str = "DumGPT-124M-base.pth",
    device: str = "cpu",
):
    """Load DumGPT from a local directory or a Hugging Face model repository."""
    local_path = Path(repo_or_dir)
    if local_path.exists():
        weights_path = local_path / filename
        config_path = local_path / "config.json"
    else:
        weights_path = Path(hf_hub_download(repo_id=repo_or_dir, filename=filename))
        config_path = Path(hf_hub_download(repo_id=repo_or_dir, filename="config.json"))

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    config = GPTConfig()
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            config = _config_from_dict(json.load(config_file))

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = GPT(config)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    return model, tiktoken.get_encoding("gpt2"), config
