"""
HellaSwag evaluation utilities for DumGPT.

Downloads the validation set, converts each question into four candidate
completions, and scores each completion using the model's token loss.
"""

import json
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_CACHE_DIR = Path(__file__).resolve().parent / "hellaswag"

HELLASWAGS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

ENC = tiktoken.get_encoding("gpt2")


# =============================================================================
# DOWNLOAD AND DATA PREPARATION
# =============================================================================

def download(split: str) -> Path:
    """Download a HellaSwag split only if it is not already cached."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"

    if not path.exists():
        url = HELLASWAGS[split]
        print(f"Downloading {url} ...")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with path.open("wb") as file, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    return path


def render_example(example: dict):
    """
    Convert one example into tensors.

    tokens: 4 candidate sequences
    mask:   1 only over the completion tokens
    label:  index of the correct completion
    """
    context = example["ctx"]
    label = int(example["label"])

    context_tokens = ENC.encode(context)
    endings = [ENC.encode(" " + ending) for ending in example["endings"]]

    token_rows = [context_tokens + ending for ending in endings]
    max_length = max(len(row) for row in token_rows)

    tokens = torch.zeros((4, max_length), dtype=torch.long)
    mask = torch.zeros((4, max_length), dtype=torch.long)

    for i, row in enumerate(token_rows):
        tokens[i, :len(row)] = torch.tensor(row)
        mask[i, len(context_tokens):len(row)] = 1

    return tokens, mask, label


def iterate_examples(split: str = "val"):
    path = download(split)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


# =============================================================================
# COMPLETION SCORING
# =============================================================================

def get_most_likely_row(tokens, mask, logits):
    """Choose the completion with the lowest average token loss."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_tokens.reshape(-1),
        reduction="none",
    )
    losses = losses.view(tokens.size(0), -1)

    # Shift the mask too, because logits at position t predict token t+1.
    shift_mask = mask[:, 1:].contiguous()
    completion_loss = (losses * shift_mask).sum(dim=1)
    completion_length = shift_mask.sum(dim=1).clamp_min(1)

    average_loss = completion_loss / completion_length
    return int(average_loss.argmin().item())


@torch.no_grad()
def evaluate_hellaswag(model, device, max_examples=None, amp_dtype=None):
    """Evaluate model accuracy on the HellaSwag validation split."""
    was_training = model.training
    model.eval()

    num_correct = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        if max_examples is not None and i >= max_examples:
            break

        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        if device.type == "cuda" and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(tokens)
        else:
            logits, _ = model(tokens)

        prediction = get_most_likely_row(tokens, mask, logits)
        num_correct += int(prediction == label)
        num_total += 1

    if was_training:
        model.train()

    return num_correct, num_total